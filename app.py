"""Streamlit demo. Loads saved results and saved sample videos only - never runs inference.   streamlit run app.py"""
import json
import re

import pandas as pd
import streamlit as st

from yoyo.common import ROOT, final_dir, load_config

st.set_page_config(page_title="Stairway counting", layout="wide")
cfg = load_config()
fd = final_dir(cfg)
EV = ROOT / "evidence"
if not (fd / "summary.json").exists():
    st.error("No results yet. Run `python -m yoyo.cli all` first.")
    st.stop()


@st.cache_data
def load(path, mtime):  # mtime in the key: re-read when the file changes
    return pd.read_csv(path)


def table(name):
    p = fd / name
    return load(str(p), p.stat().st_mtime) if p.exists() else pd.DataFrame()


S = json.loads((fd / "summary.json").read_text())
events, groups, people, standing = table("events.csv"), table("groups.csv"), table("people.csv"), table("standing_groups.csv")


def clip(folder, pattern):
    d = EV / folder
    return sorted(d.glob(pattern)) if d.exists() else []


def play(path, caption):
    st.video(str(path))
    st.caption(f"`{path.name}` - {caption}")


def interval(path):
    m = re.search(r"(\d+)-(\d+)s", path.name)
    return f"{int(m[1]) // 60:02d}:{int(m[1]) % 60:02d}-{int(m[2]) // 60:02d}:{int(m[2]) % 60:02d} of the video" if m else ""


st.title("Multi-camera stairway people & group counting")
c = st.columns(5)
c[0].metric("Unique people", S["unique_people"])
c[1].metric("Going UP", S["up"])
c[2].metric("Going DOWN", S["down"])
c[3].metric("Groups UP", S["groups_up"])
c[4].metric("Groups DOWN", S["groups_down"])

tab_p, tab_r, tab_b = st.tabs(["Proof of concept", "Results", "Multi-Camera Calibration & Track Stitching (Bonus)"])

with tab_p:
    st.info("Every video here is a **30-second proof-of-concept sample** cut from the original footage and drawn from the saved results "
            "(no model is run in this app). They are not the full 35-minute videos.")

    st.header("Overall proof of concept - Requirements 1 to 4 together")
    ov = clip("overview_1-4", "*.mp4")
    if ov:
        st.write(f"**Separate video: all four requirements in one clip.** Cam5 (left) and Cam2 (right), synchronized, {interval(ov[0])}. "
                 "Top bar counts tracking, crossings, cross-camera duplicates removed and groups.")
        play(ov[0], "")
        e1, g1 = clip("overview_1-4", "*_events.csv"), clip("overview_1-4", "*_groups.csv")
        with st.expander("Events and groups drawn in this clip"):
            if e1:
                st.dataframe(pd.read_csv(e1[0]), width="stretch")
            if g1:
                st.dataframe(pd.read_csv(g1[0]), width="stretch")
    else:
        st.warning("Overview clip missing: `python tools/make_overview_evidence.py cam5 cam2 1515`")

    st.divider()
    st.header("Proof of concept 1 - Person detection and tracking")
    st.write("**Separate video per camera.** Every person has a box with tracker id (T) and global person id (P); camera name and video "
             "time are shown. Choose a camera:")
    cl = {p.name.split("_")[0]: p for p in clip("checklist1_tracking", "cam*.mp4")}
    if cl:
        pick = st.selectbox("Camera (proof of concept 1)", sorted(cl), key="poc1")
        play(cl[pick], f"{interval(cl[pick])}; the busiest 30 s of that camera")

    st.divider()
    st.header("Proof of concept 2 - Completed UP/DOWN crossings")
    st.write("**Separate video.** Green = UP, red = DOWN, with event id, the path walked as an arrow, and `done` when the crossing is complete. "
             "People who only stand or turn around get nothing.")
    for p in clip("checklist2_crossings", "*.mp4"):
        play(p, f"Cam5, {interval(p)}")

    st.divider()
    st.header("Proof of concept 3 - Same person / same crossing across cameras counted once")
    st.write("**Separate video.** Cam5 and Cam2 side by side at the same video time. A crossing seen by both cameras has the same colour and "
             "`MATCH` on both views and is counted once.")
    for p in clip("checklist3_dedup", "*.mp4"):
        play(p, f"Cam5 + Cam2, {interval(p)}")

    st.divider()
    st.header("Proof of concept 4 - Group detection and counting")
    st.write("**Separate video.** Coloured box `GROUP <id> x<size> UP/DOWN` around people who cross together; a dashed `standing xN` box marks "
             "people standing together (not counted).")
    for p in clip("checklist4_groups", "*.mp4"):
        play(p, f"Cam5, {interval(p)}")

def manual_comparison():
    """Cam5 25:00-27:00 (1500-1620 s): the manual counts against the saved results; everything on the model side is computed here."""
    w = events[events.cams.str.split(";").apply(lambda c: "cam5" in c) & events.t_end.between(1500, 1620)]
    gi = {int(i): g for g in groups.itertuples() for i in str(g.event_ids).split(";")}
    sizes = {d: sorted(g.size for g in {gi[i].group_id: gi[i] for i in w[w.direction == d].event_id if i in gi}.values()) for d in ("UP", "DOWN")}
    solo = {d: int(((w.direction == d) & w.entity.str.startswith("S")).sum()) for d in ("UP", "DOWN")}
    ngr = {d: len(sizes[d]) for d in sizes}
    pers = {d: int((w.direction == d).sum()) for d in sizes}
    ent = {d: solo[d] + ngr[d] for d in sizes}
    man = dict(pers=(12, 15), solo=(5, 7), ngr=(3, 3), ent=(8, 10))
    man_sizes = {"UP": [2, 2, 3], "DOWN": [2, 2, 4]}
    pct = lambda a, b: f"{100 * a / b:.0f}%"
    fmt = lambda n, sz: f"{n} (sizes {', '.join(map(str, sz))})"
    rows = [("Persons UP", 12, pers["UP"], pct(pers["UP"], 12)), ("Persons DOWN", 15, pers["DOWN"], pct(pers["DOWN"], 15)),
            ("Persons total", 27, pers["UP"] + pers["DOWN"], pct(pers["UP"] + pers["DOWN"], 27)),
            ("Solo UP", 5, solo["UP"], pct(solo["UP"], 5)), ("Solo DOWN", 7, solo["DOWN"], pct(solo["DOWN"], 7)),
            ("Groups UP", fmt(3, man_sizes["UP"]), fmt(ngr["UP"], sizes["UP"]), "exact" if sizes["UP"] == man_sizes["UP"] else "count exact, sizes differ" if ngr["UP"] == 3 else "differs"),
            ("Groups DOWN", fmt(3, man_sizes["DOWN"]), fmt(ngr["DOWN"], sizes["DOWN"]), "exact" if sizes["DOWN"] == man_sizes["DOWN"] else "count exact, sizes differ" if ngr["DOWN"] == 3 else "differs"),
            ("Entities UP (group = 1)", 8, ent["UP"], pct(ent["UP"], 8)), ("Entities DOWN (group = 1)", 10, ent["DOWN"], pct(ent["DOWN"], 10)),
            ("Entities total", 18, ent["UP"] + ent["DOWN"], pct(ent["UP"] + ent["DOWN"], 18))]
    return pd.DataFrame(rows, columns=["Metric", "Manual", "Final", "Agreement"]).astype(str)


with tab_r:
    st.subheader("Ground truth comparison")
    st.caption("Cam5, 25:00-27:00: manual count vs result.")
    st.table(manual_comparison().set_index("Metric"))
    st.caption("Count agreement (result / manual), not event-level matching.")

    st.subheader("Unique crossing events (de-duplicated across cameras)")
    f = st.radio("Direction", ["ALL", "UP", "DOWN"], horizontal=True)
    st.dataframe(events if f == "ALL" else events[events.direction == f], width="stretch")

    st.subheader("Groups")
    fmt_sizes = lambda d: ", ".join(f"{n} groups of {k}" for k, n in sorted(d.items(), key=lambda x: int(x[0]))) or "none"
    st.write(f"**Group sizes going UP:** {fmt_sizes(S['group_sizes_up'])}")
    st.write(f"**Group sizes going DOWN:** {fmt_sizes(S['group_sizes_down'])}")
    sizes = groups[groups.direction != "NONE"].groupby(["direction", "size"]).size().rename("groups").reset_index()
    if len(sizes):
        st.bar_chart(sizes.pivot(index="size", columns="direction", values="groups").fillna(0))
    st.dataframe(groups, width="stretch")
    st.subheader("Standing groups (not counted)")
    st.dataframe(standing, width="stretch")

with tab_b:
    from yoyo import bonus_view
    bonus_view.render(cfg)
