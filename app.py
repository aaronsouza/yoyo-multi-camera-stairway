"""Streamlit demo. Loads saved results only - never runs inference.   streamlit run app.py"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from yoyo.common import ROOT, final_dir, load_config, work

st.set_page_config(page_title="Stairway counting", layout="wide")
cfg = load_config()
fd, vd = final_dir(cfg), work(cfg) / "videos"
if not (fd / "summary.json").exists():
    st.error("No results yet. Run `python -m yoyo.cli all` first.")
    st.stop()

S = json.loads((fd / "summary.json").read_text())
events = pd.read_csv(fd / "events.csv")
groups = pd.read_csv(fd / "groups.csv")
people = pd.read_csv(fd / "people.csv")

st.title("Multi-camera stairway people & group counting")
c = st.columns(5)
c[0].metric("Unique people", S["unique_people"])
c[1].metric("Going UP", S["up"])
c[2].metric("Going DOWN", S["down"])
c[3].metric("Groups UP", S["groups_up"])
c[4].metric("Groups DOWN", S["groups_down"])
st.caption(f"Without cross-camera de-duplication the same crossings would be counted "
           f"UP {S['naive_up_sum_over_cameras']} / DOWN {S['naive_down_sum_over_cameras']} times.")

tab_v, tab_e, tab_g, tab_val, tab_f = st.tabs(["Camera video", "Crossings", "Groups", "Validation", "Floor plan"])

with tab_v:
    cams = sorted(S["per_camera"])
    cam = st.selectbox("Camera", cams)
    pc = S["per_camera"][cam]
    m = st.columns(4)
    m[0].metric("People seen here", pc["unique_people"])
    m[1].metric("UP (this camera)", pc["up"])
    m[2].metric("DOWN (this camera)", pc["down"])
    m[3].metric("Naive per-track UP/DOWN", f"{pc['naive_up']}/{pc['naive_down']}")
    vids = sorted(vd.glob(f"{cam}*.mp4")) if vd.exists() else []
    if vids:
        pick = st.selectbox("Video", [v.name for v in vids]) if len(vids) > 1 else vids[0].name
        st.video(str(vd / pick))
    else:
        st.info("No annotated video rendered for this camera yet (`python -m yoyo.cli render`).")
    st.dataframe(events[events.cams.str.split(";").apply(lambda x: cam in x)], use_container_width=True)

with tab_e:
    st.subheader("Unique crossing events (de-duplicated across cameras)")
    f = st.radio("Direction", ["ALL", "UP", "DOWN"], horizontal=True)
    st.dataframe(events if f == "ALL" else events[events.direction == f], use_container_width=True)
    st.subheader("People")
    st.dataframe(people, use_container_width=True)

with tab_g:
    st.write("Group sizes going UP:", S["group_sizes_up"] or "none", " | DOWN:", S["group_sizes_down"] or "none")
    sizes = groups[groups.direction != "NONE"].groupby(["direction", "size"]).size().rename("groups").reset_index()
    if len(sizes):
        st.bar_chart(sizes.pivot(index="size", columns="direction", values="groups").fillna(0))
    st.dataframe(groups, use_container_width=True)

with tab_val:
    vp = fd / "validation.json"
    if vp.exists():
        v = json.loads(vp.read_text())
        st.json({k: x for k, x in v.items() if k not in ("false_positives", "missed")})
        st.write("False positives", pd.DataFrame(v["false_positives"]))
        st.write("Missed crossings", pd.DataFrame(v["missed"]))
    else:
        st.info("Run `python -m yoyo.cli validate --gt ground_truth.csv --start-s A --end-s B`.")

with tab_f:
    img = fd / "floor_tracks.png"
    st.image(str(img)) if img.exists() else st.info("Bonus: calibration not run (`python -m yoyo.cli calibrate`).")
