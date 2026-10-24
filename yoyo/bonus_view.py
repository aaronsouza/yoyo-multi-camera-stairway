"""Streamlit page of the bonus: floor plan, calibration status, stitched person trajectories with replay, camera views at a chosen time.
Reads saved results only (calibration/calibration.json, results/calibration/*); never runs a detector. Camera frames need the original videos."""
import json

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .calibrate import CAL_DIR, Calibration, out_dir
from .common import ROOT, cam_dir, videos

CAM_COLORS = {"cam1": "#1f77b4", "cam2": "#ff7f0e", "cam3": "#2ca02c", "cam4": "#9467bd", "cam5": "#d62728", "cam6": "#8c564b", "cam7": "#17becf"}


@st.cache_data
def _read(path, mtime):
    p = str(path)
    return pd.read_csv(p)


def _csv(path):
    return _read(str(path), path.stat().st_mtime) if path.exists() else pd.DataFrame()


@st.cache_data
def _json(path, mtime):
    return json.loads(open(path, encoding="utf-8").read())


def _j(path):
    return _json(str(path), path.stat().st_mtime) if path.exists() else {}


def floor_figure(cal, meta, line=None, cams_on=(), show_stairs=True, mosaic=True, frames=True, step_s=1.0):
    """Interactive top-down plan: ground mosaic, camera positions and coverage, and the trajectory of one person with a replay animation."""
    ext = meta["extent"]
    fig = go.Figure()
    img = out_dir_img()
    if mosaic and img:
        fig.add_layout_image(dict(source=img, xref="x", yref="y", x=ext[0], y=ext[3], sizex=ext[1] - ext[0], sizey=ext[3] - ext[2],
                                  sizing="stretch", opacity=0.5, layer="below"))
    for c in cams_on:
        col = CAM_COLORS[c]
        for kind, alpha, dash in (("approximate", 0.07, "dot"), ("reliable", 0.16, "solid")):
            for poly in meta["coverage"].get(c, {}).get(kind, []):
                p = np.array(poly + [poly[0]])
                fig.add_trace(go.Scatter(x=p[:, 0], y=p[:, 1], mode="lines", fill="toself", line=dict(color=col, width=1, dash=dash),
                                         fillcolor=_rgba(col, alpha), name=f"{c} {kind}", legendgroup=c, showlegend=False,
                                         hoverinfo="text", text=f"{c}: {kind} coverage"))
    for c in cal.cams:
        e = cal.cal["cameras"][c]
        x, y = e["position_m"]
        h = np.radians(e["heading_deg"])
        fig.add_trace(go.Scatter(x=[x, x + 3.5 * np.cos(h)], y=[y, y + 3.5 * np.sin(h)], mode="lines", line=dict(color=CAM_COLORS[c], width=3), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[c], textposition="bottom center", marker=dict(symbol="triangle-up", size=14, color=CAM_COLORS[c],
                                 line=dict(color="black", width=1)), name=c, hovertext=f"{c}: height {e['height_m']:.1f} m, looking {e['heading_deg']:.0f} deg", hoverinfo="text"))
    if show_stairs and meta.get("stairs_base"):
        b = np.array(meta["stairs_base"])
        fig.add_trace(go.Scatter(x=b[:, 0], y=b[:, 1], mode="lines", line=dict(color="black", width=4), name="foot of the staircase (from cam5; the stairs themselves are not mapped)"))
    n_static = len(fig.data)
    if line is not None and len(line):
        _add_trajectory(fig, line, frames, step_s)
    fig.update_xaxes(range=[ext[0], ext[1]], title="metres (x)", constrain="domain")
    fig.update_yaxes(range=[ext[2], ext[3]], title="metres (y)", scaleanchor="x", scaleratio=1)
    fig.update_layout(height=720, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.12), plot_bgcolor="#f4f4f4", dragmode="pan")
    return fig


def _rgba(hex_color, a):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _add_trajectory(fig, line, frames, step_s):
    """Observed segments as solid lines, unobserved gaps as thin dotted lines, direction arrows, start/end, and a replay animation."""
    line = line.sort_values("t").reset_index(drop=True)
    first = len(fig.data)
    for s, g in line.groupby("segment"):
        fig.add_trace(go.Scatter(x=g.X, y=g.Y, mode="lines+markers", line=dict(color="#e41a1c", width=3), marker=dict(size=4, color=np.where(g.quality == "reliable", "#e41a1c", "white"),
                                 line=dict(color="#e41a1c", width=1)), customdata=np.c_[g.t, g.cams, g.quality], hovertemplate="t=%{customdata[0]:.1f}s<br>%{customdata[1]}<br>%{customdata[2]}<extra></extra>",
                                 name="observed", legendgroup="obs", showlegend=bool(s == line.segment.min())))
        sub = g.iloc[::max(len(g) // 8, 1)]
        if len(sub) > 1:
            fig.add_trace(go.Scatter(x=sub.X, y=sub.Y, mode="markers", marker=dict(symbol="arrow", size=12, angleref="previous", color="#e41a1c"), hoverinfo="skip", showlegend=False))
    segs = [g for _, g in line.groupby("segment")]
    for a, b in zip(segs, segs[1:]):
        p, q = a.iloc[-1], b.iloc[0]
        fig.add_trace(go.Scatter(x=[p.X, q.X], y=[p.Y, q.Y], mode="lines", line=dict(color="#888", width=1.5, dash="dot"), name="unobserved gap (straight line is NOT data)",
                                 legendgroup="gap", showlegend=bool(a is segs[0]), hovertext=f"unobserved for {q.t - p.t:.1f} s", hoverinfo="text"))
    fig.add_trace(go.Scatter(x=[line.X.iloc[0]], y=[line.Y.iloc[0]], mode="markers", marker=dict(size=13, color="#2ca02c", line=dict(color="black", width=1)), name="first seen"))
    fig.add_trace(go.Scatter(x=[line.X.iloc[-1]], y=[line.Y.iloc[-1]], mode="markers", marker=dict(size=13, symbol="square", color="#111"), name="last seen"))
    if not frames:
        return
    ts = np.arange(line.t.iloc[0], line.t.iloc[-1] + step_s, step_s)
    if len(ts) > 900:
        ts = np.linspace(line.t.iloc[0], line.t.iloc[-1], 900)
    cur = len(fig.data)
    start = line.iloc[0]
    fig.add_trace(go.Scatter(x=[start.X], y=[start.Y], mode="markers", marker=dict(size=18, symbol="star", color="gold", line=dict(color="black", width=1.5)), name="replay position"))
    fr = []
    for t in ts:
        i = int(np.abs(line.t.values - t).argmin())
        seen = abs(line.t.values[i] - t) <= 1.0
        r = line.iloc[i]
        fr.append(go.Frame(name=f"{t:.0f}", traces=[cur], data=[go.Scatter(x=[r.X if seen else None], y=[r.Y if seen else None],
                                                                              hovertext=f"t={t:.0f}s {'observed by ' + r.cams if seen else 'not observed'}", hoverinfo="text")]))
    fig.frames = fr
    fig.update_layout(updatemenus=[dict(type="buttons", x=0.0, y=1.08, direction="left", buttons=[
        dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=120, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
        dict(label="⏸ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=0, x=0.12, len=0.88, currentvalue=dict(prefix="video time (s): "), steps=[dict(label=f.name, method="animate", args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) for f in fr])])


def out_dir_img():
    p = ROOT / "results" / "calibration" / "floorplan.png"
    if not p.exists():
        return None
    from PIL import Image
    return Image.open(p)


def camera_frame(cfg, cam, tid, t):
    """The camera's frame at video time t with the person's box drawn, or None when the video is not on this machine."""
    try:
        v = videos(cfg)[cam]
        tr = _csv(cam_dir(cfg, cam) / "tracks.csv")
    except Exception:
        return None
    g = tr[tr.tid == tid]
    if g.empty:
        return None
    r = g.iloc[(g.t - t).abs().argsort().iloc[0]]
    if abs(r.t - t) > 1.0:
        return None
    cap = cv2.VideoCapture(str(v))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    cv2.rectangle(fr, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (0, 0, 255), 5)
    return cv2.cvtColor(cv2.resize(fr, (640, 360)), cv2.COLOR_BGR2RGB)


def render(cfg):
    st.header("Multi-Camera Calibration & Track Stitching (Bonus)")
    if not (CAL_DIR / "calibration.json").exists() or not (out_dir(cfg) / "trajectories.csv.gz").exists():
        st.info("Not computed yet: `python -m yoyo.cli calibrate` then `python -m yoyo.cli stitch`.")
        return
    cal = Calibration()
    meta = _j(out_dir(cfg) / "floorplan.json")
    val = _j(CAL_DIR / "validation.json")
    idx = _csv(out_dir(cfg) / "lines_index.csv")
    traj_path = out_dir(cfg) / "trajectories.csv.gz"
    st.warning("**Approximate calibration.** One focal length for all cameras and each camera's ground plane come from the paving seams; cameras are tied "
               "together by synchronised foot points (no ReID). Floor positions are on the plaza plane in metres (scale from adult height, about ±15%). "
               "The staircase is not mapped. Dotted outlines on the plan = approximate zones; nothing beyond them is shown.")
    tab_plan, tab_cal, tab_stitch = st.tabs(["Floor plan & trajectories", "Calibration status", "Stitching quality"])

    with tab_plan:
        left, right = st.columns([3, 1])
        with right:
            st.subheader("Find a person")
            multi = idx[(idx.n_cameras >= 2) & (idx.duration_s >= 10)].sort_values("duration_s", ascending=False)
            mode = st.radio("Search by", ["Pick from list", "Type a global ID"], horizontal=True)
            if mode == "Type a global ID":
                gid = st.number_input("Global person ID (gid)", min_value=1, value=int(multi.gid.iloc[0]) if len(multi) else 1, step=1)
                cand = idx[idx.gid == gid]
                if cand.empty:
                    st.info("This ID has no trajectory on the floor plan (never seen on the plaza plane).")
                    return_line = None
                    cand_ids = []
                else:
                    cand_ids = list(cand.line_id)
            else:
                only_multi = st.checkbox("Only people seen by 2+ cameras", True)
                pool = multi if only_multi else idx.sort_values("duration_s", ascending=False)
                cand_ids = list(pool.line_id.head(300))
            line_id = st.selectbox("Trajectory", cand_ids, format_func=lambda i: _label(idx, i)) if cand_ids else None
            cams_on = st.multiselect("Show camera coverage", cal.cams, default=[])
            mosaic = st.checkbox("Ground-texture background", True)
            if line_id:
                row = idx[idx.line_id == line_id].iloc[0]
                st.caption(f"Cameras: {row.cameras}  \nSeen {row.t0:.0f}-{row.t1:.0f} s of the video ({row.duration_s:.0f} s), walked {row.path_m:.0f} m, "
                           f"{int(row.segments)} observed segment(s). " + ("The global ID had to be split: it mixed several people." if row.id_was_split else ""))
        with left:
            line = None
            if line_id:
                traj = _read_traj(str(traj_path), traj_path.stat().st_mtime, line_id)
                line = traj
            fig = floor_figure(cal, meta, line, cams_on, mosaic=mosaic)
            st.plotly_chart(fig, width="stretch")
            st.caption("Solid red line = observed positions (filled dot = reliable, open dot = approximate). Dotted grey line = the person was not observed there; "
                       "the straight line is only a hint, not data. ★ = replay position (press ▶ Play or drag the slider).")
        if line_id:
            st.subheader("Camera views at one moment")
            t = st.slider("Video time (s)", float(line.t.min()), float(line.t.max()), float(line.t.min()), step=0.5)
            members = _csv(out_dir(cfg) / "line_members.csv")
            mem = members[(members.line_id == line_id) & (members.t0 <= t + 1) & (members.t1 >= t - 1)]
            cols = st.columns(max(len(mem), 1))
            shown = 0
            for col, r in zip(cols, mem.itertuples()):
                im = camera_frame(cfg, r.cam, int(r.tid), t)
                if im is not None:
                    col.image(im, caption=f"{r.cam} at {t:.1f} s (red box = this person)")
                    shown += 1
            if not shown:
                st.info("No camera saw this person at that moment" if len(mem) == 0 else "Camera frames need the original videos on this machine.")

    with tab_cal:
        st.subheader("Cameras on the floor plan")
        rows = []
        for c in cal.cams:
            e = cal.cal["cameras"][c]
            pc = val.get("per_camera", {}).get(c, {})
            rows.append(dict(camera=c, x_m=round(e["position_m"][0], 1), y_m=round(e["position_m"][1], 1), looking_deg=round(e["heading_deg"]), height_m=round(e["height_m"], 1),
                             seam_fit_error_deg=round(pc.get("seam_median_error_deg", np.nan), 2), seam_share=round(pc.get("seam_inlier_length_share", np.nan), 2),
                             vote_evidence=(f"{pc['vote_evidence']:.2f}" if pc.get("vote_evidence") else "reference camera")))
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"Shared focal length {cal.cal['focal_px']:.0f} px. Seam fit error = median angle between a seam line and the fitted ground direction. "
                   "Vote evidence = height of the best translation vote over its 99th percentile (about 1 = no overlap found; 2.7+ = clear).")
        ls = val.get("landmark_stats")
        if ls:
            c1, c2, c3 = st.columns(3)
            c1.metric("Landmark agreement, rms", f"{ls['rms_m']:.2f} m")
            c2.metric("median", f"{ls['median_m']:.2f} m")
            c3.metric("worst", f"{ls['max_m']:.2f} m")
            st.caption(f"{ls['n_observations']} observations of 3 static landmarks read by eye; none was used to fit the calibration. "
                       "A fourth landmark (tent corner) was dropped because its ground contact is hidden in all views.")
            lm = [dict(landmark=k, cameras=v["cameras"], max_spread_m=round(v["max_pair_spread_m"], 1), used=v["used"], note=v["note"]) for k, v in val["landmarks"].items()]
            st.dataframe(pd.DataFrame(lm), hide_index=True, width="stretch")
        tb = val.get("time_shift_baseline")
        if tb:
            st.subheader("Do cameras really agree on where people are?")
            d = pd.DataFrame(tb)
            st.dataframe(d.rename(columns=dict(pair="camera pair", real="matches within 1 m, same instant", shifted="same, at instants 7-20 s apart", excess="excess", real_share="share that is real")),
                         hide_index=True, width="stretch")
            st.caption("If two cameras were registered wrongly there would be no excess. Every pair shows one; crowds make chance matches common, so the share is modest.")

    with tab_stitch:
        st.subheader("Trajectories")
        n_multi = int((idx.n_cameras >= 2).sum())
        a, b, c_ = st.columns(3)
        a.metric("Trajectories", len(idx))
        b.metric("Seen by 2+ cameras", n_multi)
        c_.metric("Global IDs split (mixed people)", int(idx.groupby("gid").id_was_split.first().sum()))
        conf = _csv(out_dir(cfg) / "identity_conflicts.csv")
        gaps = _csv(out_dir(cfg) / "gaps.csv")
        if len(gaps):
            st.write("Gaps inside trajectories (never interpolated):", gaps.kind.value_counts().to_dict())
        rv = _j(CAL_DIR / "stitch_review_summary.json")
        if rv:
            st.subheader("Checked against manually reviewed identities")
            st.dataframe(pd.DataFrame(rv["rounds"]), hide_index=True, width="stretch")
            st.caption(rv["note"])
        if len(conf):
            with st.expander("Why global IDs were split"):
                st.dataframe(conf.head(200), hide_index=True, width="stretch")
    st.subheader("30-second proof")
    ev = sorted((ROOT / "evidence" / "bonus").glob("*.mp4")) if (ROOT / "evidence" / "bonus").exists() else []
    if ev:
        st.video(str(ev[0]))
        st.caption(f"`{ev[0].name}`: two synchronized cameras and the same person's trajectory on the floor plan.")


def _label(idx, i):
    r = idx[idx.line_id == i].iloc[0]
    return f"{i}  |  {r.cameras}  |  {r.duration_s:.0f}s, {r.path_m:.0f} m"


@st.cache_data
def _traj_all(path, mtime):
    return pd.read_csv(path)


def _read_traj(path, mtime, line_id):
    d = _traj_all(path, mtime)
    return d[d.line_id == line_id].reset_index(drop=True)
