"""Stage 4: completed stair crossings via a two-zone state machine.

A crossing is counted only when a person (global identity) was seen dwelling in one end zone and later dwelling in
the other end zone within `max_cross_s`. Stopping, turning around mid-stairs or only touching one zone gives no event.
Visits from all cameras are merged per identity, so one crossing seen by three cameras is one event.
"""
import numpy as np
import pandas as pd
from matplotlib.path import Path as MPath


def zone_labels(df, W, H, zones):
    """Label each detection by the zone containing its foot point ('' = none). zones: name -> normalised polygon."""
    pts = np.c_[(df.x1 + df.x2) / 2, df.y2]
    lab = np.full(len(df), "", dtype=object)
    for name, poly in zones.items():
        lab[MPath(np.array(poly) * [W, H]).contains_points(pts)] = name
    return lab


def visits_for_track(t, zone, cam, gap_tol, min_rows):
    out, cur = [], None
    for ti, z in zip(t, zone):
        if cur and z == cur["zone"]:
            cur["t1"] = ti
            cur["n"] += 1
        elif cur and z == "" and ti - cur["t1"] <= gap_tol:
            continue  # brief dropout / walking between zone edge pixels
        else:
            if cur:
                out.append(cur)
            cur = dict(zone=z, t0=ti, t1=ti, n=1, cams={cam}) if z else None
    if cur:
        out.append(cur)
    return [v for v in out if v["n"] >= min_rows]


def transitions(visits, max_cross, min_cross=0.0):
    col = []
    for v in sorted(visits, key=lambda v: (v["t0"], v["t1"])):
        if col and col[-1]["zone"] == v["zone"]:
            c = col[-1]
            c["t1"] = max(c["t1"], v["t1"])
            c["cams"] = c["cams"] | v["cams"]
        else:
            col.append(dict(v, cams=set(v["cams"])))
    ev = []
    for a, b in zip(col, col[1:]):
        if min_cross <= b["t0"] - a["t1"] <= max_cross:
            ev.append(dict(direction="UP" if a["zone"] == "bottom" else "DOWN",
                           t_start=a["t1"], t_end=b["t0"], cams=a["cams"] | b["cams"]))
    return ev


def path_ok(rows, ev, e, fps_hint=10.0):
    """A crossing must be *seen*: dense observations between the zones, no position jumps, net motion in the right way."""
    gap = ev["t_end"] - ev["t_start"]
    m = rows[(rows.t >= ev["t_start"] - 1e-6) & (rows.t <= ev["t_end"] + 1e-6)]
    if len(m) < 3 or len(m) < e.get("min_coverage", 0.5) * gap * fps_hint:
        return False
    step = np.hypot(np.diff(m.fx.values), np.diff(m.fy.values)) / m.h.values[1:]
    if step.max() > e.get("max_jump_heights", 1.0):
        return False
    dy = m.fy.values[-1] - m.fy.values[0]
    return dy < 0 if ev["direction"] == "UP" else dy > 0


def merge_across_cameras(ev, window):
    """Crossings seen by several cameras for one identity (same direction, close in time) are one event."""
    out = []
    for (gid, d), g in ev.sort_values("t_end").groupby(["gid", "direction"]):
        cur = None
        for r in g.itertuples():
            if cur and r.t_end - cur["t_end"] <= window:
                cur["cams"] |= set(r.cams.split(";"))
                cur["t_start"], cur["t_end"] = min(cur["t_start"], r.t_start), max(cur["t_end"], r.t_end)
            else:
                if cur:
                    out.append(cur)
                cur = dict(gid=gid, direction=d, t_start=r.t_start, t_end=r.t_end, cams=set(r.cams.split(";")))
        if cur:
            out.append(cur)
    res = pd.DataFrame(out, columns=["gid", "direction", "t_start", "t_end", "cams"])
    res["cams"] = res.cams.apply(lambda c: ";".join(sorted(c)))
    return res


def compute_events(tracks, tracklets, zones, meta, cfg):
    """tracks: cam -> DataFrame; tracklets: cam,tid,gid; returns (events, naive per-camera counts)."""
    e, off = cfg["events"], cfg.get("offsets_s", {})
    per_gid, naive, rows_by_key = {}, [], {}
    gid_of = {(r.cam, r.tid): r.gid for r in tracklets.itertuples()}
    for cam, df in tracks.items():
        if cam not in zones or df.empty:
            continue
        df = df.sort_values(["tid", "t"]).copy()
        df["zone"] = zone_labels(df, meta[cam]["width"], meta[cam]["height"], zones[cam])
        for tid, g in df.groupby("tid"):
            v = visits_for_track(g.t.values + off.get(cam, 0), g.zone.values, cam, e["gap_tol_s"], e["min_dwell_rows"])
            for ev in transitions(v, e["max_cross_s"], e.get("min_cross_s", 0)):  # per-tracklet, no cross-camera dedup
                naive.append(dict(cam=cam, direction=ev["direction"]))
            if (cam, tid) in gid_of:
                rows_by_key.setdefault((gid_of[(cam, tid)], cam) if e.get("per_camera", True) else (gid_of[(cam, tid)], ""), []).append(
                    pd.DataFrame(dict(t=g.t.values + off.get(cam, 0), fx=(g.x1 + g.x2).values / 2, fy=g.y2.values, h=(g.y2 - g.y1).values)))
                key = (gid_of[(cam, tid)], cam) if e.get("per_camera", True) else (gid_of[(cam, tid)], "")
                per_gid.setdefault(key, []).extend(v)
    rows = []
    for key, v in per_gid.items():
        gid = key[0]
        path = pd.concat(rows_by_key[key]).sort_values("t") if e.get("verify_path", True) else None
        for ev in transitions(v, e["max_cross_s"], e.get("min_cross_s", 0)):
            if path is not None and not path_ok(path, ev, e):
                continue
            rows.append(dict(gid=gid, direction=ev["direction"], t_start=round(ev["t_start"], 2),
                             t_end=round(ev["t_end"], 2), cams=";".join(sorted(ev["cams"]))))
    ev = pd.DataFrame(rows, columns=["gid", "direction", "t_start", "t_end", "cams"])
    ev = merge_across_cameras(ev, e.get("merge_window_s", 6.0))
    ev = ev.sort_values("t_start").reset_index(drop=True)
    ev.insert(0, "event_id", np.arange(1, len(ev) + 1))
    return ev, pd.DataFrame(naive, columns=["cam", "direction"])
