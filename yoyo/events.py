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


def fast_jump(m, max_speed):
    """True when two consecutive observations imply a speed above `max_speed` body heights per second."""
    dt = np.diff(m.t.values)
    ok = dt > 0
    speed = np.hypot(np.diff(m.fx.values), np.diff(m.fy.values))[ok] / m.h.values[1:][ok] / dt[ok]
    return bool(len(speed) and speed.max() > max_speed)


def path_ok(rows, ev, e, fps_hint=10.0):
    """A crossing must be *seen*: dense observations between the zones, no position jumps, net motion in the right way."""
    pad = e.get("path_pad_s", 0.0)  # widened zones leave a short gap, so judge the path over the visits' edges too
    gap = ev["t_end"] - ev["t_start"] + 2 * pad
    m = rows[(rows.t >= ev["t_start"] - pad - 1e-6) & (rows.t <= ev["t_end"] + pad + 1e-6)]
    if len(m) < 3 or len(m) < e.get("min_coverage", 0.5) * gap * fps_hint:
        return False
    step = np.hypot(np.diff(m.fx.values), np.diff(m.fy.values)) / m.h.values[1:]
    if step.max() > e.get("max_jump_heights", 1.0):
        return False
    if e.get("max_speed_h_per_s") and fast_jump(m, e["max_speed_h_per_s"]):
        return False  # a body-height jump in a fraction of a second is a swapped track ID, not a walker
    dy = m.fy.values[-1] - m.fy.values[0]
    if abs(dy) < e.get("min_net_dy", 0.0) * e.get("frame_h", 1.0):
        return False  # hovering at the stair, not travelling along it
    return dy < 0 if ev["direction"] == "UP" else dy > 0


def edge_events(g, visits, W, H, e, off=0.0):
    """Inferred crossings for people who leave/enter through the left or right frame edge on the stairs.

    DOWN: dwelt in the top zone, then walked steadily down the stairs and left the frame at an edge before the bottom
    zone. UP is the mirror image (entered at an edge, reached the top zone). Needs a real top-zone dwell, a net foot-y
    change of at least `min_rise`, a monotone dense path and no bottom visit (those are normal crossings)."""
    out, m = [], e["margin"]
    t = g.t.values + off
    fx, fy, h = (g.x1 + g.x2).values / 2 / W, g.y2.values / H, (g.y2 - g.y1).values / H
    near = (g.x1.values <= m * W) | (g.x2.values >= (1 - m) * W)
    tops = [v for v in visits if v["zone"] == "top"]

    def ok(i0, i1, down):
        seg = slice(min(i0, i1), max(i0, i1) + 1)
        dur = abs(t[i1] - t[i0])
        n = len(t[seg])
        if not e["min_dur_s"] <= dur <= e["max_dur_s"] or n < max(3, 0.5 * dur * 10):
            return False
        f = fy[seg] if i1 > i0 else fy[seg][::-1]  # chronological
        rise = f[-1] - f[0]
        if (rise if down else -rise) < e["min_rise"]:
            return False
        back = (np.maximum.accumulate(f) - f) if down else (f - np.minimum.accumulate(f))
        if back.max() > e.get("max_backtrack", 0.04):
            return False
        step = np.hypot(np.diff(fx[seg]), np.diff(fy[seg])) / np.maximum(h[seg][1:], 1e-6)
        if e.get("max_speed_h_per_s") and len(step) and (step / np.maximum(np.diff(t[seg]), 1e-6)).max() > e["max_speed_h_per_s"]:
            return False
        return step.size == 0 or step.max() <= e.get("max_jump_heights", 1.0)

    if tops and near[-1]:
        tv = tops[-1]
        if True:
            lo = int(np.searchsorted(t, tv["t1"] - e.get("ref_s", 3.0) - 1e-6))
            hi = max(int(np.searchsorted(t, tv["t1"] - 1e-6)), lo)
            i0 = min(lo + int(np.argmin(fy[lo:hi + 1])), len(t) - 1)  # highest point reached on the stairs just before leaving
            if ok(i0, len(t) - 1, True):
                out.append(dict(direction="DOWN", t_start=float(t[i0]), t_end=float(t[-1]), rise=float(fy[-1] - fy[i0])))
    if tops and near[0]:
        tv = tops[0]
        if True:
            lo = min(int(np.searchsorted(t, tv["t0"] - 1e-6)), len(t) - 1)
            hi = min(int(np.searchsorted(t, tv["t0"] + e.get("ref_s", 3.0) + 1e-6)), len(t) - 1)
            i1 = lo + int(np.argmin(fy[lo:hi + 1]))
            if ok(0, i1, False):
                out.append(dict(direction="UP", t_start=float(t[0]), t_end=float(t[i1]), rise=float(fy[0] - fy[i1])))
    return out


def drop_flips(rows):
    """One identity cannot walk the stairs one way and back within seconds: that is jitter near a zone border.
    Of consecutive opposite-direction events of an identity within `flip` seconds keep the one that travelled further."""
    out, by = [], {}
    for r in rows:
        by.setdefault((r["gid"], r["cams"]), []).append(r)
    for g in by.values():
        g = sorted(g, key=lambda r: r["t_end"])
        i = 0
        while i < len(g) - 1:
            a, b = g[i], g[i + 1]
            if a["flip"] and a["direction"] != b["direction"] and b["t_start"] - a["t_end"] <= a["flip"]:
                g.pop(i if a["dy"] < b["dy"] else i + 1)
                i = max(i - 1, 0)
            else:
                i += 1
        out.extend(g)
    return [{k: v for k, v in r.items() if k not in ("dy", "flip")} for r in out]


def merge_across_cameras(ev, window):
    """Crossings seen by several cameras for one identity (same direction, close in time) are one event.
    Two events seen by the same camera are two crossings, however close in time."""
    out = []
    for (gid, d), g in ev.sort_values("t_end").groupby(["gid", "direction"]):
        cur = None
        for r in g.itertuples():
            if cur and r.t_end - cur["t_end"] <= window and not (cur["cams"] & set(r.cams.split(";"))):
                cur["cams"] |= set(r.cams.split(";"))
                cur["inferred"] = cur["inferred"] and r.inferred
                cur["t_start"], cur["t_end"] = min(cur["t_start"], r.t_start), max(cur["t_end"], r.t_end)
            else:
                if cur:
                    out.append(cur)
                cur = dict(gid=gid, direction=d, t_start=r.t_start, t_end=r.t_end, cams=set(r.cams.split(";")), inferred=bool(r.inferred))
        if cur:
            out.append(cur)
    res = pd.DataFrame(out, columns=["gid", "direction", "t_start", "t_end", "cams", "inferred"])
    res["cams"] = res.cams.apply(lambda c: ";".join(sorted(c)))
    return res


def match_events(a, b, window, col="t_start"):
    """One-to-one matching of same-direction events of two cameras, closest start times first. Returns [(index_a, index_b)]."""
    edges = []
    for d in ("UP", "DOWN"):
        ia, ib = a.index[a.direction == d], b.index[b.direction == d]
        for i in ia:
            for j in ib[(b.loc[ib, col] - a.loc[i, col]).abs() <= window]:
                edges.append((abs(b.loc[j, col] - a.loc[i, col]), i, j))
    used_a, used_b, out = set(), set(), []
    for _, i, j in sorted(edges):
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            out.append((i, j))
    return out


def dedup_cross_camera(ev, c):
    """The same crossing filmed by two cameras whose people could not be linked by appearance (ReID is weak: different people in
    dark jackets score 0.84) is matched by time instead: same direction, start within `window_s`, one-to-one. Only camera pairs
    listed in `pairs` ([primary, secondary]) are used - pairs whose overlap was proven with the shifted-baseline test. The primary
    camera's event is kept, the secondary is dropped and recorded in `dedup_with`."""
    ev = ev.copy()
    ev["dedup_with"] = ""
    cams = ev.cams.str.split(";")
    drop = []
    for prim, sec in c.get("pairs", []):
        a = ev[(cams.apply(lambda x: x == [prim])) & ~ev.index.isin(drop)]
        b = ev[(cams.apply(lambda x: x == [sec])) & ~ev.index.isin(drop)]
        for i, j in match_events(a, b, c.get("window_s", 1.5)):
            ev.at[i, "cams"] = ";".join(sorted({prim, sec}))
            ev.at[i, "dedup_with"] = f"{ev.at[j, 'gid']}@{sec}"
            drop.append(j)
    return ev.drop(index=drop)


def compute_events(tracks, tracklets, zones, meta, cfg):
    """tracks: cam -> DataFrame; tracklets: cam,tid,gid; returns (events, naive per-camera counts)."""
    e, off = cfg["events"], cfg.get("offsets_s", {})
    ecam = lambda c: {**e, **e.get("cam_overrides", {}).get(c, {})}  # per-camera event settings (e.g. widened zones)
    per_gid, naive, rows_by_key, inferred = {}, [], {}, []
    gid_of = {(r.cam, r.tid): r.gid for r in tracklets.itertuples()}
    ed = e.get("edge", {})
    n_local = 0
    for cam, df in tracks.items():
        if cam not in zones or df.empty:
            continue
        df = df.sort_values(["tid", "t"]).copy()
        df["zone"] = zone_labels(df, meta[cam]["width"], meta[cam]["height"], zones[cam])
        for tid, g in df.groupby("tid"):
            ec = ecam(cam)
            v = visits_for_track(g.t.values + off.get(cam, 0), g.zone.values, cam, ec["gap_tol_s"], ec["min_dwell_rows"])
            for ev in transitions(v, ec["max_cross_s"], ec.get("min_cross_s", 0)):  # per-tracklet, no cross-camera dedup
                naive.append(dict(cam=cam, direction=ev["direction"]))
            if e.get("local_events", False) and (cam, tid) not in gid_of:
                n_local += 1
                gid_of[(cam, tid)] = -n_local  # no ReID embedding (short track): local identity, never merged
            if ed.get("enabled", False) and cam in ed.get("cams", [cam]) and (cam, tid) in gid_of:
                for x in edge_events(g, v, meta[cam]["width"], meta[cam]["height"], dict(e, **ed), off.get(cam, 0)):
                    inferred.append(dict(gid=gid_of[(cam, tid)], cam=cam, **x))
            if (cam, tid) in gid_of:
                rows_by_key.setdefault((gid_of[(cam, tid)], cam) if e.get("per_camera", True) else (gid_of[(cam, tid)], ""), []).append(
                    pd.DataFrame(dict(t=g.t.values + off.get(cam, 0), fx=(g.x1 + g.x2).values / 2, fy=g.y2.values, h=(g.y2 - g.y1).values)))
                key = (gid_of[(cam, tid)], cam) if e.get("per_camera", True) else (gid_of[(cam, tid)], "")
                per_gid.setdefault(key, []).extend(v)
    rows = []
    for key, v in per_gid.items():
        gid = key[0]
        ec = ecam(key[1]) if key[1] else e
        path = pd.concat(rows_by_key[key]).sort_values("t") if e.get("verify_path", True) else None
        for ev in transitions(v, ec["max_cross_s"], ec.get("min_cross_s", 0)):
            if path is not None and not path_ok(path, ev, dict(ec, frame_h=meta[key[1]]["height"] if key[1] else 1.0)):
                continue
            if ec.get("flip_window_s", 0) and path is not None:
                m = path[(path.t >= ev["t_start"] - ec.get("path_pad_s", 0)) & (path.t <= ev["t_end"] + ec.get("path_pad_s", 0))]
                ev["dy"] = abs(float(m.fy.values[-1] - m.fy.values[0])) if len(m) else 0.0
            rows.append(dict(gid=gid, direction=ev["direction"], t_start=round(ev["t_start"], 2),
                             t_end=round(ev["t_end"], 2), cams=";".join(sorted(ev["cams"])), dy=ev.get("dy", 0.0),
                             flip=ec.get("flip_window_s", 0), inferred=False))
    win = e.get("merge_window_s", 6.0)
    for r in inferred:  # edge-inferred crossings: skip when the same identity already has that crossing
        ec = ecam(r["cam"])
        if any(k["gid"] == r["gid"] and k["direction"] == r["direction"] and k["t_start"] <= r["t_end"] + win and k["t_end"] + win >= r["t_start"]
               for k in rows):
            continue
        rows.append(dict(gid=r["gid"], direction=r["direction"], t_start=round(r["t_start"], 2), t_end=round(r["t_end"], 2),
                         cams=r["cam"], dy=r["rise"] * meta[r["cam"]]["height"], flip=ec.get("flip_window_s", 0), inferred=True))
    rows = drop_flips(rows)
    ev = pd.DataFrame(rows, columns=["gid", "direction", "t_start", "t_end", "cams", "inferred"])
    ev = merge_across_cameras(ev, e.get("merge_window_s", 6.0))
    ev["confirmed_by"] = ""
    for i, r in ev[ev.inferred].iterrows():  # another camera saw the same-direction crossing at the same time (confirmation only)
        o = ev[(ev.direction == r.direction) & (~ev.cams.str.split(";").apply(lambda c: bool(set(c) & set(r.cams.split(";"))))) &
               ((ev.t_end - r.t_end).abs() <= ed.get("confirm_window_s", 6.0))]
        ev.at[i, "confirmed_by"] = ";".join(sorted(set(";".join(o.cams).split(";")))) if len(o) else ""
    if e.get("cross_camera_dedup", {}).get("pairs"):
        ev = dedup_cross_camera(ev, e["cross_camera_dedup"])
    ev = ev.sort_values("t_start").reset_index(drop=True)
    ev.insert(0, "event_id", np.arange(1, len(ev) + 1))
    return ev, pd.DataFrame(naive, columns=["cam", "direction"])
