"""Bonus: continuous person trajectories on the floor plan, stitched across cameras from the saved tracks and global IDs.

Rules (nothing is invented):
  * only observed foot points inside the calibrated plaza area are used; stairs and far-field points are left out (calibrate.Calibration)
  * tracklets of one global ID are joined into one trajectory ("line") only if physically consistent:
      - overlapping in time (two cameras see the person together): median distance of the two paths <= `conflict_m`
      - separated by a gap: gap <= `max_gap_s` and the implied speed <= `max_speed`
    otherwise the ID is split: a ReID mistake (two people sharing an ID) becomes two lines and is logged as a conflict
  * where cameras overlap, simultaneous observations are averaged (reliable ones weigh more)
  * gaps between observations are kept as gaps (no interpolation); they are listed as camera handovers or tracking gaps
"""
import numpy as np
import pandas as pd

from .calibrate import Calibration, out_dir
from .common import cam_dir, final_dir

DEFAULTS = dict(conflict_m=5.0, max_gap_s=60.0, max_speed=2.5, overlap_min_s=1.0, bin_s=0.5, seg_gap_s=1.5, min_obs=5, jump_m=1.5)


def observations(cfg, cal, cams, log=print):
    """Per-detection floor positions of every tracked person that has a global ID (reliable or approximate points only)."""
    tl = pd.read_csv(final_dir(cfg) / "tracklets.csv")
    rows = []
    for c in cams:
        gid = dict(zip(tl[tl.cam == c].tid, tl[tl.cam == c].gid))
        t = pd.read_csv(cam_dir(cfg, c) / "tracks.csv")
        t = t[t.tid.isin(gid) & (t.y2 < 1074) & (t.x1 > 4) & (t.x2 < 1916)].copy()
        fl, cls = cal.to_floor(c, np.c_[(t.x1 + t.x2) / 2, t.y2])
        ok = cls != "unmapped"
        t = t[ok]
        rows.append(pd.DataFrame(dict(cam=c, tid=t.tid.values, gid=t.tid.map(gid).astype(int).values, t=t.t.values,
                                      X=fl[ok, 0], Y=fl[ok, 1], cls=cls[ok])))
        log(f"[stitch] {c}: {ok.sum()} of {len(ok)} detections mapped to the floor ({ok.mean():.0%})")
    return pd.concat(rows, ignore_index=True)


def tracklet_paths(obs, p):
    """One smoothed path per (cam, tid): median filter of the foot points (3 frames), jumps larger than `jump_m` cut the track in two."""
    paths = []
    for (cam, tid), g in obs.groupby(["cam", "tid"], sort=False):
        g = g.sort_values("t")
        if len(g) < p["min_obs"]:
            continue
        xy = g[["X", "Y"]].rolling(3, center=True, min_periods=1).median().values
        t = g.t.values
        step = np.hypot(*np.diff(xy, axis=0).T)
        cut = np.flatnonzero((step > p["jump_m"]) & (np.diff(t) < 1.0)) + 1
        for k, (a, b) in enumerate(zip(np.r_[0, cut], np.r_[cut, len(g)])):
            if b - a >= p["min_obs"]:
                paths.append(dict(cam=cam, tid=int(tid), gid=int(g.gid.iloc[0]), k=k, t=t[a:b], xy=xy[a:b], cls=g.cls.values[a:b]))
    return paths


def pair_score(a, b, par):
    """Evidence that two tracklets of one global ID are the same person: positive = same, negative = different.
    Overlapping in time: seconds of agreement (+) or disagreement (-1.5 per second) between the two cameras' positions.
    Separated in time: a weak + if a walk between them is possible, - if not (too long a gap, or faster than walking allows
    once the calibration error `conflict_m` is taken into account)."""
    lo, hi = max(a["t"][0], b["t"][0]), min(a["t"][-1], b["t"][-1])
    if hi - lo >= par["overlap_min_s"]:
        sel = (b["t"] >= lo) & (b["t"] <= hi)
        if not sel.any():
            return 0.0, None
        mx = np.interp(b["t"][sel], a["t"], a["xy"][:, 0])
        my = np.interp(b["t"][sel], a["t"], a["xy"][:, 1])
        d = float(np.median(np.hypot(b["xy"][sel, 0] - mx, b["xy"][sel, 1] - my)))
        return (hi - lo) * (1.0 if d <= par["conflict_m"] else -1.5), f"simultaneous in {a['cam']} and {b['cam']} but {d:.1f} m apart" if d > par["conflict_m"] else None
    first, second = (a, b) if a["t"][-1] <= b["t"][0] else (b, a)
    gap = float(second["t"][0] - first["t"][-1])
    dist = float(np.hypot(*(first["xy"][-1] - second["xy"][0])))
    if gap > par["max_gap_s"]:
        return -2.0, f"gap {gap:.0f} s"
    if dist > par["max_speed"] * max(gap, 0.0) + par["conflict_m"]:
        return -2.0, f"{dist:.1f} m in {gap:.1f} s ({dist / max(gap, 0.5):.1f} m/s)"
    return 0.3, None


def _agglomerate(groups, S, accept):
    """Greedy average-linkage merging: repeatedly merge the pair of groups with the largest total evidence that `accept` allows."""
    while len(groups) > 1:
        best, bi = 0.0, None
        for x in range(len(groups)):
            for y in range(x + 1, len(groups)):
                sc = S[np.ix_(groups[x], groups[y])].sum()
                if sc > best and accept(groups[x], groups[y]):
                    best, bi = sc, (x, y)
        if bi is None:
            break
        x, y = bi
        groups[x] = groups[x] + groups[y]
        del groups[y]
    return groups


def stitch_gid(paths, par):
    """Group the tracklets of one global ID into lines (people), in two stages.
    1. strong evidence only - two cameras seeing the person at the same time at the same place (+) or in different places (-):
       average-linkage clustering, so one contradictory pair is outvoted by the others instead of the first arrival deciding;
    2. weak evidence - groups separated in time are joined only if a walk between them is possible AND nothing in either group
       contradicts the other (no simultaneous disagreement, no impossible jump).
    Returns (lines, reasons for every remaining split)."""
    n = len(paths)
    if n == 1:
        return [list(paths)], []
    strong, weak = np.zeros((n, n)), np.zeros((n, n))
    why = {}
    for i in range(n):
        for j in range(i + 1, n):
            sc, w = pair_score(paths[i], paths[j], par)
            lo, hi = max(paths[i]["t"][0], paths[j]["t"][0]), min(paths[i]["t"][-1], paths[j]["t"][-1])
            (strong if hi - lo >= par["overlap_min_s"] else weak)[i, j] = (strong if hi - lo >= par["overlap_min_s"] else weak)[j, i] = sc
            if w:
                why[(i, j)] = why[(j, i)] = w
    groups = _agglomerate([[i] for i in range(n)], strong, lambda g, h: True)
    groups = _agglomerate(groups, weak + strong, lambda g, h: strong[np.ix_(g, h)].min() >= 0 and weak[np.ix_(g, h)].min() >= 0)
    reasons = []
    if len(groups) > 1:
        for x in range(len(groups)):
            for y in range(x + 1, len(groups)):
                r = [why[(i, j)] for i in groups[x] for j in groups[y] if (i, j) in why]
                reasons.append(r[0] if r else "no supporting overlap or feasible walk")
    return [[paths[i] for i in g] for g in groups], reasons


def fuse(line, gid, k, par):
    """Observed positions of one line at `bin_s` resolution; simultaneous cameras are averaged (reliable weight 1, approximate 0.3)."""
    t = np.concatenate([m["t"] for m in line])
    xy = np.concatenate([m["xy"] for m in line])
    w = np.where(np.concatenate([m["cls"] for m in line]) == "reliable", 1.0, 0.3)
    cam = np.concatenate([[m["cam"]] * len(m["t"]) for m in line])
    cls = np.concatenate([m["cls"] for m in line])
    b = np.round(t / par["bin_s"]).astype(int)
    df = pd.DataFrame(dict(b=b, x=xy[:, 0] * w, y=xy[:, 1] * w, w=w, cam=cam, rel=(cls == "reliable")))
    df["xr"], df["yr"] = df.x / df.w, df.y / df.w
    spread = df.groupby("b").agg(sx=("xr", lambda v: v.max() - v.min()), sy=("yr", lambda v: v.max() - v.min()))
    far = spread.index[np.hypot(spread.sx, spread.sy) > par["conflict_m"]]
    keep = df[~df.b.isin(far)]
    top = df[df.b.isin(far)].sort_values("w", ascending=False).drop_duplicates("b")
    df = pd.concat([keep, top])
    g = df.groupby("b").agg(x=("x", "sum"), y=("y", "sum"), w=("w", "sum"), cams=("cam", lambda s: ";".join(sorted(set(s)))), rel=("rel", "max")).reset_index()
    out = pd.DataFrame(dict(line_id=f"{gid}-{k}", gid=gid, t=g.b * par["bin_s"], X=g.x / g.w, Y=g.y / g.w, cams=g.cams,
                            n_cams=g.cams.str.count(";") + 1, quality=np.where(g.rel, "reliable", "approximate")))
    seg = np.r_[0, np.cumsum(np.diff(out.t.values) > par["seg_gap_s"])]
    out["segment"] = seg
    return out


def run(cfg, cal=None, cams=None, log=print, **kw):
    par = {**DEFAULTS, **kw}
    cal = cal or Calibration()
    cams = cams or cal.cams
    obs = observations(cfg, cal, cams, log)
    paths = tracklet_paths(obs, par)
    by_gid = {}
    for p in paths:
        by_gid.setdefault(p["gid"], []).append(p)
    traj, gaps, conflicts, members = [], [], [], []
    for gid, ps in by_gid.items():
        lines, reasons = stitch_gid(ps, par)
        if reasons:
            conflicts.append(dict(gid=gid, lines=len(lines), reasons=" | ".join(reasons[:3])))
        for k, L in enumerate(lines):
            d = fuse(L, gid, k, par)
            traj.append(d)
            members += [dict(line_id=f"{gid}-{k}", gid=gid, cam=m["cam"], tid=m["tid"], t0=float(m["t"][0]), t1=float(m["t"][-1])) for m in L]
            for s0, s1 in zip(range(d.segment.max()), range(1, d.segment.max() + 1)):
                a, b = d[d.segment == s0].iloc[-1], d[d.segment == s1].iloc[0]
                dist = float(np.hypot(a.X - b.X, a.Y - b.Y))
                gaps.append(dict(line_id=a.line_id, t_from=a.t, t_to=b.t, seconds=b.t - a.t, from_cams=a.cams, to_cams=b.cams, distance_m=dist,
                                 kind="camera handover" if not set(a.cams.split(";")) & set(b.cams.split(";")) else "tracking gap"))
    traj = pd.concat(traj, ignore_index=True)
    gaps, conflicts = pd.DataFrame(gaps), pd.DataFrame(conflicts)
    d = out_dir(cfg)
    traj = traj.sort_values(["line_id", "t"]).reset_index(drop=True)
    step = np.hypot(traj.groupby("line_id").X.diff(), traj.groupby("line_id").Y.diff()).where(traj.groupby("line_id").segment.diff() == 0, 0).fillna(0)
    idx = traj.assign(step=step).groupby("line_id").agg(
        gid=("gid", "first"), t0=("t", "min"), t1=("t", "max"), n_obs=("t", "size"), path_m=("step", "sum"), segments=("segment", "max"),
        cameras=("cams", lambda s: ";".join(sorted(set(";".join(s).split(";"))))), reliable_share=("quality", lambda q: float((q == "reliable").mean()))).reset_index()
    idx["segments"] += 1
    idx["duration_s"] = idx.t1 - idx.t0
    idx["n_cameras"] = idx.cameras.str.count(";") + 1
    idx["id_was_split"] = idx.gid.map(traj.groupby("gid").line_id.nunique() > 1)
    idx.to_csv(d / "lines_index.csv", index=False)
    traj.to_csv(d / "trajectories.csv.gz", index=False)
    gaps.to_csv(d / "gaps.csv", index=False)
    conflicts.to_csv(d / "identity_conflicts.csv", index=False)
    pd.DataFrame(members).to_csv(d / "line_members.csv", index=False)
    multi = traj.groupby("line_id").cams.apply(lambda s: len(set(";".join(s).split(";"))))
    log(f"[stitch] {traj.line_id.nunique()} trajectories from {len(by_gid)} global IDs; {int((multi > 1).sum())} span several cameras; "
        f"{len(conflicts)} IDs had to be split ({conflicts.lines.sum() - len(conflicts) if len(conflicts) else 0} extra lines)")
    return traj, gaps, conflicts
