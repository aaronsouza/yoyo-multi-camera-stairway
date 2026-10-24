"""Bonus: how good is the calibration? Two independent checks, both written to calibration/validation.json.

1. Landmark reprojection: a physical point seen by several cameras must land on the same floor position. The landmarks (calibration/landmarks.yaml) were
   read by eye and are not used to fit anything. Error = distance of each camera's estimate from the mean of the cameras.
2. Time-shift baseline: foot points of two cameras at the same instant match within 1 m more often than at instants 7-20 s apart. The excess is
   the share of matches that are real (same people), which exists only if the two cameras are registered correctly.
"""
import itertools
import json

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

from .calibrate import CAL_DIR, Calibration
from .common import cam_dir


def landmark_errors(cal):
    lm = yaml.safe_load((CAL_DIR / "landmarks.yaml").read_text())["landmarks"]
    out, all_err = {}, []
    for name, v in lm.items():
        pos, idx = {}, {}
        for c, px in v["pixels"].items():
            fl, cls = cal.to_floor(c, [px])
            C = np.r_[px, 1.0] @ cal.G[c][2]
            pos[c], idx[c] = fl[0], float(1 / C)
        pts = np.array(list(pos.values()))
        mean = pts.mean(0)
        err = {c: float(np.hypot(*(p - mean))) for c, p in pos.items()}
        spread = max(np.hypot(*(a - b)) for a, b in itertools.combinations(pts, 2))
        out[name] = dict(used=bool(v["use"]), cameras=len(pos), error_m=err, distance_index=idx, max_pair_spread_m=float(spread), note=v["note"])
        if v["use"]:
            all_err += list(err.values())
    stats = dict(n_observations=len(all_err), rms_m=float(np.sqrt(np.mean(np.square(all_err)))), median_m=float(np.median(all_err)), max_m=float(np.max(all_err)))
    return out, stats


def baseline_test(cfg, cal, step=60, shifts_s=(-20, -10, -7, 7, 10, 20), gate=1.0, log=print):
    from . import register as reg
    P = {}
    for c in cal.cams:
        tr = pd.read_csv(cam_dir(cfg, c) / "tracks.csv")
        tr = tr[(tr.frame % step == 0) & (tr.y2 < 1074) & (tr.x1 > 4) & (tr.x2 < 1916) & (tr.y2 - tr.y1 > 40)]
        fl, cls = cal.to_floor(c, np.c_[(tr.x1 + tr.x2) / 2, tr.y2])
        ok = cls != "unmapped"
        P[c] = pd.DataFrame(dict(frame=tr.frame.values[ok], X=fl[ok, 0], Y=fl[ok, 1]))

    def count(a, b, shift):
        B = P[b].assign(frame=P[b].frame + shift)
        ga, gb = dict(list(P[a].groupby("frame"))), dict(list(B.groupby("frame")))
        m = 0
        for fr in set(ga) & set(gb):
            d, _ = cKDTree(gb[fr][["X", "Y"]].values).query(ga[fr][["X", "Y"]].values, distance_upper_bound=gate)
            m += int(np.isfinite(d).sum())
        return m

    rows = []
    for a, b in itertools.combinations(cal.cams, 2):
        real = count(a, b, 0)
        base = float(np.mean([count(a, b, int(s) * step) for s in shifts_s]))  # whole sampling steps (one step = one second), so frames stay aligned
        rows.append(dict(pair=f"{a}-{b}", real=real, shifted=round(base), excess=round(real - base), real_share=round(max(real - base, 0) / max(real, 1), 3)))
        log(f"[check] {a}-{b}: {real} matches, {base:.0f} by chance, {rows[-1]['real_share']:.0%} real")
    return rows


def run(cfg, log=print, baseline=True):
    cal = Calibration()
    lm, stats = landmark_errors(cal)
    res = dict(landmarks=lm, landmark_stats=stats)
    if baseline:
        res["time_shift_baseline"] = baseline_test(cfg, cal, log=log)
        res["third_camera_check"] = third_camera_check(cfg, cal, log=log)
    res["per_camera"] = {c: dict(seam_median_error_deg=e["seam_fit"]["median_err_deg"], seam_inlier_length_share=e["seam_fit"]["inlier_frac"],
                                 camera_height_m=e["height_m"], vote_evidence=(e["vote_evidence"] or {}).get("ratio")) for c, e in cal.cal["cameras"].items()}
    (CAL_DIR / "validation.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"[check] landmark error rms {stats['rms_m']:.2f} m, median {stats['median_m']:.2f} m, max {stats['max_m']:.2f} m over {stats['n_observations']} observations")
    return res


def third_camera_check(cfg, cal, n=600, gates=(0.5, 1.0, 1.5), shift_s=(-15, 15), seed=0, log=print):
    """Independent test of calibration + stitching together. A person seen by two cameras has a floor position; a THIRD camera that was not used for that
    trajectory and covers the position reliably should have a detected person near it. Reports the hit rate and the hit rate when the same position is
    looked up at an unrelated moment (chance, crowds make it high)."""
    d = cfg_out(cfg)
    idx = pd.read_csv(d / "lines_index.csv", dtype={"line_id": str})
    mem = pd.read_csv(d / "line_members.csv", dtype={"line_id": str})
    traj = pd.read_csv(d / "trajectories.csv.gz", dtype={"line_id": str})
    cand = idx[(idx.n_cameras >= 2) & (idx.duration_s >= 5)].line_id
    rng = np.random.default_rng(seed)
    det = {}
    for c in cal.cams:
        tr = pd.read_csv(cam_dir(cfg, c) / "tracks.csv")
        tr = tr[(tr.frame % 60 == 0) & (tr.y2 < 1074) & (tr.x1 > 4) & (tr.x2 < 1916) & (tr.y2 - tr.y1 > 40)]
        fl, cls = cal.to_floor(c, np.c_[(tr.x1 + tr.x2) / 2, tr.y2])
        ok = cls != "unmapped"
        tr = tr[ok]
        det[c] = {int(fr): g[["X", "Y"]].values for fr, g in pd.DataFrame(dict(frame=tr.frame.values, X=fl[ok, 0], Y=fl[ok, 1])).groupby("frame")}
    pts = traj[traj.line_id.isin(cand) & (traj.quality == "reliable")]
    pts = pts[(pts.t * 59.94 // 60 * 60).astype(int).isin(set().union(*[set(v) for v in det.values()]))]
    rows = pts.sample(min(n * 3, len(pts)), random_state=seed)
    members = {k: set(v.cam) for k, v in mem.groupby("line_id")}
    trials = ch_trials = 0
    real_d, chance_d = [], []
    for r in rows.itertuples():
        for c in cal.cams:
            if c in members[r.line_id] or c in r.cams.split(";"):
                continue
            H = np.linalg.inv(cal.H[c])
            p = H @ np.array([r.X, r.Y, 1.0])
            p = p[:2] / p[2]
            C = float(np.r_[p, 1] @ cal.G[c][2])
            if not (C > 1 / 6 and 20 < p[0] < 1900 and 20 < p[1] < 1060):
                continue
            from matplotlib.path import Path as MPath
            if geo_stairs(c, p):
                continue
            for shift, store in ((0, "real"),) + tuple((s, "chance") for s in shift_s):
                fr = int(round((r.t + shift) * 59.94 / 60)) * 60
                D = det[c].get(fr)
                if D is None or not len(D):
                    continue
                near = float(np.min(np.hypot(D[:, 0] - r.X, D[:, 1] - r.Y)))
                if store == "real":
                    trials += 1
                    real_d.append(near)
                else:
                    ch_trials += 1
                    chance_d.append(near)
        if trials >= n:
            break
    real_d, chance_d = np.array(real_d), np.array(chance_d)
    res = dict(trials=int(trials), chance_trials=int(ch_trials), median_distance_m=float(np.median(real_d)), chance_median_distance_m=float(np.median(chance_d)),
               by_gate={str(g): dict(hit_rate=float((real_d <= g).mean()), chance_hit_rate=float((chance_d <= g).mean())) for g in gates})
    for g, v in res["by_gate"].items():
        log(f"[check] third camera: person within {g} m of the predicted spot in {v['hit_rate']:.0%} of {trials} cases (chance {v['chance_hit_rate']:.0%})")
    return res


def cfg_out(cfg):
    from .calibrate import out_dir
    return out_dir(cfg)


def geo_stairs(cam, px):
    from matplotlib.path import Path as MPath
    from .geometry import STAIRS
    return bool(STAIRS.get(cam)) and MPath(np.array(STAIRS[cam]) * [1920, 1080]).contains_point(px)
