"""Bonus: calibrate all cameras onto one floor plan (metres) and stitch tracks across cameras.

Pipeline (all from saved tracks plus a few frames of video; no detector or ReID is run):
  1. people-free background of every camera (median of frames spread over the recording)
  2. paving seams -> one shared focal length and each camera's ground-plane orientation        (geometry.py)
  3. camera height from people's pixel height -> metric plan per camera                          (geometry.py)
  4. cameras tied to one common plan by time-synchronised foot-point voting + joint ICP, no ReID (register.py)
  5. calibration.json (per-camera floor homography, camera position and heading, reliability) + floor-plan mosaic
Validation lives in calibration_check.py. Stitching lives in stitch.py.

Not modelled: lens distortion, and the staircase (not on the plaza plane): stair detections are left unmapped, never invented.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import geometry as geo
from . import register as reg
from .common import ROOT, cam_dir, load_config, read_json, videos, work

CAL_DIR = ROOT / "calibration"  # small, versioned: the calibration itself
OUT = "calibration"  # under results/: backgrounds, mosaic, trajectories (large, not versioned)
RELIABLE_INDEX, APPROX_INDEX = 6.0, 10.0  # 1/C: distance from the camera in camera heights. Landmark checks: <=0.4 m error up to ~7, <=1.9 m up to ~10


def out_dir(cfg):
    d = work(cfg) / OUT
    d.mkdir(parents=True, exist_ok=True)
    return d


def backgrounds(cfg, cams, n=15, log=print):
    """Median of n frames spread over the recording: moving people disappear. Cached as PNG."""
    d = out_dir(cfg) / "bg"
    d.mkdir(exist_ok=True)
    vids = videos(cfg)
    res = {}
    for c in cams:
        p = d / f"{c}_bg.png"
        if not p.exists():
            cap = cv2.VideoCapture(str(vids[c]))
            frames = []
            for k in range(n):
                cap.set(cv2.CAP_PROP_POS_MSEC, (200 + k * 110) * 1000)
                ok, f = cap.read()
                if ok:
                    frames.append(f)
            cv2.imwrite(str(p), np.median(np.stack(frames), 0).astype(np.uint8))
            log(f"[calibrate] background {c}")
        res[c] = cv2.imread(str(p))
    return res


def floor_homography(cam_cal):
    """3x3 pixel -> floor (metres) homography of one camera from its calibration entry."""
    G = np.array(cam_cal["G"])
    S = np.diag([cam_cal["height_m"], cam_cal["height_m"], 1.0])
    return reg.sim_matrix(cam_cal["registration"]) @ S @ G


def orient(cal, face="cam5", up_deg=90.0):
    """Rotate the whole floor frame so that camera `face` looks 'up' on the plan (towards the stairs, as one would draw it).
    Applies to every H_floor, position and heading; idempotent (it first undoes any earlier rotation)."""
    th = np.radians(up_deg - cal["cameras"][face]["heading_deg"])
    Q = np.eye(3)
    Q[:2, :2] = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
    for c, e in cal["cameras"].items():
        e["H_floor"] = (Q @ np.array(e["H_floor"])).tolist()
        e["position_m"] = (Q @ np.r_[e["position_m"], 1.0])[:2].tolist()
        e["heading_deg"] = float((e["heading_deg"] + np.degrees(th) + 180) % 360 - 180)
    cal["frame_rotation_deg"] = float(cal.get("frame_rotation_deg", 0.0) + np.degrees(th))
    return cal


def run(cfg=None, cams=None, log=print, ref="cam1", order=("cam1", "cam3", "cam5", "cam7", "cam4", "cam6", "cam2")):
    cfg = cfg or load_config()
    cams = cams or sorted(videos(cfg))
    W, H = 1920, 1080
    bgs = backgrounds(cfg, cams, log=log)
    segs = {c: geo.seam_segments(bgs[c], c) for c in cams}
    sizes = {c: (bgs[c].shape[1], bgs[c].shape[0]) for c in cams}
    f, R, st = geo.fit_all(segs, sizes)
    log(f"[calibrate] shared focal length {f:.0f} px; seam fit: " + ", ".join(f"{c} {v['median_err_deg']:.2f} deg" for c, v in st.items()))
    tracks = {c: pd.read_csv(cam_dir(cfg, c) / "tracks.csv") for c in cams}
    G, h = {}, {}
    for c in cams:
        G[c] = geo.ground_homography(R[c], f, *sizes[c])
        h[c], _ = geo.camera_height(G[c], R[c], f, *sizes[c], tracks[c][["x1", "y1", "x2", "y2"]].values)
        log(f"[calibrate] {c}: camera height ~{h[c]:.1f} m")
    P = {c: reg.plan_points(tracks[c], G[c], c, *sizes[c], h[c], step=30, min_c=0.10) for c in cams}
    T, M, evidence = reg.register_all(P, [c for c in order if c in cams], h, log=log)
    cal = dict(focal_px=f, image_size=[W, H], principal_point="image centre", units="metres",
               note="Floor plan = plaza ground plane. Axes follow the paving seams of the reference camera (" + ref + "); origin is that camera's nadir. "
                    "Scale comes from assuming adults are 1.7 m tall (about +-15%).", cameras={})
    for c in cams:
        entry = dict(R=np.asarray(R[c]).tolist(), G=np.asarray(G[c]).tolist(), height_m=float(h[c]), registration=T[c],
                     seam_fit=st[c], vote_evidence=evidence.get(c))
        Hf = floor_homography(entry)
        entry["H_floor"] = Hf.tolist()
        nad = reg.to_common(T[c], np.zeros((1, 2)))[0]
        fwd = np.array(R[c]) @ np.array([0, 0, 1.0])  # optical axis in the camera's own plan axes
        n_ = R[c][:, 2] * (1 if R[c][1, 2] > 0 else -1)
        d1, d2 = R[c][:, 0], -np.cross(n_, R[c][:, 0])
        axis = np.array([float(d1 @ [0, 0, 1]), float(d2 @ [0, 0, 1])])
        hv = reg.sim_matrix(T[c])[:2, :2] @ axis
        entry["position_m"] = nad.tolist()
        entry["heading_deg"] = float(np.degrees(np.arctan2(hv[1], hv[0])))
        cal["cameras"][c] = entry
    cal = orient(cal)
    CAL_DIR.mkdir(exist_ok=True)
    (CAL_DIR / "calibration.json").write_text(json.dumps(cal, indent=1, default=float))
    log(f"[calibrate] wrote {CAL_DIR / 'calibration.json'}")
    return cal


class Calibration:
    """Loaded calibration: map pixels to floor metres with a reliability class. Never invents positions: unreliable or off-plane points are NaN."""

    def __init__(self, path=None):
        self.cal = json.loads(Path(path or CAL_DIR / "calibration.json").read_text())
        self.cams = sorted(self.cal["cameras"])
        self.H = {c: np.array(self.cal["cameras"][c]["H_floor"]) for c in self.cams}
        self.G = {c: np.array(self.cal["cameras"][c]["G"]) for c in self.cams}

    def to_floor(self, cam, xy, W=1920, H=1080):
        """xy: (n, 2) pixel foot points. Returns (floor_xy (n,2), cls) with cls in {'reliable', 'approximate', 'unmapped'}."""
        from matplotlib.path import Path as MPath
        xy = np.asarray(xy, float).reshape(-1, 2)
        p = np.c_[xy, np.ones(len(xy))]
        C = p @ self.G[cam][2]
        q = p @ self.H[cam].T
        with np.errstate(divide="ignore", invalid="ignore"):
            fl = q[:, :2] / q[:, 2:3]
            idx = 1.0 / C
        stairs = MPath(np.array(geo.STAIRS[cam]) * [W, H]).contains_points(xy) if geo.STAIRS.get(cam) else np.zeros(len(xy), bool)
        cls = np.full(len(xy), "unmapped", dtype=object)
        ok = (C > 0) & ~stairs
        cls[ok & (idx <= APPROX_INDEX)] = "approximate"
        cls[ok & (idx <= RELIABLE_INDEX)] = "reliable"
        fl[cls == "unmapped"] = np.nan
        return fl, cls

    def camera_table(self):
        rows = []
        for c in self.cams:
            e = self.cal["cameras"][c]
            rows.append(dict(camera=c, x_m=e["position_m"][0], y_m=e["position_m"][1], heading_deg=e["heading_deg"], height_m=e["height_m"]))
        return pd.DataFrame(rows)
