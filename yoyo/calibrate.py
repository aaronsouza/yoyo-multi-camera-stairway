"""Bonus: project foot points of every camera onto one floor plan and draw per-person stitched trajectories.

homographies.yaml:  cam: {image: [[x,y],...], floor: [[X,Y],...]}   (>=4 matching ground-plane points, pixels / floor px)
Limitation: a homography is only valid for one plane; people on the stairs sit above the street plane, so stair
tracks are displaced. Calibrate on the street plane and keep stair points as an approximation.
"""
import cv2
import numpy as np
import pandas as pd
import yaml
from .common import ROOT, final_dir, read_json, cam_dir

COLORS = [(66, 135, 245), (60, 180, 75), (245, 130, 48), (145, 30, 180), (0, 160, 160), (240, 50, 230)]


def run(cfg, cams, log=print):
    hp = ROOT / "homographies.yaml"
    if not hp.exists():
        log("[calibrate] homographies.yaml missing (see tools/annotate.py --mode calib/floor)")
        return
    H = {}
    for cam, v in yaml.safe_load(hp.read_text()).items():
        if len(v.get("image", [])) >= 4 and len(v["image"]) == len(v["floor"]):
            H[cam], _ = cv2.findHomography(np.float32(v["image"]), np.float32(v["floor"]), cv2.RANSAC, 5.0)
    tl = pd.read_csv(final_dir(cfg) / "tracklets.csv")
    gid = {(r.cam, r.tid): r.gid for r in tl.itertuples()}
    rows = []
    for cam in cams:
        if cam not in H or not (cam_dir(cfg, cam) / "tracks.csv").exists():
            continue
        df = pd.read_csv(cam_dir(cfg, cam) / "tracks.csv")
        off = cfg.get("offsets_s", {}).get(cam, 0)
        df["gid"] = [gid.get((cam, t)) for t in df.tid]
        df = df.dropna(subset=["gid"])
        pts = np.float32(np.c_[(df.x1 + df.x2) / 2, df.y2]).reshape(-1, 1, 2)
        fl = cv2.perspectiveTransform(pts, H[cam]).reshape(-1, 2)
        rows.append(pd.DataFrame(dict(cam=cam, gid=df.gid.astype(int), t=df.t + off, X=fl[:, 0], Y=fl[:, 1])))
    if not rows:
        return
    ft = pd.concat(rows).sort_values(["gid", "t"])
    ft.to_csv(final_dir(cfg) / "floor_tracks.csv", index=False)
    plan = ROOT / "floorplan.png"
    img = cv2.imread(str(plan)) if plan.exists() else np.full((800, 1000, 3), 255, np.uint8)
    for g, x in ft.groupby("gid"):
        if len(x) < 10:
            continue
        pts = x[["X", "Y"]].values.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, COLORS[g % len(COLORS)], 2, cv2.LINE_AA)
        cv2.putText(img, f"P{g}", tuple(pts[0, 0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[g % len(COLORS)], 1)
    cv2.imwrite(str(final_dir(cfg) / "floor_tracks.png"), img)
    log(f"[calibrate] {len(ft)} floor points from {ft.cam.nunique()} cameras")
