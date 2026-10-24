"""Bonus: the floor-plan picture and the coverage of every camera.

The plan background is a mosaic of the cameras' own people-free backgrounds warped onto the calibrated plaza plane, so the paving seams must line
up across camera borders - a direct visual check of the calibration. Coverage polygons come from the same maps and carry the reliability class:
reliable (<= 6 camera heights away), approximate (6-10), everything else is not mapped.
"""
import json

import cv2
import numpy as np

from .calibrate import APPROX_INDEX, RELIABLE_INDEX, Calibration, out_dir

PPM = 20
NEAR_LIMIT = 0.55


def _grid(cal, cam, step=16, W=1920, H=1080):
    xs, ys = np.meshgrid(np.arange(0, W, step), np.arange(0, H, step))
    xy = np.c_[xs.ravel(), ys.ravel()].astype(float)
    fl, cls = cal.to_floor(cam, xy)
    return xy, fl, cls


def extent(cal, pad=4.0):
    pts = []
    for c in cal.cams:
        _, fl, cls = _grid(cal, c)
        pts.append(fl[cls != "unmapped"])
        pts.append(np.array([cal.cal["cameras"][c]["position_m"]]))
    p = np.concatenate(pts)
    return (np.percentile(p[:, 0], 0.5) - pad, np.percentile(p[:, 0], 99.5) + pad, np.percentile(p[:, 1], 0.5) - pad, np.percentile(p[:, 1], 99.5) + pad)


def to_px(ext, xy, ppm=PPM):
    xy = np.asarray(xy, float)
    return np.c_[(xy[:, 0] - ext[0]) * ppm, (ext[3] - xy[:, 1]) * ppm]


def coverage(cal, ext, ppm=PPM, step=12):
    """Polygons (in metres) of each camera's reliable and approximate floor area."""
    res = {}
    for c in cal.cams:
        xy, fl, cls = _grid(cal, c, step)
        res[c] = {}
        for name in ("reliable", "approximate"):
            keep = (cls == "reliable") if name == "reliable" else (cls != "unmapped")
            Wp, Hp = int((ext[1] - ext[0]) * ppm), int((ext[3] - ext[2]) * ppm)
            m = np.zeros((Hp, Wp), np.uint8)
            q = to_px(ext, fl[keep], ppm).astype(int)
            q = q[(q[:, 0] >= 0) & (q[:, 0] < Wp) & (q[:, 1] >= 0) & (q[:, 1] < Hp)]
            m[q[:, 1], q[:, 0]] = 255
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
            cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polys = []
            for k in cnt:
                if cv2.contourArea(k) > 40:
                    k = cv2.approxPolyDP(k, 2.0, True).reshape(-1, 2).astype(float)
                    polys.append(np.c_[k[:, 0] / ppm + ext[0], ext[3] - k[:, 1] / ppm].tolist())
            res[c][name] = polys
    return res


def mosaic(cfg, cal, ext, ppm=PPM):
    """Top-down picture of the plaza: every camera's background warped onto the plan, the best-resolved camera winning at each point."""
    bgdir = out_dir(cfg) / "bg"
    Wp, Hp = int((ext[1] - ext[0]) * ppm), int((ext[3] - ext[2]) * ppm)
    S = np.array([[ppm, 0, -ext[0] * ppm], [0, -ppm, ext[3] * ppm], [0, 0, 1.0]])
    best = np.zeros((Hp, Wp), np.float32)
    img = np.full((Hp, Wp, 3), 238, np.uint8)
    for c in cal.cams:
        bg = cv2.imread(str(bgdir / f"{c}_bg.png"))
        H = S @ cal.H[c]
        ys, xs = np.mgrid[0:1080, 0:1920]
        xy = np.c_[xs.ravel(), ys.ravel()].astype(float)
        C = np.c_[xy, np.ones(len(xy))] @ cal.G[c][2]
        q = np.where((C > 1.0 / RELIABLE_INDEX) & (C < NEAR_LIMIT), C, 0).reshape(1080, 1920).astype(np.float32)
        from matplotlib.path import Path as MPath
        from .geometry import STAIRS
        if STAIRS.get(c):
            q[MPath(np.array(STAIRS[c]) * [1920, 1080]).contains_points(xy).reshape(1080, 1920)] = 0
        wq = cv2.warpPerspective(q, H, (Wp, Hp), flags=cv2.INTER_NEAREST)
        bgn = cv2.convertScaleAbs(bg, alpha=190.0 / max(float(np.median(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY))), 1.0))
        wi = cv2.warpPerspective(bgn, H, (Wp, Hp), flags=cv2.INTER_LINEAR)
        use = wq > best
        img[use] = wi[use]
        best[use] = wq[use]
    img[best == 0] = (238, 238, 238)
    return img, best > 0


STAIRS_BASE_PX = {"cam5": [[76, 440], [1431, 412]]}


def stairs_base(cal):
    out = []
    for cam, pts in STAIRS_BASE_PX.items():
        p = np.c_[np.array(pts, float), np.ones(len(pts))] @ cal.H[cam].T
        out = (p[:, :2] / p[:, 2:3]).tolist()
    return out


def build(cfg, log=print):
    cal = Calibration()
    ext = extent(cal)
    img, covered = mosaic(cfg, cal, ext)
    d = out_dir(cfg)
    cv2.imwrite(str(d / "floorplan.png"), img)
    cov = coverage(cal, ext)
    meta = dict(extent=list(ext), ppm=PPM, coverage=cov, stairs_base=stairs_base(cal), covered_fraction=float(covered.mean()),
                note="reliable: <=%.0f camera heights from the camera; approximate: <=%.0f; stairs and the rest are not mapped" % (RELIABLE_INDEX, APPROX_INDEX))
    (d / "floorplan.json").write_text(json.dumps(meta))
    log(f"[floorplan] {img.shape[1]}x{img.shape[0]} px, extent x {ext[0]:.1f}..{ext[1]:.1f} m, y {ext[2]:.1f}..{ext[3]:.1f} m")
    return img, meta
