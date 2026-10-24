"""Bonus (calibration + stitching) checks on synthetic data with known answers."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from yoyo import geometry as geo
from yoyo import register as reg
from yoyo import stitch

W, H, F = 1920, 1080, 1500.0


def _camera(pitch_deg=25.0, yaw_deg=10.0, hc=3.0):
    """Rotation whose columns are two perpendicular ground directions and the downward plane normal, in camera coordinates."""
    p, y = np.radians(pitch_deg), np.radians(yaw_deg)
    n = np.array([0.0, np.cos(p), np.sin(p)])  # points down and forward
    d1 = np.array([np.cos(y), -np.sin(y) * np.sin(p), np.sin(y) * np.cos(p)])
    d1 -= n * (n @ d1)
    d1 /= np.linalg.norm(d1)
    d2 = np.cross(n, d1)
    return np.c_[d1, d2, n], hc


def _pix(X):
    K = geo.K_of(F, W, H)
    p = (K @ X.T).T
    return p[:, :2] / p[:, 2:3]


def _grid_segments(R, hc, rng, noise=0.3):
    """Image segments of two perpendicular families of ground lines (a slab grid) seen by the camera."""
    n, d1, d2 = R[:, 2], R[:, 0], R[:, 1]
    segs = []
    for u, v, offs in ((d1, d2, np.arange(-40.0, -0.5, 0.6)), (d2, d1, np.arange(-25.0, 25.0, 0.7))):
        for off in offs:
            a = np.linspace(-40.0, 40.0, 400)
            X = hc * n + off * v + a[:, None] * u
            X = X[X[:, 2] > 0.5]
            if len(X) < 5:
                continue
            q = _pix(X)
            ok = (q[:, 0] > 0) & (q[:, 0] < W) & (q[:, 1] > 0) & (q[:, 1] < H)
            q = q[ok]
            if len(q) > 5 and np.hypot(*(q[-1] - q[0])) > 80:
                segs.append(np.r_[q[0], q[-1]] + rng.normal(0, noise, 4))
    return np.array(segs)


def test_manhattan_fit_recovers_shared_focal_length_and_ground_orientation():
    # one camera alone cannot fix the focal length (it drifts to infinity), several cameras sharing it can
    Rs, segs = {}, {}
    for k, (pitch, yaw) in enumerate(((20, 10), (25, 25), (30, 40), (28, 80))):
        R, hc = _camera(pitch, yaw)
        Rs[f"c{k}"], segs[f"c{k}"] = R, _grid_segments(R, hc, np.random.default_rng(k))
    f, Rf, st = geo.fit_all(segs, {c: (W, H) for c in segs}, f0=1300.0)
    assert abs(f - F) / F < 0.05
    for c in segs:
        nf = Rf[c][:, 2] * (1 if Rf[c][1, 2] > 0 else -1)
        assert np.degrees(np.arccos(np.clip(nf @ Rs[c][:, 2], -1, 1))) < 1.0  # ground normal
        assert st[c]["median_err_deg"] < 0.5


def test_ground_homography_maps_pixels_to_metric_plan():
    R, hc = _camera()
    G = geo.ground_homography(R, F, W, H)
    n, d1, d2 = R[:, 2], R[:, 0], R[:, 1]
    # two ground points 5 m apart along a seam direction: their plan distance (units of camera height) times hc is 5 m
    P = np.array([hc * n + 8 * d1, hc * n + 13 * d1])
    ab, _ = geo.to_plan(G, _pix(P))
    assert abs(np.hypot(*(ab[1] - ab[0])) * hc - 5.0) < 0.05
    Q = np.array([hc * n + 8 * d1, hc * n + 8 * d1 + 3 * d2])  # and perpendicular 3 m
    ab, _ = geo.to_plan(G, _pix(Q))
    assert abs(np.hypot(*(ab[1] - ab[0])) * hc - 3.0) < 0.05


def _walkers(rng, n_people=40, frames=60):
    rows = []
    for p in range(n_people):
        start = rng.uniform(0, 30, 2)
        vel = rng.normal(0, 0.8, 2)
        t0 = rng.integers(0, frames // 2)
        for f in range(t0, min(t0 + 25, frames)):
            rows.append((f * 30, p, *(start + vel * (f - t0) * 0.5)))
    return pd.DataFrame(rows, columns=["frame", "tid", "x", "y"])


def test_registration_recovers_rotation_scale_and_translation_without_identities():
    rng = np.random.default_rng(3)
    A = _walkers(rng)
    # camera B sees the same people through a plan turned by 270 degrees, scaled 1/1.1 and shifted; its track ids are unrelated
    true = dict(rot=3, scale=1.1, tx=7.0, ty=-3.0)
    Rm = reg.ROT[true["rot"]]
    inv = np.linalg.inv(true["scale"] * Rm)
    xyB = (A[["x", "y"]].values - [true["tx"], true["ty"]]) @ inv.T
    B = pd.DataFrame(dict(frame=A.frame, tid=A.tid + 1000, x=xyB[:, 0], y=xyB[:, 1]))
    v = reg.vote_pair(A, B, np.linspace(0.8, 1.4, 13))
    assert v["rot"] == 3 and abs(v["scale"] - 1.1) < 0.06
    assert abs(v["tx"] - 7.0) < 1.0 and abs(v["ty"] + 3.0) < 1.0


def _path(cam, t0, t1, p0, p1, cls="reliable", tid=1, gid=1):
    t = np.arange(t0, t1, 0.1)
    xy = np.c_[np.linspace(p0[0], p1[0], len(t)), np.linspace(p0[1], p1[1], len(t))]
    return dict(cam=cam, tid=tid, gid=gid, k=0, t=t, xy=xy, cls=np.array([cls] * len(t)))


def test_overlapping_cameras_are_fused_and_inconsistent_ones_are_split():
    par = stitch.DEFAULTS
    a = _path("cam1", 0, 20, (0, 0), (20, 0), tid=1)
    b = _path("cam2", 5, 25, (5.5, 0.3), (25.5, 0.3), tid=2)  # same walker seen by a second camera, 0.5 m off
    lines, reasons = stitch.stitch_gid([a, b], par)
    assert len(lines) == 1 and not reasons
    fused = stitch.fuse(lines[0], 1, 0, par)
    both = fused[(fused.t > 6) & (fused.t < 19)]
    assert (both.n_cams == 2).all()
    c = _path("cam2", 5, 25, (15, 8), (35, 8), tid=3)  # 'same ID' but 8 m away at the same time: a different person
    lines, reasons = stitch.stitch_gid([a, c], par)
    assert len(lines) == 2 and "simultaneous" in reasons[0]


def test_one_contradicting_tracklet_is_outvoted_not_allowed_to_split_the_rest():
    par = stitch.DEFAULTS
    a = _path("cam1", 0, 20, (0, 0), (20, 0), tid=1)
    b = _path("cam2", 5, 25, (5.5, 0.3), (25.5, 0.3), tid=2)   # consistent with a
    c = _path("cam3", 0, 25, (0.8, 0.2), (20.8, 0.2), tid=3)   # consistent with a and b
    x = _path("cam4", 6, 10, (9, 14), (13, 14), tid=4)          # another person: 14 m away from everyone while they overlap
    lines, reasons = stitch.stitch_gid([x, a, b, c], par)       # the bad tracklet arrives first
    assert len(lines) == 2
    assert sorted(len(L) for L in lines) == [1, 3] and any(m["tid"] == 4 for L in lines if len(L) == 1 for m in L)
    assert "simultaneous" in reasons[0]


def test_gaps_are_kept_not_interpolated_and_impossible_jumps_are_split():
    par = stitch.DEFAULTS
    a = _path("cam1", 0, 10, (0, 0), (10, 0), tid=1)
    b = _path("cam2", 16, 26, (15, 0), (25, 0), tid=2)  # 6 s later, 5 m on: a normal walk, a handover
    lines, _ = stitch.stitch_gid([a, b], par)
    assert len(lines) == 1
    fused = stitch.fuse(lines[0], 1, 0, par)
    assert fused[(fused.t > 10.5) & (fused.t < 15.5)].empty  # nothing was invented inside the gap
    assert fused.segment.max() == 1
    far = _path("cam2", 12, 22, (60, 0), (70, 0), tid=3)  # 50 m in 2 s: not the same person
    lines, reasons = stitch.stitch_gid([a, far], par)
    assert len(lines) == 2 and "m/s" in reasons[0]


def test_unreliable_far_field_and_stairs_are_left_unmapped():
    from yoyo.calibrate import Calibration
    cal = Calibration()
    fl, cls = cal.to_floor("cam5", [[960, 1000], [960, 100], [960, 300]])  # near ground, above the horizon, on the stairs
    assert cls[0] in ("reliable", "approximate") and np.isfinite(fl[0]).all()
    assert cls[1] == "unmapped" and np.isnan(fl[1]).all()
    assert cls[2] == "unmapped" and np.isnan(fl[2]).all()
