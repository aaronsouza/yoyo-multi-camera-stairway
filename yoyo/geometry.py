"""Bonus calibration geometry: ground-plane orientation of every camera from the paving seams (no landmark picking needed).

The plaza is paved with rectangular slabs, so its seams form two families of lines that are perpendicular on the ground (a "Manhattan"
frame). With one focal length shared by all cameras (same camera model) and the principal point at the image centre, each camera's rotation
follows from fitting its seam lines. That makes each camera's ground plane metric up to a scale (the camera height); cameras are then tied to
one common plan with a 4-degree-of-freedom similarity from a few landmarks (see register.py).

Not modelled: lens distortion (error grows towards the image edge) and the staircase, which is not on the plaza plane.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# Rough ground masks (fractions of the frame) that keep seam lines of the plaza and drop sky, building facade, stairs and the far crowd.
# Chosen by eye from stills; the optimisation treats every other line as an outlier.
GROUND = {
    "cam1": [[0.0, 0.30], [0.86, 0.20], [0.86, 1.0], [0.0, 1.0]],
    "cam2": [[0.0, 0.38], [0.66, 0.30], [1.0, 0.14], [1.0, 1.0], [0.0, 1.0]],
    "cam3": [[0.15, 0.20], [1.0, 0.18], [1.0, 1.0], [0.15, 1.0]],
    "cam4": [[0.0, 0.50], [0.6, 0.38], [1.0, 0.44], [1.0, 1.0], [0.0, 1.0]],
    "cam5": [[0.0, 0.42], [1.0, 0.42], [1.0, 1.0], [0.0, 1.0]],
    "cam6": [[0.05, 0.47], [0.62, 0.44], [1.0, 0.50], [1.0, 1.0], [0.05, 1.0]],
    "cam7": [[0.0, 0.30], [1.0, 0.16], [1.0, 1.0], [0.0, 1.0]],
}


# Regions where a foot point is NOT on the plaza plane: the stairs. Foot points elsewhere (and not too close to the horizon) are used for
# registration and trajectories. Chosen by eye from stills; polygons in fractions of the frame.
STAIRS = {
    "cam1": [[0.86, 0.0], [1.0, 0.0], [1.0, 1.0], [0.86, 1.0]],
    "cam2": [[0.0, 0.0], [0.70, 0.0], [0.70, 0.10], [0.15, 0.36], [0.0, 0.36]],
    "cam3": [[0.0, 0.18], [0.14, 0.18], [0.14, 1.0], [0.0, 1.0]],
    "cam4": [[0.88, 0.93], [1.0, 0.88], [1.0, 1.0], [0.88, 1.0]],
    "cam5": [[0.0, 0.0], [1.0, 0.0], [1.0, 0.41], [0.0, 0.41]],
    "cam6": [[0.47, 0.0], [1.0, 0.0], [1.0, 0.47], [0.48, 0.15]],
    "cam7": [],
}


def seam_segments(bg, cam, min_len=60):
    """Straight line segments (x1, y1, x2, y2) of the ground region of a people-free background image."""
    g = cv2.GaussianBlur(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(g)[0]
    if lines is None:
        return np.zeros((0, 4))
    s = lines.reshape(-1, 4)
    s = s[np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]) >= min_len]
    H, W = g.shape
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [(np.array(GROUND[cam]) * [W, H]).astype(np.int32)], 255)
    mid = ((s[:, :2] + s[:, 2:]) / 2).astype(int)
    return s[mask[np.clip(mid[:, 1], 0, H - 1), np.clip(mid[:, 0], 0, W - 1)] > 0]


def K_of(f, W, H):
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])


def line_normals(seg, f, W, H):
    """Unit normals (in camera coordinates) of the planes through the camera centre and each image segment."""
    K = K_of(f, W, H)
    p1 = np.c_[seg[:, :2], np.ones(len(seg))]
    p2 = np.c_[seg[:, 2:], np.ones(len(seg))]
    m = np.cross(np.linalg.solve(K, p1.T).T, np.linalg.solve(K, p2.T).T)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def residuals(R, normals):
    """Sine of the angle between each seam plane and the nearer of the two perpendicular ground directions R[:,0], R[:,1]."""
    r = np.abs(normals @ R[:, :2])
    return r.min(axis=1)


def init_rotation(seg, f, W, H, rng=np.random.default_rng(0), iters=3000):
    """Start value: the two perpendicular ground directions that explain the most line length (random pairs of lines -> vanishing points)."""
    n = line_normals(seg, f, W, H)
    w = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    best, bs = None, -1
    for _ in range(iters):
        i, j = rng.choice(len(seg), 2, replace=False)
        d1 = np.cross(n[i], n[j])
        if np.linalg.norm(d1) < 1e-6:
            continue
        d1 /= np.linalg.norm(d1)
        inl1 = np.abs(n @ d1) < 0.02
        # the perpendicular family: directions orthogonal to d1 explained by the remaining lines
        rest = np.flatnonzero(~inl1)
        if len(rest) < 5:
            continue
        for _ in range(8):
            a, b = rng.choice(rest, 2, replace=False)
            d2 = np.cross(n[a], n[b])
            d2 = d2 - d1 * (d1 @ d2)
            if np.linalg.norm(d2) < 1e-6:
                continue
            d2 /= np.linalg.norm(d2)
            sc = w[(np.abs(n @ d1) < 0.02) | (np.abs(n @ d2) < 0.02)].sum()
            if sc > bs:
                best, bs = (d1, d2), sc
    if best is None:
        raise ValueError("no two perpendicular seam families found (too few lines in the ground region)")
    d1, d2 = best
    return np.c_[d1, d2, np.cross(d1, d2)]


def fit_all(segs, sizes, f0=1550.0):
    """Joint fit of one shared focal length and every camera's rotation. segs: cam -> segments; sizes: cam -> (W, H)."""
    cams = sorted(segs)
    R0 = {c: init_rotation(segs[c], f0, *sizes[c]) for c in cams}
    x0 = np.r_[np.log(f0), np.concatenate([Rotation.from_matrix(R0[c]).as_rotvec() for c in cams])]
    w = {c: np.sqrt(np.hypot(segs[c][:, 2] - segs[c][:, 0], segs[c][:, 3] - segs[c][:, 1])) for c in cams}
    scale = np.sin(np.radians(0.7))

    def fun(x):
        f = np.exp(x[0])
        out = []
        for k, c in enumerate(cams):
            R = Rotation.from_rotvec(x[1 + 3 * k:4 + 3 * k]).as_matrix()
            out.append(residuals(R, line_normals(segs[c], f, *sizes[c])) * w[c])
        return np.concatenate(out)

    sol = least_squares(fun, x0, loss="soft_l1", f_scale=scale * 7.0, max_nfev=200)
    f = float(np.exp(sol.x[0]))
    R = {c: Rotation.from_rotvec(sol.x[1 + 3 * k:4 + 3 * k]).as_matrix() for k, c in enumerate(cams)}
    stats = {}
    for c in cams:
        r = np.degrees(np.arcsin(np.clip(residuals(R[c], line_normals(segs[c], f, *sizes[c])), 0, 1)))
        L = np.hypot(segs[c][:, 2] - segs[c][:, 0], segs[c][:, 3] - segs[c][:, 1])
        ok = r < 1.0
        stats[c] = dict(lines=int(len(r)), inlier_frac=float(L[ok].sum() / L.sum()), median_err_deg=float(np.median(r[ok])) if ok.any() else None)
    return f, R, stats


def ground_homography(R, f, W, H):
    """3x3 map from pixels to ground-plane coordinates (a, b), in units of the camera height h, as a right-handed top-down plan
    (the plan's X axis is one seam family; the 90-degree ambiguity is settled when cameras are registered to each other)."""
    n = R[:, 2] * (1 if R[1, 2] > 0 else -1)  # plane normal pointing from the camera down to the ground (camera y axis points down)
    d1 = R[:, 0]
    d2 = -np.cross(n, d1)  # right-handed when seen from above
    return np.vstack([d1, d2, n]) @ np.linalg.inv(K_of(f, W, H))


def to_plan(G, xy):
    p = np.c_[np.asarray(xy, float).reshape(-1, 2), np.ones(len(np.atleast_2d(xy)))] @ G.T
    return p[:, :2] / p[:, 2:3], p[:, 2]


def rectify(bg, G, extent, ppu):
    """Top-down image of the ground of `bg` over extent=(a0, a1, b0, b1) (camera-height units) at ppu pixels per unit."""
    a0, a1, b0, b1 = extent
    Wp, Hp = int((a1 - a0) * ppu), int((b1 - b0) * ppu)
    S = np.array([[ppu, 0, -a0 * ppu], [0, -ppu, b1 * ppu], [0, 0, 1.0]])  # plan -> output pixels (b up)
    return cv2.warpPerspective(bg, S @ G, (Wp, Hp), flags=cv2.INTER_LINEAR)


def camera_height(G, R, f, W, H, boxes, person_m=1.7, max_n=3000, seed=0):
    """Camera height (metres) from the pixel height of people standing on the ground plane, assuming adults are about `person_m` tall.
    boxes: array (x1, y1, x2, y2). Exact pinhole geometry: the head is the foot point lifted by person_m along the plane normal.
    Returns (median, spread) over many detections; accurate to the spread of human height and of the box tops (roughly +-15%)."""
    rng = np.random.default_rng(seed)
    b = boxes[(boxes[:, 3] - boxes[:, 1] > 60) & (boxes[:, 3] < H - 6) & (boxes[:, 1] > 6)]
    b = b[rng.choice(len(b), min(max_n, len(b)), replace=False)]
    K = K_of(f, W, H)
    n = R[:, 2] * (1 if R[1, 2] > 0 else -1)
    foot = np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3], np.ones(len(b))]
    r = np.linalg.solve(K, foot.T).T                       # rays through the foot pixels
    C = r @ n                                              # = n . r  (>0 for ground pixels)
    ok = C > 0.05
    r, C, b = r[ok], C[ok], b[ok]
    meas = b[:, 3] - b[:, 1]

    def height_px(h):  # pixel length of a person_m tall person standing at the foot ray, camera at height h
        Xf = (h / C)[:, None] * r
        Xh = Xf - n[None, :] * person_m
        pf = (K @ Xf.T).T
        ph = (K @ Xh.T).T
        return np.abs(ph[:, 1] / ph[:, 2] - pf[:, 1] / pf[:, 2])

    lo, hi = np.full(len(b), 0.5), np.full(len(b), 40.0)  # bisection per detection: height_px decreases with h
    for _ in range(40):
        mid = (lo + hi) / 2
        big = height_px(mid) > meas
        lo = np.where(big, mid, lo)
        hi = np.where(big, hi, mid)
    h = (lo + hi) / 2
    return float(np.median(h)), float(np.percentile(h, 75) - np.percentile(h, 25))
