"""Bonus calibration step 2: tie the camera ground planes together WITHOUT using ReID.

After geometry.py every camera has a metric top-down plan (up to the camera height, estimated from people's heights). Because all cameras see the
same slab grid, two plans differ only by a rotation that is a multiple of 90 degrees, a small scale correction and a translation. Cameras are
time-synchronised, so every pair of foot points seen at the same instant by two cameras votes for one translation; the true translation collects
votes from every shared person, while chance pairings spread out. Each pair of tracks votes once in total (weight 1 / frames seen together), so
people who stand still do not dominate. Identities are never used here, so the later stitching test is independent of the calibration.
"""
import numpy as np
import pandas as pd

from .geometry import GROUND, STAIRS, ground_homography, to_plan

ROT = [np.array([[1, 0], [0, 1]]), np.array([[0, -1], [1, 0]]), np.array([[-1, 0], [0, -1]]), np.array([[0, 1], [-1, 0]])]


def plan_points(tracks, G, cam, W, H, h_m, step=30, min_c=0.14, min_box=40):
    """Foot points of one camera in its own metric plan (metres), reliable detections only, sampled every `step` frames."""
    t = tracks[(tracks.frame % step == 0) & (tracks.y2 < H - 6) & (tracks.y2 - tracks.y1 > min_box) & (tracks.x1 > 4) & (tracks.x2 < W - 4)].copy()
    xy = np.c_[(t.x1 + t.x2) / 2, t.y2]
    ab, C = to_plan(G, xy)
    # keep points on the plaza plane (inside the ground mask) and not too close to the horizon, where one pixel is many metres
    from matplotlib.path import Path as MPath
    on_stairs = MPath(np.array(STAIRS[cam]) * [W, H]).contains_points(xy) if STAIRS.get(cam) else np.zeros(len(xy), bool)
    ok = ~on_stairs & (C > min_c)
    t = t[ok]
    return pd.DataFrame(dict(frame=t.frame.values, tid=t.tid.values, x=ab[ok, 0] * h_m, y=ab[ok, 1] * h_m))


def vote_pair(A, B, scales, bin_m=0.5, span_m=150.0):
    """Best (rotation, scale, tx, ty) taking B's plan into A's plan, from same-frame foot point pairs. Returns dict with the vote evidence."""
    m = A.merge(B, on="frame", suffixes=("_a", "_b"))
    if len(m) == 0:
        return None
    n = m.groupby(["tid_a", "tid_b"]).frame.transform("size").values
    w = 1.0 / n
    nb = int(2 * span_m / bin_m)
    edges = np.linspace(-span_m, span_m, nb + 1)
    best = None
    for r, Rm in enumerate(ROT):
        xb = m.x_b.values * Rm[0, 0] + m.y_b.values * Rm[0, 1]
        yb = m.x_b.values * Rm[1, 0] + m.y_b.values * Rm[1, 1]
        for s in scales:
            dx, dy = m.x_a.values - s * xb, m.y_a.values - s * yb
            H, _, _ = np.histogram2d(dx, dy, bins=[edges, edges], weights=w)
            Hs = H + np.roll(H, 1, 0) + np.roll(H, -1, 0) + np.roll(H, 1, 1) + np.roll(H, -1, 1) + np.roll(np.roll(H, 1, 0), 1, 1) \
                + np.roll(np.roll(H, 1, 0), -1, 1) + np.roll(np.roll(H, -1, 0), 1, 1) + np.roll(np.roll(H, -1, 0), -1, 1)  # 3x3 sum (1.5 m)
            i = np.unravel_index(Hs.argmax(), Hs.shape)
            peak = float(Hs[i])
            if best is None or peak > best["peak"]:
                best = dict(rot=r, scale=float(s), tx=float((edges[i[0]] + edges[i[0] + 1]) / 2), ty=float((edges[i[1]] + edges[i[1] + 1]) / 2),
                            peak=peak, background=float(np.median(Hs[Hs > 0])) if (Hs > 0).any() else 0.0, p99=float(np.percentile(Hs[Hs > 0], 99)),
                            pairs=int(len(np.unique(m[["tid_a", "tid_b"]].values, axis=0))))
    return best


def apply(T, xy):
    """Map points of the source plan to the target plan with a registration dict from vote_pair."""
    Rm = ROT[T["rot"]]
    p = np.asarray(xy, float) @ Rm.T * T["scale"]
    return p + np.array([T["tx"], T["ty"]])


def sim_matrix(T):
    """3x3 similarity of a registration dict: plan of the camera -> common plan."""
    M = np.eye(3)
    M[:2, :2] = T["scale"] * ROT[T["rot"]] @ rot2(T.get("dtheta", 0.0))
    M[:2, 2] = [T["tx"], T["ty"]]
    return M


def rot2(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def to_common(T, xy):
    return (np.c_[np.asarray(xy, float), np.ones(len(xy))] @ sim_matrix(T).T)[:, :2]


def cloud(P, T, cams):
    """All registered cameras' points in the common plan, with globally unique track ids."""
    out = []
    for i, c in enumerate(cams):
        d = P[c].copy()
        xy = to_common(T[c], d[["x", "y"]].values)
        d["x"], d["y"] = xy[:, 0], xy[:, 1]
        d["tid"] = d.tid + (i + 1) * 10 ** 7
        out.append(d)
    return pd.concat(out)


def matches(P, T, cams, gate=2.5):
    """Mutual nearest-neighbour pairs between cameras at the same frame in the current common plan: (cam_a, cam_b, pa, pb)."""
    from scipy.spatial import cKDTree
    pts = {}
    for c in cams:
        xy = to_common(T[c], P[c][["x", "y"]].values)
        pts[c] = (P[c].frame.values, xy)
    rows = []
    for i, a in enumerate(cams):
        for b in cams[i + 1:]:
            fa, xa = pts[a]
            fb, xb = pts[b]
            common = np.intersect1d(fa, fb)
            ia = np.argsort(fa, kind="stable")
            ib = np.argsort(fb, kind="stable")
            sa, sb = fa[ia], fb[ib]
            for fr in common:
                ra = ia[np.searchsorted(sa, fr):np.searchsorted(sa, fr, "right")]
                rb = ib[np.searchsorted(sb, fr):np.searchsorted(sb, fr, "right")]
                ta, tb = cKDTree(xa[ra]), cKDTree(xb[rb])
                d, j = ta.query(xb[rb], distance_upper_bound=gate)
                for jb, (dd, ja) in enumerate(zip(d, j)):
                    if np.isfinite(dd):
                        d2, back = tb.query(xa[ra[ja]])
                        if back == jb:
                            rows.append((a, b, ra[ja], rb[jb], dd))
    return rows


def refine(P, T, cams, ref, iters=6, gate=2.0, log=print):
    """Joint trimmed-ICP refinement of every camera's (small rotation, scale, translation) against the other cameras, using same-frame
    mutual nearest neighbours as correspondences (no identities). Returns the updated registrations and match statistics."""
    from scipy.optimize import least_squares
    free = [c for c in cams if c != ref]
    for it in range(iters):
        M = matches(P, T, cams, gate)
        if not M:
            break
        xy = {c: P[c][["x", "y"]].values for c in cams}
        A = np.array([cams.index(m[0]) for m in M])
        B = np.array([cams.index(m[1]) for m in M])
        pa = np.array([xy[m[0]][m[2]] for m in M])
        pb = np.array([xy[m[1]][m[3]] for m in M])

        def pack(Tt):
            return np.array([[Tt[c].get("dtheta", 0.0), np.log(Tt[c]["scale"]), Tt[c]["tx"], Tt[c]["ty"]] for c in free]).ravel()

        def unpack(x):
            Tn = {c: dict(T[c]) for c in cams}
            for k, c in enumerate(free):
                Tn[c].update(dtheta=x[4 * k], scale=float(np.exp(x[4 * k + 1])), tx=x[4 * k + 2], ty=x[4 * k + 3])
            return Tn

        def fun(x):
            Tn = unpack(x)
            Ms = {c: sim_matrix(Tn[c]) for c in cams}
            ra = np.empty_like(pa)
            rb = np.empty_like(pb)
            for ci, c in enumerate(cams):
                s = A == ci
                if s.any():
                    ra[s] = (np.c_[pa[s], np.ones(s.sum())] @ Ms[c].T)[:, :2]
                s = B == ci
                if s.any():
                    rb[s] = (np.c_[pb[s], np.ones(s.sum())] @ Ms[c].T)[:, :2]
            return (ra - rb).ravel()

        sol = least_squares(fun, pack(T), loss="soft_l1", f_scale=0.8, max_nfev=60)
        T = unpack(sol.x)
        r = np.hypot(*fun(sol.x).reshape(-1, 2).T)
        log(f"  ICP {it}: {len(M)} matches, median residual {np.median(r):.2f} m")
    return T, M


def register_all(P, order, hm, log=print):
    """Register cameras one by one into the plan of order[0]: vote against everything registered so far, then refine jointly."""
    ref = order[0]
    T = {ref: dict(rot=0, scale=1.0, tx=0.0, ty=0.0, dtheta=0.0)}
    done = [ref]
    evidence = {}
    for c in order[1:]:
        A = cloud(P, T, done)
        sc = np.linspace(0.7, 1.45, 26)
        v = vote_pair(A, P[c], sc)
        evidence[c] = dict(peak=v["peak"], p99=v["p99"], ratio=v["peak"] / v["p99"], pairs=v["pairs"], scale=v["scale"])
        T[c] = dict(rot=v["rot"], scale=v["scale"], tx=v["tx"], ty=v["ty"], dtheta=0.0)
        done.append(c)
        log(f"[register] {c}: rot {v['rot'] * 90} deg, scale {v['scale']:.2f}, t=({v['tx']:.1f},{v['ty']:.1f}) evidence peak/p99 = {v['peak'] / v['p99']:.2f} ({v['pairs']} track pairs)")
        T, _ = refine(P, T, done, ref, iters=3, log=log)
    T, M = refine(P, T, done, ref, iters=6, log=log)
    return T, M, evidence
