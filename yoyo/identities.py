"""Stage 3: tracklets -> global person IDs (appearance + time + same-camera constraints)."""
import numpy as np
import pandas as pd
from .common import cam_dir, final_dir


def agglomerate(S, conflict, thr):
    """Average-linkage clustering with cannot-link constraints that propagate on merge."""
    n = len(S)
    if n == 0:
        return np.array([], int)
    NEG = -9.0
    sums, size = S.astype(np.float64).copy(), np.ones(n)
    active, conflict = np.ones(n, bool), conflict.copy()
    avg = np.where(conflict, NEG, S).astype(np.float64)
    np.fill_diagonal(avg, NEG)
    bj = avg.argmax(1)
    bv = avg[np.arange(n), bj]
    label = np.arange(n)
    while True:
        i = int(bv.argmax())
        if bv[i] < thr:
            break
        j = int(bj[i])
        label[label == j] = i
        sums[i] += sums[j]
        sums[:, i] = sums[i]
        size[i] += size[j]
        active[j] = False
        conflict[i] |= conflict[j]
        conflict[:, i] = conflict[i]
        row = np.where(conflict[i] | ~active, NEG, sums[i] / (size[i] * size))
        row[i] = NEG
        avg[i] = row
        avg[:, i] = row
        avg[j] = NEG
        avg[:, j] = NEG
        bv[j] = NEG
        for r in np.where(active)[0]:
            if r == i or bj[r] in (i, j):
                bj[r] = avg[r].argmax()
                bv[r] = avg[r, bj[r]]
            elif avg[r, i] > bv[r]:
                bj[r], bv[r] = i, avg[r, i]
    return label


def build_tracklets(cfg, cams):
    rows, embs = [], []
    off = cfg.get("offsets_s", {})
    for cam in cams:
        cd = cam_dir(cfg, cam)
        if not (cd / "tracks.csv").exists() or not (cd / "embeddings.npz").exists():
            continue
        tr = pd.read_csv(cd / "tracks.csv")
        z = np.load(cd / "embeddings.npz")
        zt, ze = z["tid"], z["emb"]  # load once: NpzFile re-decompresses on every access
        order = np.argsort(zt, kind="stable")
        zt, ze = zt[order], ze[order]
        starts = np.r_[0, np.flatnonzero(np.diff(zt)) + 1, len(zt)]
        e = {}
        for i0, i1 in zip(starts[:-1], starts[1:]):
            m = ze[i0:i1].mean(0)
            e[zt[i0]] = m / (np.linalg.norm(m) + 1e-9)
        g = tr.groupby("tid").agg(n=("frame", "size"), t0=("t", "min"), t1=("t", "max")).reset_index()
        g = g[(g.n >= cfg["identities"]["min_track_frames"]) & g.tid.isin(e)]
        for r in g.itertuples():
            rows.append(dict(cam=cam, tid=r.tid, n=r.n, t0=r.t0 + off.get(cam, 0), t1=r.t1 + off.get(cam, 0)))
            embs.append(e[r.tid])
    return pd.DataFrame(rows), (np.stack(embs) if embs else np.zeros((0, 1)))


def assign_global_ids(tl, E, cfg):
    c = cfg["identities"]
    if tl.empty:
        return tl.assign(gid=[])
    S = E @ E.T
    cam = tl.cam.values
    same = cam[:, None] == cam[None, :]
    S = S - c["same_cam_penalty"] * same
    t0, t1 = tl.t0.values, tl.t1.values
    overlap = np.minimum(t1[:, None], t1[None]) - np.maximum(t0[:, None], t0[None])
    gap = -overlap
    conflict = np.where(same, (overlap > c["overlap_tol_s"]) | (gap > c["reconnect_gap_s"]),
                        gap > c["max_gap_cross_s"])
    np.fill_diagonal(conflict, False)
    lab = agglomerate(S, conflict, c["thr"])
    _, gid = np.unique(lab, return_inverse=True)
    return tl.assign(gid=gid + 1)


def run(cfg, cams, log=print):
    tl, E = build_tracklets(cfg, cams)
    tl = assign_global_ids(tl, E, cfg)
    tl.to_csv(final_dir(cfg) / "tracklets.csv", index=False)
    log(f"[identities] {len(tl)} tracklets -> {tl.gid.nunique() if len(tl) else 0} unique people")
    return tl
