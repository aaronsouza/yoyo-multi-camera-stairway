"""Build a blind review set for the track stitching: pairs of same-moment crops from two cameras.

Positive pairs: two cameras that the stitching put on ONE trajectory. Negative pairs: two cameras whose tracklets share a global ID but were
split into DIFFERENT trajectories (the stitching says: two people). Each pair is shown as two crops side by side with only an index on it;
the answer key goes to a separate file so the reviewer can judge "same person / different people / cannot tell" without knowing.
Usage: python tools/stitch_review.py [n_pos=24] [n_neg=12] [seed=7]"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yoyo.calibrate import out_dir
from yoyo.common import ROOT, cam_dir, load_config, videos


def candidates(mem, same_line):
    rows = []
    for gid, g in mem.groupby("gid"):
        g = g.reset_index(drop=True)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g.iloc[i], g.iloc[j]
                if a.cam == b.cam or (a.line_id == b.line_id) != same_line:
                    continue
                lo, hi = max(a.t0, b.t0), min(a.t1, b.t1)
                if hi - lo >= 2.0:
                    rows.append(dict(gid=gid, line_a=a.line_id, line_b=b.line_id, cam_a=a.cam, tid_a=a.tid, cam_b=b.cam, tid_b=b.tid, t=(lo + hi) / 2))
    return pd.DataFrame(rows)


def crop(cfg, cap, tracks, tid, t, fps):
    g = tracks[(tracks.tid == tid)]
    r = g.iloc[(g.t - t).abs().argsort().iloc[0]]
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame))
    ok, fr = cap.read()
    if not ok:
        return None
    h = r.y2 - r.y1
    if h < 90 or r.y2 > 1074 or r.x1 < 4 or r.x2 > 1916:
        return None
    pad = int(0.08 * h)
    c = fr[max(int(r.y1) - pad, 0):int(r.y2) + pad, max(int(r.x1) - pad, 0):int(r.x2) + pad]
    return cv2.resize(c, (int(c.shape[1] * 260 / c.shape[0]), 260))


def main(n_pos=24, n_neg=12, seed=7):
    cfg = load_config()
    mem = pd.read_csv(out_dir(cfg) / "line_members.csv")
    rng = np.random.default_rng(seed)
    vids, caps, tr = videos(cfg), {}, {}
    pos = candidates(mem, True)
    neg = candidates(mem, False)
    pos = pos.drop_duplicates("line_a").sample(frac=1, random_state=seed)
    neg = neg.drop_duplicates("gid").sample(frac=1, random_state=seed)
    picked = []
    for kind, df, n in (("positive", pos, n_pos), ("negative", neg, n_neg)):
        got = 0
        for r in df.itertuples():
            for c in (r.cam_a, r.cam_b):
                if c not in tr:
                    tr[c] = pd.read_csv(cam_dir(cfg, c) / "tracks.csv")
                    caps[c] = cv2.VideoCapture(str(vids[c]))
            ca = crop(cfg, caps[r.cam_a], tr[r.cam_a], r.tid_a, r.t, 59.94)
            cb = crop(cfg, caps[r.cam_b], tr[r.cam_b], r.tid_b, r.t, 59.94)
            if ca is None or cb is None:
                continue
            picked.append((kind, r, ca, cb))
            got += 1
            if got == n:
                break
    order = rng.permutation(len(picked))
    out = ROOT / "evidence" / "bonus" / "stitch_review"
    out.mkdir(parents=True, exist_ok=True)
    key = []
    per = 6
    for page in range(0, len(order), per):
        rows = []
        for k, idx in enumerate(order[page:page + per]):
            kind, r, ca, cb = picked[idx]
            n = page + k + 1
            row = np.full((276, 520, 3), 30, np.uint8)
            row[8:8 + ca.shape[0], 8:8 + min(ca.shape[1], 240)] = ca[:, :240]
            row[8:8 + cb.shape[0], 268:268 + min(cb.shape[1], 240)] = cb[:, :240]
            cv2.putText(row, f"#{n}", (250, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            rows.append(row)
            key.append(dict(n=n, expected=kind, gid=r.gid, line_a=r.line_a, line_b=r.line_b, cam_a=r.cam_a, tid_a=r.tid_a, cam_b=r.cam_b, tid_b=r.tid_b, t=round(r.t, 1)))
        while len(rows) % 2:
            rows.append(np.zeros_like(rows[0]))
        grid = np.vstack([np.hstack(rows[i:i + 2]) for i in range(0, len(rows), 2)])
        cv2.imwrite(str(out / f"review_{page // per + 1}.png"), grid)
    pd.DataFrame(key).to_csv(out / "_answer_key.csv", index=False)
    print(len(picked), "pairs;", len(range(0, len(order), per)), "images in", out)


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:]]
    main(*a)
