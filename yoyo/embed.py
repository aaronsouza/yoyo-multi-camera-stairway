"""Stage 2: ReID embeddings on a few good crops per tracklet. Output: cams/<cam>/embeddings.npz"""
import cv2
import numpy as np
import pandas as pd
from .common import ROOT, cam_dir, read_json


def hsv_feat(crop):
    h = crop.shape[0]
    feats = []
    for a, b in ((0.15, 0.55), (0.55, 1.0)):  # torso, legs
        hsv = cv2.cvtColor(crop[int(a * h):int(b * h)], cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
        feats.append(np.sqrt(hist / (hist.sum() + 1e-9)))
    f = np.concatenate(feats)
    return f / (np.linalg.norm(f) + 1e-9)


class Embedder:
    def __init__(self, cfg):
        r = cfg["reid"]
        self.hw, self.net, self.backend = r.get("hsv_weight", 0.5), None, "hsv"
        if r["backend"] == "osnet":
            try:
                import torch
                import torchreid
                self.torch = torch
                m = torchreid.models.build_model("osnet_x0_25", num_classes=1000, pretrained=r.get("weights") is None)
                if r.get("weights"):
                    torchreid.utils.load_pretrained_weights(m, str(ROOT / r["weights"]))
                self.net, self.backend = m.eval(), "osnet"
            except Exception as e:
                print(f"[embed] WARNING: OSNet unavailable ({e}); falling back to colour-histogram ReID")

    def __call__(self, crops):
        hs = np.stack([hsv_feat(c) for c in crops])
        if self.net is None:
            return hs.astype(np.float32)
        t = self.torch
        x = np.stack([cv2.resize(c, (128, 256))[:, :, ::-1] for c in crops]).astype(np.float32) / 255
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        with t.no_grad():
            o = self.net(t.from_numpy(x.transpose(0, 3, 1, 2)).float()).numpy()
        o /= np.linalg.norm(o, axis=1, keepdims=True) + 1e-9
        return np.concatenate([o, self.hw * hs], 1).astype(np.float32)


def select_obs(tracks, meta, k, min_h):
    W, H = meta["width"], meta["height"]
    t = tracks.assign(h=tracks.y2 - tracks.y1)
    t = t[(t.h >= min_h) & (t.x1 > 3) & (t.y1 > 3) & (t.x2 < W - 3) & (t.y2 < H - 3)]
    sel = []
    for _, g in t.groupby("tid"):
        g = g.sort_values("frame")
        idx = np.linspace(0, len(g) - 1, min(k, len(g))).astype(int)
        sel.append(g.iloc[np.unique(idx)])
    return pd.concat(sel) if sel else t.iloc[:0]


def process_camera(cfg, cam, video, emb=None, force=False, log=print):
    cd = cam_dir(cfg, cam)
    out = cd / "embeddings.npz"
    prog = read_json(cd / "progress.json", {})
    if out.exists() and not force:
        log(f"[embed] {cam}: cached")
        return
    if not prog.get("done"):
        log(f"[embed] {cam}: detection not finished, skipping")
        return
    tracks = pd.read_csv(cd / "tracks.csv")
    r = cfg["reid"]
    n = tracks.groupby("tid").size()
    tracks = tracks[tracks.tid.isin(n[n >= cfg["identities"]["min_track_frames"]].index)]  # skip noise tracks
    sel = select_obs(tracks, prog, r["obs_per_track"], r["min_box_h"])
    emb = emb or Embedder(cfg)
    by_frame = {f: g for f, g in sel.groupby("frame")}
    cap = cv2.VideoCapture(str(video))
    tids, feats, buf, buft = [], [], [], []

    def run():
        if buf:
            feats.append(emb(buf))
            tids.extend(buft)
            buf.clear()
            buft.clear()

    pos = None  # sequential decode: far cheaper than a keyframe seek per crop on 60 fps H.264
    for f in sorted(by_frame):
        if pos is None or f < pos or f - pos > 400:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            pos = f
        while pos < f and cap.grab():
            pos += 1
        ok, fr = cap.read()
        pos += 1
        if not ok:
            continue
        for row in by_frame[f].itertuples():
            c = fr[int(row.y1):int(row.y2), int(row.x1):int(row.x2)]
            if c.size:
                buf.append(c)
                buft.append(row.tid)
        if len(buf) >= 32:
            run()
    run()
    np.savez_compressed(out, tid=np.array(tids), emb=np.concatenate(feats) if feats else np.zeros((0, 1)),
                        backend=emb.backend)
    log(f"[embed] {cam}: {len(tids)} crops ({emb.backend})")
