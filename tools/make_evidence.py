"""Checklist 1 evidence: one 30 s annotated clip per camera, drawn from saved tracks (no inference), plus a check that
the boxes drawn match tracks.csv and that every clip decodes. Usage: python tools/make_evidence.py [--cams cam1 ...]"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yoyo.common import ROOT, cam_dir, load_config, read_json, videos
from yoyo.render import render_camera

DUR = 30


def busiest_window(tracks, fps, stride, n_frames, dur=DUR):
    """Start second of the `dur`-second window with the most tracked people per sampled frame (a representative scene)."""
    per_s = tracks.groupby((tracks.frame // stride * stride / fps).astype(int)).size()
    per_s = per_s.reindex(range(int(n_frames / fps)), fill_value=0)
    roll = per_s.rolling(dur).sum().shift(-(dur - 1))
    return int(roll.idxmax())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", nargs="*")
    a = ap.parse_args()
    cfg = load_config()
    vids = videos(cfg)
    out_dir = ROOT / "evidence" / "checklist1_tracking"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for cam in a.cams or sorted(vids):
        meta = read_json(cam_dir(cfg, cam) / "progress.json")
        tr = pd.read_csv(cam_dir(cfg, cam) / "tracks.csv")
        fps, stride = meta["fps"], meta["stride"]
        s0 = busiest_window(tr, fps, stride, meta["n_frames"])
        out = out_dir / f"{cam}_{s0}-{s0 + DUR}s.mp4"
        stats = []
        render_camera(cfg, cam, vids[cam], s0, s0 + DUR, force=True, out_path=out, stats=stats)
        frames = [f for f, _ in stats]
        expect = tr[tr.frame.isin(frames)].groupby("frame").size().reindex(frames, fill_value=0)
        drawn = np.array([n for _, n in stats])
        match = bool((drawn == expect.values).all())
        cap = cv2.VideoCapture(str(out))
        n_out, w, h, rate = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(3)), int(cap.get(4)), cap.get(5)
        sc = w / meta["width"]
        rng, hit, tot = np.random.default_rng(0), 0, 0
        for i in rng.choice(len(frames), min(25, len(frames)), replace=False):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if not ok:
                continue
            for r in tr[tr.frame == frames[i]].itertuples():
                x, y = int(r.x1 * sc), int(r.y1 * sc)
                patch = fr[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2].reshape(-1, 3).astype(int)
                bg = fr[max(y - 6, 0), max(x - 6, 0)].astype(int)
                tot += 1
                hit += int(np.abs(patch - bg).sum(1).max() > 40)
        cap.release()
        report[cam] = dict(file=str(out.relative_to(ROOT)), start_s=s0, end_s=s0 + DUR, frames_out=n_out, expected_frames=len(frames),
                           fps=round(rate, 2), size=f"{w}x{h}", mb=round(out.stat().st_size / 1e6, 1),
                           mean_people_per_frame=round(float(drawn.mean()), 1), boxes_drawn=int(drawn.sum()),
                           boxes_match_tracks_csv=match, box_pixels_found=f"{hit}/{tot}")
        print(cam, report[cam])
    (out_dir / "check.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
