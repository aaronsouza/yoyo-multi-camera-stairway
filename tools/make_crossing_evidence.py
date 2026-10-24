"""Checklist 2 evidence: 30 s clip of completed crossings (direction arrow, track/identity IDs, event labels) drawn from the saved
tracks and events.csv, plus a review sheet to tick off by hand. No inference. Usage: python tools/make_crossing_evidence.py cam5 1500 1620"""
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yoyo.common import ROOT, cam_dir, final_dir, load_config, load_zones, read_json, videos  # noqa: E402

DUR = 30
GREEN, RED, GREY = (60, 200, 60), (50, 50, 240), (170, 170, 170)


def main(cam, lo, hi):
    cfg = load_config()
    meta = read_json(cam_dir(cfg, cam) / "progress.json")
    fps, stride, W, H = meta["fps"], meta["stride"], meta["width"], meta["height"]
    ev = pd.read_csv(final_dir(cfg) / "events.csv")
    ev = ev[ev.cams.str.split(";").apply(lambda c: cam in c)].reset_index(drop=True)
    tl = pd.read_csv(final_dir(cfg) / "tracklets.csv")
    tr = pd.read_csv(cam_dir(cfg, cam) / "tracks.csv")
    gid = dict(zip(tl[tl.cam == cam].tid, tl[tl.cam == cam].gid))
    tr["gid"] = tr.tid.map(gid)
    # review sheet for the whole interval (for the human to mark TP / FP / duplicate / wrong direction)
    sheet = ev[ev.t_end.between(lo, hi)].copy()
    sheet["video_time"] = sheet.t_end.apply(lambda t: f"{int(t // 60):02d}:{t % 60:04.1f}")
    for col in ("verdict (TP / FP / DUP / WRONG_DIR)", "notes"):
        sheet[col] = ""
    out_dir = ROOT / "evidence" / "checklist2_crossings"
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet[["event_id", "video_time", "direction", "gid", "t_start", "t_end", "inferred", "confirmed_by", "entity",
           "verdict (TP / FP / DUP / WRONG_DIR)", "notes"]].to_csv(out_dir / f"{cam}_{lo}-{hi}_review_sheet.csv", index=False)
    # busiest 30 s window of completed crossings inside [lo, hi]
    best = max(range(lo, hi - DUR + 1), key=lambda s: ev.t_end.between(s, s + DUR).sum())
    s0, s1 = best, best + DUR
    out = out_dir / f"{cam}_crossings_{s0}-{s1}s.mp4"
    sc = 1280 / W
    ow, oh = 1280, int(H * sc) // 2 * 2
    zones = load_zones()[cam]
    by_frame = {f: g for f, g in tr[(tr.t >= s0 - 1) & (tr.t <= s1 + 1)].groupby("frame")}
    cap = cv2.VideoCapture(str(videos(cfg)[cam]))
    f0 = int(np.ceil(s0 * fps / stride)) * stride
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}",
           "-r", str(fps / stride), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    # path of each event: foot points of its identity in this camera between t_start and t_end
    paths = {}
    for r in ev.itertuples():
        g = tr[(tr.gid == r.gid) & tr.t.between(r.t_start, r.t_end)].sort_values("t")
        paths[r.event_id] = np.c_[(g.x1 + g.x2) / 2 * sc, g.y2 * sc], g.t.values
    f = f0
    while f / fps <= s1 and cap.grab():
        if f % stride == 0:
            ok, fr = cap.retrieve()
            if ok:
                t = f / fps
                fr = cv2.resize(fr, (ow, oh))
                ov = fr.copy()
                for name, poly in zones.items():
                    cv2.fillPoly(ov, [(np.array(poly) * [ow, oh]).astype(np.int32)], (0, 200, 255) if name == "top" else (255, 180, 0))
                fr = cv2.addWeighted(ov, 0.15, fr, 0.85, 0)
                act = ev[(ev.t_start - 0.5 <= t) & (t <= ev.t_end + 3.0)]
                hot = {}
                for r in act.itertuples():
                    hot.setdefault(r.gid, []).append(r)
                for r in by_frame.get(f, pd.DataFrame()).itertuples():
                    x1, y1, x2, y2 = (int(v * sc) for v in (r.x1, r.y1, r.x2, r.y2))
                    evs = hot.get(r.gid)
                    lab = f"T{r.tid % 100000}" + (f" P{int(r.gid)}" if r.gid == r.gid else "")
                    if evs:
                        e = evs[-1]
                        col = GREEN if e.direction == "UP" else RED
                        cv2.rectangle(fr, (x1, y1), (x2, y2), col, 3)
                        done = t >= e.t_end
                        tag = f"E{e.event_id} {e.direction}{' (edge-inferred)' if e.inferred else ''}{' DONE' if done else ''}"
                        cv2.putText(fr, tag, (x1, max(14, y1 - 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
                        cv2.putText(fr, lab, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
                    else:
                        cv2.rectangle(fr, (x1, y1), (x2, y2), GREY, 1)
                        cv2.putText(fr, lab, (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1, cv2.LINE_AA)
                for r in act.itertuples():  # direction arrow: where the person was at the start of the crossing -> where they are now
                    pts, ts = paths[r.event_id]
                    if len(pts) < 2:
                        continue
                    col = GREEN if r.direction == "UP" else RED
                    upto = pts[ts <= t] if t < r.t_end else pts
                    if len(upto) >= 2:
                        cv2.polylines(fr, [upto.astype(np.int32)], False, col, 2, cv2.LINE_AA)
                        cv2.arrowedLine(fr, tuple(upto[-2].astype(int)), tuple(upto[-1].astype(int)), col, 3, tipLength=3.0)
                done = ev[(ev.t_end >= s0) & (ev.t_end <= t)]
                cv2.rectangle(fr, (0, 0), (ow, 28), (0, 0, 0), -1)
                cv2.putText(fr, f"{cam}  video {int(t // 60):02d}:{t % 60:04.1f} (t={t:.1f}s)   crossings completed in this clip: "
                            f"UP {(done.direction == 'UP').sum()}  DOWN {(done.direction == 'DOWN').sum()}   | green=UP red=DOWN, arrow = path walked",
                            (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                p.stdin.write(fr.tobytes())
        f += 1
    p.stdin.close()
    p.wait()
    inw = ev[ev.t_end.between(s0, s1)]
    inw[["event_id", "gid", "direction", "t_start", "t_end", "inferred"]].to_csv(out_dir / f"{cam}_crossings_{s0}-{s1}s_events.csv", index=False)
    print(out, "events in clip:", len(inw), inw.direction.value_counts().to_dict())


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
