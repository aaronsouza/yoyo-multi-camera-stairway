"""Checklist 3 evidence: two synchronized cameras side by side (same video time), global person IDs on every box, and every
cross-camera duplicate crossing that was merged shown with the SAME colour and label on both views. Drawn from saved tracks and
events.csv, no inference. Usage: python tools/make_multicam_evidence.py cam5 cam2"""
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yoyo.common import ROOT, cam_dir, final_dir, load_config, read_json, videos

DUR = 30
PAL = [(245, 130, 48), (200, 60, 200), (40, 220, 220), (66, 135, 245), (240, 200, 60), (150, 100, 240), (20, 140, 140), (240, 90, 120)]
GREEN, RED, GREY = (60, 200, 60), (50, 50, 240), (150, 150, 150)


def main(prim, sec):
    cfg = load_config()
    ev = pd.read_csv(final_dir(cfg) / "events.csv")
    ev["dedup_with"] = ev.dedup_with.fillna("")
    tl = pd.read_csv(final_dir(cfg) / "tracklets.csv")
    both = ev[ev.dedup_with.str.endswith("@" + sec)].copy()
    both["sec_gid"] = both.dedup_with.str.split("@").str[0].astype(int)
    s0 = max(range(0, 2070), key=lambda s: both.t_start.between(s, s + DUR - 1).sum())
    s1 = s0 + DUR
    out_dir = ROOT / "evidence" / "checklist3_dedup"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{prim}+{sec}_sync_{s0}-{s1}s.mp4"
    cams = {}
    for c in (prim, sec):
        m = read_json(cam_dir(cfg, c) / "progress.json")
        t = pd.read_csv(cam_dir(cfg, c) / "tracks.csv")
        t = t[(t.t >= s0 - 1) & (t.t <= s1 + 1)].copy()
        t["gid"] = t.tid.map(dict(zip(tl[tl.cam == c].tid, tl[tl.cam == c].gid)))
        cams[c] = dict(meta=m, by=dict(list(t.groupby("frame"))), cap=cv2.VideoCapture(str(videos(cfg)[c])))
    fps, stride = cams[prim]["meta"]["fps"], cams[prim]["meta"]["stride"]
    pw = 800
    ph = int(cams[prim]["meta"]["height"] * pw / cams[prim]["meta"]["width"]) // 2 * 2
    ow, oh = pw * 2, ph + 30
    f0 = int(np.ceil(s0 * fps / stride)) * stride
    for c in cams.values():
        c["cap"].set(cv2.CAP_PROP_POS_FRAMES, f0)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}",
           "-r", str(fps / stride), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    shown = {prim: [], sec: []}
    mcol = {e.event_id: PAL[k % len(PAL)] for k, e in enumerate(both.itertuples())}
    for e in ev.itertuples():
        cs = e.cams.split(";")
        matched = e.event_id in mcol
        col = mcol[e.event_id] if matched else (GREEN if e.direction == "UP" else RED)
        if prim in cs:
            shown[prim].append((e.gid, e.event_id, e.direction, e.t_start, e.t_end, matched, col))
        if matched:
            shown[sec].append((int(e.dedup_with.split("@")[0]), e.event_id, e.direction, e.t_start, e.t_end, True, col))
        elif sec in cs:
            shown[sec].append((e.gid, e.event_id, e.direction, e.t_start, e.t_end, False, col))
    f = f0
    while f / fps <= s1:
        panels, ok_all = [], True
        for name, c in cams.items():
            while c.get("pos", f0 - 1) < f:
                if not c["cap"].grab():
                    ok_all = False
                    break
                c["pos"] = c.get("pos", f0 - 1) + 1
            ok, fr = c["cap"].retrieve() if ok_all else (False, None)
            if not ok:
                ok_all = False
                break
            sc = pw / c["meta"]["width"]
            fr = cv2.resize(fr, (pw, ph))
            t = f / fps
            for k, r in enumerate(c["by"].get(f, pd.DataFrame()).itertuples()):
                x1, y1, x2, y2 = (int(v * sc) for v in (r.x1, r.y1, r.x2, r.y2))
                hit = [x for x in shown[name] if x[0] == r.gid and x[3] - 2.0 <= t <= x[4] + 3.0]
                gid_s = f"P{int(r.gid)}" if r.gid == r.gid else ""
                if hit:
                    gid_, eid, d, ts, te, matched, col = hit[-1]
                    lab = f"{gid_s} E{eid} {d}{' MATCH' if matched else ''}{' done' if t >= te else ''}"
                    cv2.rectangle(fr, (x1, y1), (x2, y2), col, 3)
                    ty = max(12, y1 - 4 - 13 * (k % 3))
                    (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(fr, (x1, ty - th - 2), (x1 + tw + 2, ty + 3), (0, 0, 0), -1)
                    cv2.putText(fr, lab, (x1 + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
                else:
                    cv2.rectangle(fr, (x1, y1), (x2, y2), GREY, 1)
                    cv2.putText(fr, gid_s, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1, cv2.LINE_AA)
            cv2.putText(fr, f"{name}  {int(t // 60):02d}:{t % 60:04.1f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            panels.append(fr)
        if not ok_all:
            break
        t = f / fps
        done = ev[(ev.t_end <= t) & (ev.t_end >= s0) & (ev.cams.str.contains(prim) | ev.cams.str.contains(sec))]
        nm = int(both[(both.t_end <= t) & (both.t_end >= s0)].shape[0])
        bar = np.zeros((30, ow, 3), np.uint8)
        cv2.putText(bar, f"{prim}|{sec}  green=UP red=DOWN | other colour = same crossing in both cameras (counted once) | grey = no crossing  "
                    f"| counted so far: UP {(done.direction == 'UP').sum()} DOWN {(done.direction == 'DOWN').sum()} (+{nm} duplicates removed)",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        p.stdin.write(np.vstack([bar, np.hstack(panels)]).tobytes())
        f += stride
        for c in cams.values():
            c["pos"] = c.get("pos", f0 - 1)
    p.stdin.close()
    p.wait()
    inw = both[both.t_start.between(s0, s1)]
    inw[["event_id", "gid", "sec_gid", "direction", "t_start", "t_end"]].rename(columns={"gid": f"gid_{prim}", "sec_gid": f"gid_{sec}"}) \
        .to_csv(out_dir / f"{prim}+{sec}_sync_{s0}-{s1}s_matches.csv", index=False)
    print(out, "matched crossings starting in clip:", len(inw))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
