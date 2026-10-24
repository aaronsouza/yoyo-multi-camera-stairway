"""Bonus proof clip (30 s): the same person in two synchronized cameras next to their trajectory on the floor plan.
Everything is drawn from saved results (calibration.json, trajectories, tracks); the plan panel only shows observed positions.
Usage: python tools/make_bonus_evidence.py <line_id> <camA> <tidA> <camB> <tidB> <start_s>      e.g. 931-0 cam6 13636 cam7 12975 712"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yoyo.calibrate import Calibration, out_dir
from yoyo.common import ROOT, cam_dir, load_config, read_json, videos

DUR = 30
RED, GOLD, WHITE = (60, 60, 230), (0, 215, 255), (255, 255, 255)
CAM_BGR = {"cam1": (180, 119, 31), "cam2": (14, 127, 255), "cam3": (44, 160, 44), "cam4": (189, 103, 148), "cam5": (40, 39, 214), "cam6": (75, 86, 140), "cam7": (190, 190, 23)}


def main(line_id, cam_a, tid_a, cam_b, tid_b, s0):
    cfg = load_config()
    cal = Calibration()
    d = out_dir(cfg)
    meta = json.loads((d / "floorplan.json").read_text())
    mosaic = cv2.imread(str(d / "floorplan.png"))
    ext, ppm = meta["extent"], meta["ppm"]
    traj = pd.read_csv(d / "trajectories.csv.gz", dtype={"line_id": str})
    line = traj[traj.line_id == line_id].sort_values("t").reset_index(drop=True)
    s1 = s0 + DUR
    seg = line[line.t.between(s0 - 2, s1 + 2)]
    pts = np.r_[seg[["X", "Y"]].values, [cal.cal["cameras"][cam_a]["position_m"], cal.cal["cameras"][cam_b]["position_m"]]]
    x0, y0, x1, y1 = pts[:, 0].min() - 5, pts[:, 1].min() - 5, pts[:, 0].max() + 5, pts[:, 1].max() + 5
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    box = (cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2)
    PW = 720
    mpp = PW / side

    def to_plan(xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        return np.c_[(xy[:, 0] - box[0]) * mpp, (box[3] - xy[:, 1]) * mpp].astype(int)

    S = np.array([[ppm, 0, -ext[0] * ppm], [0, -ppm, ext[3] * ppm], [0, 0, 1.0]])
    M = np.array([[mpp, 0, -box[0] * mpp], [0, -mpp, box[3] * mpp], [0, 0, 1.0]]) @ np.linalg.inv(S)
    base = cv2.warpPerspective(mosaic, M, (PW, PW), borderValue=(238, 238, 238))
    base = cv2.addWeighted(base, 0.55, np.full_like(base, 245), 0.45, 0)
    for c in (cam_a, cam_b):
        for kind, a in (("approximate", 0.10), ("reliable", 0.22)):
            for poly in meta["coverage"][c][kind]:
                p = to_plan(poly)
                ov = base.copy()
                cv2.fillPoly(ov, [p], CAM_BGR[c])
                base = cv2.addWeighted(ov, a, base, 1 - a, 0)
                cv2.polylines(base, [p], True, CAM_BGR[c], 1, cv2.LINE_AA)
    sb = to_plan(meta["stairs_base"])
    cv2.line(base, tuple(sb[0]), tuple(sb[1]), (30, 30, 30), 4)
    cv2.putText(base, "foot of staircase (stairs not mapped)", (int(sb[0][0]), int(sb[0][1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    for c in cal.cams:
        p = to_plan([cal.cal["cameras"][c]["position_m"]])[0]
        if 0 <= p[0] < PW and 0 <= p[1] < PW:
            h = np.radians(cal.cal["cameras"][c]["heading_deg"])
            q = p + np.array([np.cos(h), -np.sin(h)]) * 2.5 * mpp
            cv2.arrowedLine(base, tuple(p), tuple(q.astype(int)), CAM_BGR[c], 3, tipLength=0.25)
            cv2.circle(base, tuple(p), 9, CAM_BGR[c], -1)
            cv2.putText(base, c, (p[0] + 10, p[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.line(base, (20, PW - 20), (20 + int(5 * mpp), PW - 20), (0, 0, 0), 3)
    cv2.putText(base, "5 m", (20, PW - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cams = {}
    for c, tid in ((cam_a, tid_a), (cam_b, tid_b)):
        m = read_json(cam_dir(cfg, c) / "progress.json")
        t = pd.read_csv(cam_dir(cfg, c) / "tracks.csv")
        t = t[(t.t >= s0 - 1) & (t.t <= s1 + 1)]
        cams[c] = dict(meta=m, by=dict(list(t.groupby("frame"))), tid=tid, cap=cv2.VideoCapture(str(videos(cfg)[c])))
    fps, stride = cams[cam_a]["meta"]["fps"], cams[cam_a]["meta"]["stride"]
    f0 = int(np.ceil(s0 * fps / stride)) * stride
    for c in cams.values():
        c["cap"].set(cv2.CAP_PROP_POS_FRAMES, f0)
        c["pos"] = f0 - 1
    CW, CH = 640, 360
    bar = 40
    ow, oh = CW + PW, max(2 * CH, PW) + bar
    out_dir_ = ROOT / "evidence" / "bonus"
    out_dir_.mkdir(parents=True, exist_ok=True)
    out = out_dir_ / f"bonus_{line_id}_{cam_a}+{cam_b}_{s0}-{s1}s.mp4"
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}", "-r", str(fps / stride), "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    lt, lx, ly = line.t.values, line.X.values, line.Y.values
    f = f0
    while f / fps <= s1:
        t = f / fps
        panels = []
        for name, c in cams.items():
            while c["pos"] < f:
                if not c["cap"].grab():
                    break
                c["pos"] += 1
            ok, fr = c["cap"].retrieve()
            if not ok:
                break
            sc = CW / c["meta"]["width"]
            fr = cv2.resize(fr, (CW, CH))
            for r in c["by"].get(f, pd.DataFrame()).itertuples():
                b = tuple(int(v * sc) for v in (r.x1, r.y1, r.x2, r.y2))
                if r.tid == c["tid"]:
                    cv2.rectangle(fr, b[:2], b[2:], RED, 3)
                    cv2.putText(fr, f"P{line.gid.iloc[0]}", (b[0], max(14, b[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(fr, b[:2], b[2:], (150, 150, 150), 1)
            cv2.rectangle(fr, (0, CH - 28), (CW, CH), (0, 0, 0), -1)
            cv2.putText(fr, f"{name}  {int(t // 60):02d}:{t % 60:04.1f}", (8, CH - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            panels.append(fr)
        if len(panels) < 2:
            break
        plan = base.copy()
        seen = lt <= t
        trail = np.flatnonzero(seen & (lt >= s0 - 2))
        for i in range(1, len(trail)):
            if lt[trail[i]] - lt[trail[i - 1]] <= 1.5:
                cv2.line(plan, tuple(to_plan([lx[trail[i - 1]], ly[trail[i - 1]]])[0]), tuple(to_plan([lx[trail[i]], ly[trail[i]]])[0]), RED, 3, cv2.LINE_AA)
        i = int(np.abs(lt - t).argmin())
        if abs(lt[i] - t) <= 1.0:
            q = to_plan([lx[i], ly[i]])[0]
            cv2.circle(plan, tuple(q), 12, GOLD, -1)
            cv2.circle(plan, tuple(q), 12, (0, 0, 0), 2)
            cv2.putText(plan, f"P{line.gid.iloc[0]}", (q[0] + 14, q[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        canvas = np.zeros((oh, ow, 3), np.uint8)
        canvas[bar:bar + CH, :CW] = panels[0]
        canvas[bar + CH:bar + 2 * CH, :CW] = panels[1]
        canvas[bar:bar + PW, CW:] = plan
        cv2.putText(canvas, f"Same person, two cameras, one floor plan | global ID {line.gid.iloc[0]} (line {line_id}) | red box = this person | gold dot = position on the plan, "
                    f"drawn only while observed", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        p.stdin.write(canvas.tobytes())
        f += stride
    p.stdin.close()
    p.wait()
    print(out)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], int(a[3]), a[4], int(a[5]), int(a[6]))
