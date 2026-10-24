"""One 30 s proof clip for requirements 1-4: two synchronized cameras side by side, drawn from saved results (no inference).
 1 tracking     : every person has a box with tracker id (T) and global person id (P)
 2 crossings    : completed crossings are green (UP) / red (DOWN) with event id, direction and the path walked (arrow)
 3 dedup        : a crossing seen by both cameras has ONE colour + MATCH on both views and is counted once
 4 groups       : a white envelope `GROUP id xN UP|DOWN` around the people crossing together; dashed `standing xN` is not counted
Usage: python tools/make_overview_evidence.py cam5 cam2 1515"""
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
GREEN, RED, GREY, WHITE = (60, 200, 60), (50, 50, 240), (150, 150, 150), (255, 255, 255)


def dashed(fr, p1, p2, col, dash=8):
    (x1, y1), (x2, y2) = p1, p2
    for a, b, c, d in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        n = max(int(np.hypot(c - a, d - b) // dash), 1)
        for k in range(0, n, 2):
            s0, s1 = k / n, min((k + 1) / n, 1.0)
            cv2.line(fr, (int(a + (c - a) * s0), int(b + (d - b) * s0)), (int(a + (c - a) * s1), int(b + (d - b) * s1)), col, 2)


def label(fr, txt, x, y, col, scale=0.45, thick=1):
    (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(fr, (x, y - h - 2), (x + w + 2, y + 3), (0, 0, 0), -1)
    cv2.putText(fr, txt, (x + 1, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


def main(prim, sec, s0):
    cfg = load_config()
    fd = final_dir(cfg)
    ev = pd.read_csv(fd / "events.csv")
    ev["dedup_with"] = ev.dedup_with.fillna("")
    gr = pd.read_csv(fd / "groups.csv")
    sg = pd.read_csv(fd / "standing_groups.csv")
    tl = pd.read_csv(fd / "tracklets.csv")
    s1 = s0 + DUR
    e_by = ev.set_index("event_id")
    group_of = {int(i): (g.group_id, g.size, g.direction) for g in gr.itertuples() for i in str(g.event_ids).split(";")}
    mcol = {e.event_id: PAL[k % len(PAL)] for k, e in enumerate(ev[ev.dedup_with.str.endswith("@" + sec)].itertuples())}

    shown = {prim: [], sec: []}
    for e in ev.itertuples():
        cs = e.cams.split(";")
        matched = e.event_id in mcol
        col = mcol[e.event_id] if matched else (GREEN if e.direction == "UP" else RED)
        if prim in cs:
            shown[prim].append((e.gid, e.event_id, e.direction, e.t_start, e.t_end, matched, col))
        if matched:
            shown[sec].append((int(e.dedup_with.split("@")[0]), e.event_id, e.direction, e.t_start, e.t_end, True, col))
        elif sec in cs and prim not in cs:
            shown[sec].append((e.gid, e.event_id, e.direction, e.t_start, e.t_end, False, col))

    cams = {}
    for c in (prim, sec):
        m = read_json(cam_dir(cfg, c) / "progress.json")
        t = pd.read_csv(cam_dir(cfg, c) / "tracks.csv")
        t = t[(t.t >= s0 - 3) & (t.t <= s1 + 3)].copy()
        t["gid"] = t.tid.map(dict(zip(tl[tl.cam == c].tid, tl[tl.cam == c].gid)))
        cams[c] = dict(meta=m, by=dict(list(t.groupby("frame"))), tr=t, cap=cv2.VideoCapture(str(videos(cfg)[c])),
                       stand={int(x): (s.t0, s.t1, s.size) for s in sg[sg.cam == c].itertuples() for x in str(s.tids).split(";")})
    fps, stride = cams[prim]["meta"]["fps"], cams[prim]["meta"]["stride"]
    pw = 800
    ph = int(cams[prim]["meta"]["height"] * pw / cams[prim]["meta"]["width"]) // 2 * 2
    bar_h = 44
    ow, oh = pw * 2, ph + bar_h
    f0 = int(np.ceil(s0 * fps / stride)) * stride
    for c in cams.values():
        c["cap"].set(cv2.CAP_PROP_POS_FRAMES, f0)
        c["pos"] = f0 - 1
    out_dir = ROOT / "evidence" / "overview_1-4"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{prim}+{sec}_requirements_1-4_{s0}-{s1}s.mp4"
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}",
           "-r", str(fps / stride), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "25", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    g_end = {g.group_id: e_by.loc[[int(i) for i in str(g.event_ids).split(";")]].t_end.max() for g in gr.itertuples()}
    f = f0
    while f / fps <= s1:
        t = f / fps
        panels, tracked = [], set()
        for name, c in cams.items():
            while c["pos"] < f:
                if not c["cap"].grab():
                    break
                c["pos"] += 1
            ok, fr = c["cap"].retrieve()
            if not ok:
                break
            sc = pw / c["meta"]["width"]
            fr = cv2.resize(fr, (pw, ph))
            rows = list(c["by"].get(f, pd.DataFrame()).itertuples())
            act = [x for x in shown[name] if x[3] - 1.5 <= t <= x[4] + 3.0]
            by_gid = {}
            for x in act:
                by_gid.setdefault(x[0], x)
            boxes = {}
            for r in rows:
                b = tuple(int(v * sc) for v in (r.x1, r.y1, r.x2, r.y2))
                x1, y1, x2, y2 = b
                gid_s = f"P{int(r.gid)}" if r.gid == r.gid else ""
                tracked.add((name, r.tid))
                x = by_gid.get(r.gid) if r.gid == r.gid else None
                if x:
                    _, eid, d, ts, te, matched, col = x
                    cv2.rectangle(fr, (x1, y1), (x2, y2), col, 3)
                    label(fr, f"T{r.tid % 100000} {gid_s} E{eid} {d}{' MATCH' if matched else ''}{' done' if t >= te else ''}", x1, max(14, y1 - 4), col)
                    boxes[r.gid] = b
                elif r.tid in c["stand"] and c["stand"][r.tid][0] <= t <= c["stand"][r.tid][1]:
                    dashed(fr, (x1, y1), (x2, y2), (210, 210, 210))
                    cv2.putText(fr, f"T{r.tid % 100000} {gid_s}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (210, 210, 210), 1, cv2.LINE_AA)
                else:
                    cv2.rectangle(fr, (x1, y1), (x2, y2), GREY, 1)
                    cv2.putText(fr, f"T{r.tid % 100000} {gid_s}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1, cv2.LINE_AA)
            for x in act:
                g = c["tr"][(c["tr"].gid == x[0]) & c["tr"].t.between(x[3], min(t, x[4]))].sort_values("t")
                if len(g) >= 2:
                    pts = np.c_[(g.x1 + g.x2) / 2 * sc, g.y2 * sc].astype(np.int32)
                    cv2.polylines(fr, [pts], False, x[6], 2, cv2.LINE_AA)
                    cv2.arrowedLine(fr, tuple(pts[-2]), tuple(pts[-1]), x[6], 3, tipLength=3.0)
            seen = {}
            for x in act:
                if x[1] in group_of and x[0] in boxes:
                    seen.setdefault(group_of[x[1]], []).append(boxes[x[0]])
            for (gid_, size, d), bs in seen.items():
                if len(bs) >= 1:
                    X1, Y1, X2, Y2 = min(b[0] for b in bs) - 6, min(b[1] for b in bs) - 6, max(b[2] for b in bs) + 6, max(b[3] for b in bs) + 6
                    cv2.rectangle(fr, (X1, Y1), (X2, Y2), WHITE, 2)
                    label(fr, f"GROUP {int(gid_)} x{int(size)} {d}", X1, max(30, Y1 - 18), WHITE, 0.55, 2)
            label(fr, f"{name}  {int(t // 60):02d}:{t % 60:04.1f}", 8, ph - 10, (0, 255, 255), 0.8, 2)
            panels.append(fr)
        if len(panels) < 2:
            break
        done = ev[(ev.t_end <= t) & (ev.t_end >= s0) & (ev.cams.str.contains(prim) | ev.cams.str.contains(sec))]
        nm = int(ev[ev.dedup_with.str.endswith("@" + sec) & (ev.t_end <= t) & (ev.t_end >= s0)].shape[0])
        gd = gr[[g_end[g.group_id] <= t and g_end[g.group_id] >= s0 for g in gr.itertuples()]]
        gd = gd[gd.cams.str.contains(prim) | gd.cams.str.contains(sec)]
        bar = np.zeros((bar_h, ow, 3), np.uint8)
        cv2.putText(bar, f"1 TRACKING: {len(tracked)} people tracked now (T=tracker id, P=person id)   |   2 CROSSINGS completed: "
                    f"UP {(done.direction == 'UP').sum()}  DOWN {(done.direction == 'DOWN').sum()}  (green=UP red=DOWN, arrow = path walked)",
                    (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.46, WHITE, 1, cv2.LINE_AA)
        cv2.putText(bar, f"3 CROSS-CAMERA: {nm} crossing(s) seen by both cameras counted once (same colour, MATCH)   |   4 GROUPS completed: "
                    f"UP {(gd.direction == 'UP').sum()}  DOWN {(gd.direction == 'DOWN').sum()}  (white box GROUP id xsize; dashed = standing, not counted)",
                    (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.46, WHITE, 1, cv2.LINE_AA)
        p.stdin.write(np.vstack([bar, np.hstack(panels)]).tobytes())
        f += stride
    p.stdin.close()
    p.wait()
    inw = ev[ev.t_end.between(s0, s1) & (ev.cams.str.contains(prim) | ev.cams.str.contains(sec))]
    inw[["event_id", "gid", "direction", "t_start", "t_end", "cams", "dedup_with", "entity"]].to_csv(out_dir / f"{prim}+{sec}_{s0}-{s1}s_events.csv", index=False)
    gi = gr[[(g_end[g.group_id] >= s0 and g_end[g.group_id] <= s1) for g in gr.itertuples()]]
    gi[["group_id", "direction", "size", "members", "t", "cams"]].to_csv(out_dir / f"{prim}+{sec}_{s0}-{s1}s_groups.csv", index=False)
    print(out, "| events", len(inw), inw.direction.value_counts().to_dict(), "| merged duplicates", int(inw.dedup_with.str.len().gt(0).sum()),
          "| groups", len(gi), gi.direction.value_counts().to_dict())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
