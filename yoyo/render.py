"""Stage 6: annotated H.264 videos (boxes, local track id, global id, group, zones, running counters)."""
import subprocess
import cv2
import numpy as np
import pandas as pd
from .common import ROOT, cam_dir, final_dir, load_zones, read_json, work

PALETTE = [(66, 135, 245), (60, 180, 75), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
           (210, 245, 60), (0, 128, 128), (170, 110, 40), (128, 0, 0)]


def _dashed_rect(fr, p1, p2, col, dash=8):
    (x1, y1), (x2, y2) = p1, p2
    for a, b, c, d in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        n = max(int(np.hypot(c - a, d - b) // dash), 1)
        for k in range(0, n, 2):
            s0, s1 = k / n, min((k + 1) / n, 1.0)
            cv2.line(fr, (int(a + (c - a) * s0), int(b + (d - b) * s0)), (int(a + (c - a) * s1), int(b + (d - b) * s1)), col, 2)


def render_camera(cfg, cam, video, start_s=0, end_s=None, force=False, log=print, out_path=None, stats=None):
    import imageio_ffmpeg
    cd, fd = cam_dir(cfg, cam), final_dir(cfg)
    vd = work(cfg) / "videos"
    vd.mkdir(exist_ok=True)
    out = out_path or vd / (f"{cam}.mp4" if end_s is None and not start_s else f"{cam}_{int(start_s)}-{int(end_s or 0)}.mp4")
    if out.exists() and not force:
        log(f"[render] {cam}: exists")
        return out
    meta = read_json(cd / "progress.json")
    tracks = pd.read_csv(cd / "tracks.csv")
    tl = pd.read_csv(fd / "tracklets.csv")
    gid = {(r.cam, r.tid): r.gid for r in tl.itertuples()}
    gr = pd.read_csv(fd / "groups.csv") if (fd / "groups.csv").exists() else pd.DataFrame(columns=["group_id", "members"])
    ev_all = pd.read_csv(fd / "events.csv") if (fd / "events.csv").exists() else pd.DataFrame(columns=["event_id", "gid", "direction", "cams", "t_start", "t_end"])
    ev = ev_all[ev_all.cams.fillna("").str.split(";").apply(lambda x: cam in x)]
    member_windows = {}
    if "event_ids" in gr.columns:
        by_id = ev_all.set_index("event_id")
        for gp in gr.itertuples():
            for i in str(gp.event_ids).split(";"):
                if int(i) in by_id.index:
                    e = by_id.loc[int(i)]
                    if cam in str(e.cams).split(";"):
                        member_windows.setdefault(int(e.gid), []).append((e.t_start - 1.5, e.t_end + 2.5, gp.group_id, gp.size, gp.direction))
    sg_path = fd / "standing_groups.csv"
    sg = pd.read_csv(sg_path) if sg_path.exists() else pd.DataFrame(columns=["sg_id", "cam", "size", "tids", "t0", "t1"])
    standing = {}
    for s_ in sg[sg.cam == cam].itertuples():
        for tid_ in str(s_.tids).split(";"):
            standing[int(tid_)] = (s_.t0, s_.t1, s_.sg_id, s_.size)
    zones = load_zones().get(cam, {})
    fps, W, H, stride = meta["fps"], meta["width"], meta["height"], meta["stride"]
    sc = cfg["render"]["width"] / W
    ow, oh = int(W * sc) // 2 * 2, int(H * sc) // 2 * 2
    by_frame = {f: g for f, g in tracks.groupby("frame")}
    cap = cv2.VideoCapture(str(video))
    f0 = int(start_s * fps) // stride * stride
    f1 = meta["n_frames"] if end_s is None else min(meta["n_frames"], int(end_s * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{ow}x{oh}", "-r", str(fps / stride), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", str(cfg["render"]["crf"]), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    f = f0
    while f < f1 and cap.grab():
        if f % stride == 0:
            ok, fr = cap.retrieve()
            if ok:
                fr = cv2.resize(fr, (ow, oh))
                for name, poly in zones.items():
                    pts = (np.array(poly) * [ow, oh]).astype(np.int32)
                    cv2.polylines(fr, [pts], True, (0, 255, 255) if name == "top" else (255, 200, 0), 2)
                if stats is not None:
                    stats.append((f, len(by_frame.get(f, ()))))
                for r in by_frame.get(f, pd.DataFrame()).itertuples():
                    g = gid.get((cam, r.tid))
                    t = f / fps
                    hit = [w for w in member_windows.get(int(g), []) if w[0] <= t <= w[1]] if g == g and g is not None else []
                    x1, y1, x2, y2 = (int(v * sc) for v in (r.x1, r.y1, r.x2, r.y2))
                    label = f"T{r.tid % 100000}" + (f" P{int(g)}" if g == g and g is not None else "")
                    if hit:
                        _, _, gp_id, gp_size, gp_dir = hit[-1]
                        col = PALETTE[int(gp_id) % len(PALETTE)]
                        cv2.rectangle(fr, (x1, y1), (x2, y2), col, 3)
                        cv2.putText(fr, f"GROUP {int(gp_id)}  x{int(gp_size)}  {gp_dir}", (x1, max(26, y1 - 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
                    elif r.tid in standing and standing[r.tid][0] <= t <= standing[r.tid][1]:
                        col = (200, 200, 200)
                        _, _, sg_id, sg_size = standing[r.tid]
                        _dashed_rect(fr, (x1, y1), (x2, y2), col)
                        cv2.putText(fr, f"standing x{int(sg_size)}", (x1, max(26, y1 - 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
                    else:
                        col = (150, 150, 150)
                        cv2.rectangle(fr, (x1, y1), (x2, y2), col, 1)
                    cv2.putText(fr, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
                t = f / fps
                done = ev[ev.t_end <= t]
                txt = f"{cam}  video {int(t // 60):02d}:{t % 60:04.1f} (t={t:6.1f}s, frame {f})  UP {(done.direction == 'UP').sum()}  DOWN {(done.direction == 'DOWN').sum()}"
                cv2.rectangle(fr, (0, 0), (ow, 22), (0, 0, 0), -1)
                cv2.putText(fr, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                p.stdin.write(fr.tobytes())
        f += 1
    p.stdin.close()
    p.wait()
    log(f"[render] {cam}: {out.name} ({out.stat().st_size / 1e6:.0f} MB)")
    return out
