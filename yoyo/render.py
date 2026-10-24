"""Stage 6: annotated H.264 videos (boxes, local track id, global id, group, zones, running counters)."""
import subprocess
import cv2
import numpy as np
import pandas as pd
from .common import ROOT, cam_dir, final_dir, load_zones, read_json, work

PALETTE = [(66, 135, 245), (60, 180, 75), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
           (210, 245, 60), (0, 128, 128), (170, 110, 40), (128, 0, 0)]


def render_camera(cfg, cam, video, start_s=0, end_s=None, force=False, log=print):
    import imageio_ffmpeg
    cd, fd = cam_dir(cfg, cam), final_dir(cfg)
    vd = work(cfg) / "videos"
    vd.mkdir(exist_ok=True)
    out = vd / (f"{cam}.mp4" if end_s is None and not start_s else f"{cam}_{int(start_s)}-{int(end_s or 0)}.mp4")
    if out.exists() and not force:
        log(f"[render] {cam}: exists")
        return out
    meta = read_json(cd / "progress.json")
    tracks = pd.read_csv(cd / "tracks.csv")
    tl = pd.read_csv(fd / "tracklets.csv")
    gid = {(r.cam, r.tid): r.gid for r in tl.itertuples()}
    gr = pd.read_csv(fd / "groups.csv") if (fd / "groups.csv").exists() else pd.DataFrame(columns=["group_id", "members"])
    gid_group = {int(m): r.group_id for r in gr.itertuples() for m in str(r.members).split(";") if m}
    ev = pd.read_csv(fd / "events.csv") if (fd / "events.csv").exists() else pd.DataFrame(columns=["direction", "cams", "t_end"])
    ev = ev[ev.cams.fillna("").str.split(";").apply(lambda x: cam in x)]
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
                for r in by_frame.get(f, pd.DataFrame()).itertuples():
                    g = gid.get((cam, r.tid))
                    grp = gid_group.get(g)
                    col = PALETTE[grp % len(PALETTE)] if grp else (180, 180, 180)
                    x1, y1, x2, y2 = (int(v * sc) for v in (r.x1, r.y1, r.x2, r.y2))
                    cv2.rectangle(fr, (x1, y1), (x2, y2), col, 2)
                    label = f"T{r.tid % 100000}" + (f" P{g}" if g else "") + (f" G{grp}" if grp else "")
                    cv2.putText(fr, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
                t = f / fps
                done = ev[ev.t_end <= t]
                txt = f"{cam}  t={t:6.1f}s  UP {(done.direction == 'UP').sum()}  DOWN {(done.direction == 'DOWN').sum()}"
                cv2.rectangle(fr, (0, 0), (ow, 22), (0, 0, 0), -1)
                cv2.putText(fr, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                p.stdin.write(fr.tobytes())
        f += 1
    p.stdin.close()
    p.wait()
    log(f"[render] {cam}: {out.name} ({out.stat().st_size / 1e6:.0f} MB)")
    return out
