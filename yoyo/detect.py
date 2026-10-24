"""Stage 1: YOLO + ByteTrack per camera. Checkpointed + resumable. Output: cams/<cam>/tracks.csv"""
import os
import cv2
import pandas as pd
from .common import ROOT, cam_dir, read_json, write_json, work

COLS = ["frame", "t", "tid", "x1", "y1", "x2", "y2", "conf"]
FLUSH_EVERY = 300  # sampled frames between checkpoints
SEG_STRIDE = 100000  # track-id offset per resume segment so IDs never collide


def _in_models_dir(fn):
    mdir = ROOT / "models"
    mdir.mkdir(exist_ok=True)
    cwd = os.getcwd()
    os.chdir(mdir)
    try:
        return fn()
    finally:
        os.chdir(cwd)


def load_model(cfg):
    from ultralytics import YOLO
    d, name = cfg["detector"], cfg["detector"]["model"]
    if not d.get("openvino", True):
        return _in_models_dir(lambda: YOLO(name))
    ov = ROOT / "models" / f"{os.path.splitext(name)[0]}_{d['imgsz']}_openvino_model"
    if not ov.exists():
        def export():
            YOLO(name).export(format="openvino", imgsz=d["imgsz"])
            os.rename(os.path.splitext(name)[0] + "_openvino_model", ov.name)
        _in_models_dir(export)
    return YOLO(str(ov), task="detect")


def tracker_yaml(cfg):
    t = cfg["tracker"]
    p = work(cfg) / "bytetrack.yaml"
    p.write_text(
        "tracker_type: bytetrack\ntrack_high_thresh: 0.25\ntrack_low_thresh: 0.1\n"
        f"new_track_thresh: 0.25\ntrack_buffer: {t['track_buffer']}\nmatch_thresh: {t['match_thresh']}\nfuse_score: True\n")
    return str(p)


def process_camera(cfg, cam, video, limit_s=None, fresh=False, log=print):
    cd = cam_dir(cfg, cam)
    prog_p, tr_p = cd / "progress.json", cd / "tracks.csv"
    if fresh:
        for p in (prog_p, tr_p):
            p.unlink(missing_ok=True)
    prog = read_json(prog_p, {})
    if prog.get("done"):
        log(f"[detect] {cam}: already done")
        return
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(3)), int(cap.get(4))
    stride = max(1, round(fps / cfg["process_fps"]))
    start = prog.get("next_frame", 0)
    seg = prog.get("segment", -1) + 1 if start > 0 else 0
    if start > 0 and tr_p.exists():  # drop rows written after the last checkpoint
        df = pd.read_csv(tr_p)
        df[df.frame < start].to_csv(tr_p, index=False)
    elif start == 0:
        tr_p.unlink(missing_ok=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    end = total if not limit_s else min(total, start + int(limit_s * fps))
    model, trk, d = load_model(cfg), tracker_yaml(cfg), cfg["detector"]
    meta = dict(fps=fps, width=W, height=H, n_frames=total, stride=stride)
    rows, f, n_done = [], start, 0

    def flush(next_frame, done):
        nonlocal rows
        if rows:
            pd.DataFrame(rows, columns=COLS).to_csv(tr_p, mode="a", index=False, header=not tr_p.exists())
            rows = []
        write_json(prog_p, dict(next_frame=next_frame, segment=seg, done=done, **meta))

    try:
        from tqdm import tqdm
        bar = tqdm(total=(end - start) // stride, desc=f"detect {cam}")
    except ImportError:
        bar = None
    while f < end and cap.grab():
        if f % stride == 0:
            ok, frame = cap.retrieve()
            if ok:
                r = model.track(frame, persist=True, classes=[0], conf=d["conf"], imgsz=d["imgsz"],
                                tracker=trk, verbose=False)[0]
                b = r.boxes
                if b is not None and b.id is not None:
                    for xyxy, cf, i in zip(b.xyxy.tolist(), b.conf.tolist(), b.id.int().tolist()):
                        rows.append([f, round(f / fps, 3), i + seg * SEG_STRIDE,
                                     *[round(v, 1) for v in xyxy], round(cf, 3)])
                n_done += 1
                if bar:
                    bar.update(1)
                if n_done % FLUSH_EVERY == 0:
                    flush(f + 1, False)
        f += 1
    finished = f >= total
    flush(f, finished)
    log(f"[detect] {cam}: frame {f}/{total} {'(done)' if finished else '(resumable)'}")
