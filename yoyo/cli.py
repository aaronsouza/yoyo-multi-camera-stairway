"""python -m yoyo.cli <stage> [--cams cam1 cam2] [--limit-s N] [--force]

Stages (each checkpoints to results/ and is skipped if already done):
  probe | detect | embed | identities | analyze | render | validate | calibrate | stitch | floorplan | calcheck | all
detect/embed are the only expensive stages. identities/analyze/render re-run in seconds-minutes from saved data.
"""
import argparse
import cv2
from . import common
from .common import load_config, videos, work, cam_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--cams", nargs="*")
    ap.add_argument("--limit-s", type=float, help="detect: only process this many seconds (benchmark / resume later)")
    ap.add_argument("--fresh", action="store_true", help="detect: discard checkpoint")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--start-s", type=float, default=0)
    ap.add_argument("--end-s", type=float)
    ap.add_argument("--gt")
    ap.add_argument("--tol", type=float, default=8.0)
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    vids = videos(cfg)
    cams = a.cams or list(vids)
    if a.stage in ("probe", "detect", "embed", "render", "all") and not vids:
        raise SystemExit(f"No videos found in {cfg['video_dir']}")
    st = a.stage

    if st == "probe":
        for c in cams:
            cap = cv2.VideoCapture(str(vids[c]))
            fps, n = cap.get(5), int(cap.get(7))
            cap.set(1, int(n * 0.1))
            ok, fr = cap.read()
            if ok:
                cv2.imwrite(str(work(cfg) / f"frame_{c}.jpg"), fr)
            print(f"{c}: {int(cap.get(3))}x{int(cap.get(4))} {fps:.2f}fps {n / fps / 60:.1f} min")
    if st in ("detect", "all"):
        from . import detect
        for c in cams:
            detect.process_camera(cfg, c, vids[c], a.limit_s, a.fresh)
    if st in ("embed", "all"):
        from . import embed
        e = embed.Embedder(cfg)
        for c in cams:
            embed.process_camera(cfg, c, vids[c], e, a.force)
    if st in ("identities", "analyze", "all"):
        from . import identities, report
        if st != "analyze" or a.force:
            identities.run(cfg, cams)
        report.analyze(cfg, cams)
    if st in ("render", "all"):
        from . import render
        for c in cams:
            render.render_camera(cfg, c, vids[c], a.start_s, a.end_s, a.force)
    if st == "validate":
        from . import validate
        validate.run(cfg, a.gt, a.start_s, a.end_s, a.tol)
    if st == "calibrate":
        from . import calibrate
        calibrate.run(cfg)
    if st == "stitch":
        from . import stitch
        stitch.run(cfg)
    if st == "floorplan":
        from . import floorplan
        floorplan.build(cfg)
    if st == "calcheck":
        from . import calibration_check
        calibration_check.run(cfg)


if __name__ == "__main__":
    main()
