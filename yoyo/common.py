import json, os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".m4v"}


def load_config(path=None):
    return yaml.safe_load(Path(path or ROOT / "config.yaml").read_text())


def work(cfg):
    d = (ROOT / cfg["work_dir"]).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def cam_dir(cfg, cam):
    d = work(cfg) / "cams" / cam
    d.mkdir(parents=True, exist_ok=True)
    return d


def final_dir(cfg):
    d = work(cfg) / "final"
    d.mkdir(parents=True, exist_ok=True)
    return d


def videos(cfg):
    vd = (ROOT / cfg["video_dir"])
    if not vd.exists():
        return {}
    return {p.stem: p for p in sorted(vd.rglob("*"))
            if p.suffix.lower() in VIDEO_EXT and "__MACOSX" not in p.parts and not p.name.startswith("._")}


def load_zones(path=None):
    return yaml.safe_load(Path(path or ROOT / "zones.yaml").read_text()) or {}


def read_json(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def write_json(p, obj):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, p)


def pick_cams(cfg, cams):
    vids = videos(cfg)
    return [c for c in (cams or list(vids))]
