"""Click-to-annotate helper.
  python tools/annotate.py --cam cam1 --mode zones   # click polygon corners; N = next zone (bottom, then top); S = save
  python tools/annotate.py --cam cam1 --mode calib   # click ground points on the camera still
  python tools/annotate.py --mode floor --cam cam1   # click the SAME points, same order, on floorplan.png
Run `python -m yoyo.cli probe` first to create results/frame_<cam>.jpg.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load(p):
    return (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", required=True)
    ap.add_argument("--mode", choices=["zones", "calib", "floor"], required=True)
    a = ap.parse_args()
    img_path = ROOT / ("floorplan.png" if a.mode == "floor" else f"results/frame_{a.cam}.jpg")
    img = cv2.imread(str(img_path))
    if img is None:
        sys.exit(f"cannot read {img_path}")
    H, W = img.shape[:2]
    zones, cur, names = {}, [], ["bottom", "top"]
    pts = []

    def draw():
        v = img.copy()
        for n, poly in zones.items():
            cv2.polylines(v, [np.int32(poly)], True, (0, 255, 0), 2)
        for i, p in enumerate(cur if a.mode == "zones" else pts):
            cv2.circle(v, tuple(map(int, p)), 4, (0, 0, 255), -1)
            cv2.putText(v, str(i + 1), tuple(map(int, p)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("annotate", v)

    def click(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            (cur if a.mode == "zones" else pts).append([x, y])
            draw()

    cv2.namedWindow("annotate")
    cv2.setMouseCallback("annotate", click)
    draw()
    while True:
        k = cv2.waitKey(30) & 0xFF
        if k == ord("n") and a.mode == "zones" and len(cur) >= 3 and len(zones) < 2:
            zones[names[len(zones)]] = cur[:]
            cur.clear()
            print("saved zone", names[len(zones) - 1])
            draw()
        elif k == ord("u"):
            (cur if a.mode == "zones" else pts).pop() if (cur if a.mode == "zones" else pts) else None
            draw()
        elif k == ord("s") or k == 27:
            break
    if a.mode == "zones":
        if len(cur) >= 3 and len(zones) < 2:
            zones[names[len(zones)]] = cur
        p = ROOT / "zones.yaml"
        d = load(p)
        d[a.cam] = {n: [[round(x / W, 4), round(y / H, 4)] for x, y in poly] for n, poly in zones.items()}
        p.write_text(yaml.safe_dump(d))
    elif k == ord("s"):
        p = ROOT / "homographies.yaml"
        d = load(p)
        d.setdefault(a.cam, {})["image" if a.mode == "calib" else "floor"] = pts
        p.write_text(yaml.safe_dump(d))
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
