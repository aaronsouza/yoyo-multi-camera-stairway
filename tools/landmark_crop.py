"""Zoomed crop of a frame with pixel rulers in ORIGINAL coordinates, for reading landmark positions by eye.
Usage: python tools/landmark_crop.py <image.png> x0 y0 x1 y1 out.png [zoom=2]"""
import sys

import cv2
import numpy as np


def main(src, x0, y0, x1, y1, out, zoom=2.0):
    img = cv2.imread(src)
    crop = cv2.resize(img[y0:y1, x0:x1], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    h, w = crop.shape[:2]
    pad = 36
    canvas = np.full((h + pad, w + pad, 3), 30, np.uint8)
    canvas[pad:, pad:] = crop
    step = 20 if (x1 - x0) <= 400 else 50
    for x in range((x0 // step + 1) * step, x1, step):
        px = int((x - x0) * zoom) + pad
        major = x % (step * 5) == 0 or step == 50
        cv2.line(canvas, (px, pad - (10 if major else 5)), (px, pad), (255, 255, 255), 1)
        cv2.line(canvas, (px, pad), (px, h + pad), (0, 255, 255), 1) if major else None
        if major:
            cv2.putText(canvas, str(x), (px - 14, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    for y in range((y0 // step + 1) * step, y1, step):
        py = int((y - y0) * zoom) + pad
        major = y % (step * 5) == 0 or step == 50
        cv2.line(canvas, (pad - (10 if major else 5), py), (pad, py), (255, 255, 255), 1)
        cv2.line(canvas, (pad, py), (w + pad, py), (0, 255, 255), 1) if major else None
        if major:
            cv2.putText(canvas, str(y), (0, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imwrite(out, canvas)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], int(a[2]), int(a[3]), int(a[4]), int(a[5]), a[6], float(a[7]) if len(a) > 7 else 2.0)
