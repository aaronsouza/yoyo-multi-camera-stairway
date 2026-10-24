"""How many cross-camera matches are real? Matches the single-camera events of two cameras at their real times and with one camera
shifted by +-30..300 s (where no real overlap exists). Matches that survive shifting are chance matches: estimated precision =
(real - shifted) / real. Usage: python tools/dedup_precision.py results/final_before_checklist3/events.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yoyo.events import match_events


def run(path, pairs=(("cam5", "cam2"), ("cam5", "cam6"), ("cam2", "cam6")), windows=(0.5, 1.0, 1.5, 2.0, 3.0)):
    ev = pd.read_csv(path)
    ev = ev[ev.cams.str.split(";").apply(len) == 1]
    rows = []
    for prim, sec in pairs:
        a = ev[ev.cams == prim].reset_index(drop=True)
        b = ev[ev.cams == sec].reset_index(drop=True)
        for w in windows:
            real = len(match_events(a, b, w))
            shifts = [s for s in range(-300, 301, 15) if abs(s) >= 30]
            base = float(np.mean([len(match_events(a, b.assign(t_start=b.t_start + s), w)) for s in shifts]))
            rows.append(dict(primary=prim, secondary=sec, window_s=w, events_a=len(a), events_b=len(b), matched=real,
                             chance_matches=round(base, 1), est_true=round(real - base, 1),
                             est_precision=round(max(real - base, 0) / real, 2) if real else 0))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(run(sys.argv[1]).to_string(index=False))
