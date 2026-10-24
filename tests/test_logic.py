"""Synthetic checks of identity merging, crossing state machine, cross-camera dedup and group detection."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from yoyo.common import load_config
from yoyo.events import compute_events
from yoyo.groups import compute_groups
from yoyo.identities import agglomerate

CFG = load_config()
W = H = 1000
ZONES = {"c1": {"bottom": [[0.4, 0.8], [0.6, 0.8], [0.6, 1.0], [0.4, 1.0]],
                "top": [[0.4, 0.0], [0.6, 0.0], [0.6, 0.2], [0.4, 0.2]]}}
ZONES["c2"] = ZONES["c1"]
META = {c: dict(width=W, height=H) for c in ZONES}


def walk(tid, ys, t0=0.0, x=500, dt=0.1, hgt=100):
    ys = np.asarray(ys, float)
    n = len(ys)
    return pd.DataFrame(dict(frame=np.arange(n), t=t0 + np.arange(n) * dt, tid=tid, x1=x - 20, y1=ys - hgt,
                             x2=x + 20, y2=ys, conf=0.9))


def ramp(a, b, n=60):
    return np.linspace(a, b, n)


def run(tracks, tl, **ev_cfg):
    tl = pd.DataFrame(tl, columns=["cam", "tid", "gid"])
    cfg = {**CFG, "events": {**CFG["events"], **ev_cfg}}
    ev, naive = compute_events(tracks, tl.assign(n=50, t0=0, t1=9), ZONES, META, cfg)
    return ev, naive, tl


def test_up_and_down():
    up = walk(1, np.r_[ramp(900, 900, 10), ramp(900, 150), ramp(150, 150, 10)])
    down = walk(2, np.r_[ramp(150, 150, 10), ramp(150, 900), ramp(900, 900, 10)])
    ev, _, _ = run({"c1": pd.concat([up, down])}, [("c1", 1, 1), ("c1", 2, 2)])
    assert sorted(ev.direction) == ["DOWN", "UP"]


def test_turnaround_and_stop_not_counted():
    turn = walk(1, np.r_[ramp(900, 900, 10), ramp(900, 500), ramp(500, 900), ramp(900, 900, 10)])
    stop = walk(2, np.r_[ramp(900, 900, 10), ramp(900, 500), np.full(200, 500.0)])
    ev, _, _ = run({"c1": pd.concat([turn, stop])}, [("c1", 1, 1), ("c1", 2, 2)])
    assert len(ev) == 0


def test_cross_camera_dedup():
    # same person (gid 1) seen climbing in two cameras -> 1 event, naive count is 2
    ys = np.r_[ramp(900, 900, 10), ramp(900, 150), ramp(150, 150, 10)]
    ev, naive, _ = run({"c1": walk(1, ys), "c2": walk(7, ys, t0=0.3)}, [("c1", 1, 1), ("c2", 7, 1)])
    assert len(ev) == 1 and ev.direction[0] == "UP" and ev.cams[0] == "c1;c2"
    assert len(naive) == 2


def test_split_across_cameras_is_one_crossing():
    # camera 1 sees only the bottom, camera 2 only the top
    a = walk(1, np.r_[ramp(900, 900, 20)], t0=0)
    b = walk(5, np.r_[ramp(150, 150, 20)], t0=6)
    ev, _, _ = run({"c1": a, "c2": b}, [("c1", 1, 1), ("c2", 5, 1)])
    assert len(ev) == 0  # default: visits are chained per camera only (cross-camera stitching inflated counts)
    ev, _, _ = run({"c1": a, "c2": b}, [("c1", 1, 1), ("c2", 5, 1)], per_camera=False, verify_path=False)
    assert len(ev) == 1 and ev.direction[0] == "UP"


def test_group_vs_strangers():
    ys = np.r_[ramp(900, 900, 10), ramp(900, 150, 80), ramp(150, 150, 10)]
    tracks = pd.concat([walk(1, ys, x=480), walk(2, ys, x=530),            # together
                        walk(3, ys, x=100, t0=60), walk(4, ys[::-1], x=900, t0=60)])  # far apart
    ev, _, tl = run({"c1": tracks}, [("c1", i, i) for i in (1, 2, 3, 4)])
    gr = compute_groups({"c1": tracks}, tl.assign(t0=0.0, t1=9.0), ev, CFG)
    assert len(gr) == 1 and gr["size"][0] == 2 and gr.direction[0] == "UP"


def test_agglomerate_respects_conflicts():
    S = np.array([[1, .9, .9], [.9, 1, .5], [.9, .5, 1]])
    conflict = np.zeros((3, 3), bool)
    conflict[1, 2] = conflict[2, 1] = True  # 1 and 2 are simultaneous in one camera
    lab = agglomerate(S, conflict, 0.7)
    assert len(set(lab)) == 2
