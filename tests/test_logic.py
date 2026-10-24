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
    ys = np.r_[ramp(900, 900, 10), ramp(900, 150), ramp(150, 150, 10)]
    ev, naive, _ = run({"c1": walk(1, ys), "c2": walk(7, ys, t0=0.3)}, [("c1", 1, 1), ("c2", 7, 1)])
    assert len(ev) == 1 and ev.direction[0] == "UP" and ev.cams[0] == "c1;c2"
    assert len(naive) == 2


def test_split_across_cameras_is_one_crossing():
    a = walk(1, np.r_[ramp(900, 900, 20)], t0=0)
    b = walk(5, np.r_[ramp(150, 150, 20)], t0=6)
    ev, _, _ = run({"c1": a, "c2": b}, [("c1", 1, 1), ("c2", 5, 1)])
    assert len(ev) == 0
    ev, _, _ = run({"c1": a, "c2": b}, [("c1", 1, 1), ("c2", 5, 1)], per_camera=False, verify_path=False)
    assert len(ev) == 1 and ev.direction[0] == "UP"


def test_group_vs_strangers():
    ys = np.r_[ramp(900, 900, 10), ramp(900, 150, 80), ramp(150, 150, 10)]
    tracks = pd.concat([walk(1, ys, x=480), walk(2, ys, x=530),
                        walk(3, ys, x=100, t0=60), walk(4, ys[::-1], x=900, t0=60)])
    ev, _, tl = run({"c1": tracks}, [("c1", i, i) for i in (1, 2, 3, 4)])
    gr = compute_groups({"c1": tracks}, tl.assign(t0=0.0, t1=9.0), ev, CFG)
    assert len(gr) == 1 and gr["size"][0] == 2 and gr.direction[0] == "UP"


def test_agglomerate_respects_conflicts():
    S = np.array([[1, .9, .9], [.9, 1, .5], [.9, .5, 1]])
    conflict = np.zeros((3, 3), bool)
    conflict[1, 2] = conflict[2, 1] = True
    lab = agglomerate(S, conflict, 0.7)
    assert len(set(lab)) == 2


EDGE = dict(local_events=True, edge=dict(enabled=True, margin=0.04, min_rise=0.10, ref_s=3.0, min_dur_s=1.0, max_dur_s=12.0,
                                         max_backtrack=0.04, confirm_window_s=6.0))


def _edge_walker(tid, x_end, y_end, hold=15, n=40):
    """Stands in the top zone, then walks diagonally to (x_end, y_end), which is the frame edge, short of the bottom zone."""
    ys = np.r_[np.full(hold, 150.0), ramp(150, y_end, n)]
    xs = np.r_[np.full(hold, 500.0), ramp(500, x_end, n)]
    d = walk(tid, ys)
    d["x1"], d["x2"] = xs - 20, xs + 20
    return d


def test_edge_exit_down_is_inferred_but_flat_exit_is_not():
    ev, _, _ = run({"c1": _edge_walker(1, 980, 560)}, [("c1", 1, 1)], **EDGE)
    assert list(ev.direction) == ["DOWN"] and bool(ev.inferred[0])
    ev, _, _ = run({"c1": _edge_walker(1, 980, 175)}, [("c1", 1, 1)], **EDGE)
    assert len(ev) == 0
    ev, _, _ = run({"c1": _edge_walker(1, 700, 560)}, [("c1", 1, 1)], **EDGE)
    assert len(ev) == 0


def test_edge_rule_off_by_default():
    ev, _, _ = run({"c1": _edge_walker(1, 980, 560)}, [("c1", 1, 1)])
    assert len(ev) == 0


def test_track_without_identity_gives_local_event():
    ys = np.r_[ramp(900, 900, 10), ramp(900, 150), ramp(150, 150, 10)]
    ev, _, _ = run({"c1": walk(5, ys)}, [], local_events=True)
    assert len(ev) == 1 and ev.gid[0] < 0 and ev.direction[0] == "UP"
    ev, _, _ = run({"c1": walk(5, ys)}, [], local_events=False)
    assert len(ev) == 0


def test_teleport_between_zones_is_not_a_crossing():
    ys = np.r_[np.full(15, 900.0), np.full(15, 150.0)]
    ev, _, _ = run({"c1": walk(1, ys)}, [("c1", 1, 1)])
    assert len(ev) == 0


def test_two_crossings_of_one_camera_are_not_merged():
    from yoyo.events import merge_across_cameras
    ev = pd.DataFrame([dict(gid=1, direction="DOWN", t_start=0.0, t_end=1.5, cams="c1", inferred=False),
                       dict(gid=1, direction="DOWN", t_start=5.0, t_end=7.0, cams="c1", inferred=False),
                       dict(gid=1, direction="DOWN", t_start=5.5, t_end=7.5, cams="c2", inferred=False)])
    out = merge_across_cameras(ev, 6.0)
    assert len(out) == 2 and sorted(out.cams) == ["c1", "c1;c2"]


def test_quick_direction_flip_keeps_one_event():
    ys = np.r_[ramp(900, 900, 10), ramp(900, 150, 40), ramp(150, 150, 5), ramp(150, 900, 40), ramp(900, 900, 10)]
    ev, _, _ = run({"c1": walk(1, ys)}, [("c1", 1, 1)], flip_window_s=8)
    assert len(ev) == 1
    ev, _, _ = run({"c1": walk(1, ys)}, [("c1", 1, 1)])
    assert len(ev) == 2


def _ev(rows):
    return pd.DataFrame([dict(gid=g, direction=d, t_start=ts, t_end=ts + 2, cams=c, inferred=False, confirmed_by="")
                         for g, d, ts, c in rows])


def test_cross_camera_event_dedup_is_one_to_one_and_directional():
    from yoyo.events import dedup_cross_camera
    cfg = dict(window_s=1.0, pairs=[["c5", "c2"]])
    ev = _ev([(1, "DOWN", 10.0, "c5"), (2, "DOWN", 10.4, "c5"), (3, "DOWN", 11.0, "c5"),
              (11, "DOWN", 10.1, "c2"), (12, "DOWN", 10.9, "c2"), (13, "UP", 10.2, "c2"), (14, "DOWN", 40.0, "c2")])
    out = dedup_cross_camera(ev, cfg)
    assert len(out) == 5
    assert sorted(out[out.cams == "c2;c5"].gid) == [1, 3]
    assert sorted(out[out.cams == "c2"].gid) == [13, 14]


def test_cross_camera_dedup_ignores_unlisted_camera_pairs():
    from yoyo.events import dedup_cross_camera
    ev = _ev([(1, "UP", 5.0, "c5"), (2, "UP", 5.2, "c6")])
    assert len(dedup_cross_camera(ev, dict(window_s=1.0, pairs=[["c5", "c2"]]))) == 2
    assert len(dedup_cross_camera(ev, dict(window_s=1.0, pairs=[["c5", "c6"]]))) == 1


def _walkers(*specs, n=80, c="c1"):
    """specs: (tid, gid, x). Everyone walks UP together over 8 s; returns tracks and the tracklet table."""
    ys = np.r_[ramp(900, 150, n)]
    tracks = pd.concat([walk(tid, ys, x=x) for tid, _, x in specs])
    tl = pd.DataFrame([dict(cam=c, tid=tid, gid=gid, n=n, t0=0.0, t1=8.0) for tid, gid, _ in specs])
    return {c: tracks}, tl


def _events(*rows):
    return pd.DataFrame([dict(event_id=i + 1, gid=g, direction="UP", t_start=t - 2, t_end=t, cams=c, inferred=False)
                         for i, (g, t, c) in enumerate(rows)],
                        columns=["event_id", "gid", "direction", "t_start", "t_end", "cams", "inferred"])


def _gcfg(**kw):
    from yoyo.groups import compute_all_groups
    return compute_all_groups, {**CFG, "groups": {**CFG["groups"], "camera_priority": ["c5", "c2"], **kw}}


def test_group_never_holds_one_person_twice():
    f, cfg = _gcfg()
    tracks, tl = _walkers((1, 1, 480), (2, 2, 530), (3, 1, 580))
    gr, _ = f(tracks, tl, _events((1, 8.0, "c1"), (2, 8.1, "c1"), (1, 8.2, "c1")), cfg)
    assert list(gr["size"]) == [2]


def test_group_lives_in_one_camera_view():
    f, cfg = _gcfg()
    tracks, tl = _walkers((1, 1, 480), (2, 2, 530), c="c5")
    gr, _ = f(tracks, tl, _events((1, 8.0, "c5"), (2, 8.1, "c2")), cfg)
    assert len(gr) == 0
    gr, _ = f(tracks, tl, _events((1, 8.0, "c5"), (2, 8.1, "c2;c5")), cfg)
    assert list(gr["size"]) == [2]


def test_long_chain_is_not_a_group():
    f, cfg = _gcfg(min_link_density=0.7)
    tracks, tl = _walkers((1, 1, 400), (2, 2, 500), (3, 3, 600), (4, 4, 700))
    gr, _ = f(tracks, tl, _events((1, 8.0, "c1"), (2, 8.1, "c1"), (3, 8.2, "c1"), (4, 8.3, "c1")), cfg)
    assert 4 not in list(gr["size"])


def test_standing_pair_is_reported_but_never_counted():
    f, cfg = _gcfg()
    still = np.full(120, 500.0)
    tracks = {"c1": pd.concat([walk(1, still, x=480), walk(2, still, x=540), walk(3, still, x=900)])}
    tl = pd.DataFrame([dict(cam="c1", tid=t, gid=t, n=120, t0=0.0, t1=12.0) for t in (1, 2, 3)])
    gr, sg = f(tracks, tl, _events(), cfg)
    assert len(gr) == 0 and list(sg["size"]) == [2] and sg.gids[0] == "1;2"
