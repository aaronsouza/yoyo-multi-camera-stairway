"""Stages 4-5 driver + results tables. Cheap: reads saved tracks, never re-runs models."""
import pandas as pd
from .common import cam_dir, final_dir, load_zones, read_json, write_json
from .events import compute_events
from .groups import compute_groups


def load_tracks(cfg, cams):
    tracks, meta = {}, {}
    for c in cams:
        cd = cam_dir(cfg, c)
        if (cd / "tracks.csv").exists():
            tracks[c] = pd.read_csv(cd / "tracks.csv")
            meta[c] = read_json(cd / "progress.json")
    return tracks, meta


def sizes(gr, d):
    return {int(k): int(v) for k, v in gr[gr.direction == d]["size"].value_counts().sort_index().items()}


def analyze(cfg, cams, log=print):
    fd = final_dir(cfg)
    tl = pd.read_csv(fd / "tracklets.csv")
    tracks, meta = load_tracks(cfg, cams)
    zones = load_zones()
    ev, naive = compute_events(tracks, tl, zones, meta, cfg)
    gr = compute_groups(tracks, tl, ev, cfg)
    ev.to_csv(fd / "events.csv", index=False)
    gr.to_csv(fd / "groups.csv", index=False)
    people = tl.groupby("gid").agg(cams=("cam", lambda s: ";".join(sorted(set(s)))), t0=("t0", "min"),
                                   t1=("t1", "max"), n_tracklets=("tid", "size")).reset_index()
    people["crossings"] = people.gid.map(ev.gid.value_counts()).fillna(0).astype(int)
    people.to_csv(fd / "people.csv", index=False)
    cnt = lambda d, k: int((d.direction == k).sum())
    per_cam = {}
    for c in tracks:
        cev = ev[ev.cams.str.split(";").apply(lambda x: c in x)] if len(ev) else ev
        per_cam[c] = dict(unique_people=int(tl[tl.cam == c].gid.nunique()), up=cnt(cev, "UP"), down=cnt(cev, "DOWN"),
                          naive_up=cnt(naive[naive.cam == c], "UP"), naive_down=cnt(naive[naive.cam == c], "DOWN"))
    summary = dict(
        unique_people=int(tl.gid.nunique()), tracklets=int(len(tl)),
        up=cnt(ev, "UP"), down=cnt(ev, "DOWN"),
        naive_up_sum_over_cameras=cnt(naive, "UP"), naive_down_sum_over_cameras=cnt(naive, "DOWN"),
        groups_total=int(len(gr)), groups_up=cnt(gr, "UP"), groups_down=cnt(gr, "DOWN"),
        group_sizes_up=sizes(gr, "UP"), group_sizes_down=sizes(gr, "DOWN"),
        per_camera=per_cam)
    write_json(fd / "summary.json", summary)
    log(f"[analyze] people={summary['unique_people']} UP={summary['up']} DOWN={summary['down']} "
        f"groups UP={summary['groups_up']} DOWN={summary['groups_down']}")
    return summary
