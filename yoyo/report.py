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


def entities(ev, gr):
    """Counting units: a group that crossed together is ONE entity, every other crossing is its own (solo) entity.
    Returns event_id -> entity key. Groups list their crossing events (`event_ids`); older group tables without that
    column fall back to identity membership, picking the group closest in time for identities that sit in several."""
    out = {r.event_id: f"S{r.event_id}" for r in ev.itertuples()}
    if "event_ids" in gr.columns:
        for g in gr.itertuples():
            for i in str(g.event_ids).split(";"):
                out[int(i)] = f"G{g.group_id}"
        return out
    member = {}
    for g in gr[gr.direction != "NONE"].itertuples():
        for m in str(g.members).split(";"):
            member.setdefault((int(m), g.direction), []).append((g.group_id, g.t))
    for r in ev.itertuples():
        cand = member.get((r.gid, r.direction))
        if cand:
            out[r.event_id] = f"G{min(cand, key=lambda c: abs(c[1] - r.t_start))[0]}"
    return out


def analyze(cfg, cams, log=print):
    fd = final_dir(cfg)
    tl = pd.read_csv(fd / "tracklets.csv")
    tracks, meta = load_tracks(cfg, cams)
    zones = load_zones()
    ev, naive = compute_events(tracks, tl, zones, meta, cfg)
    gr = compute_groups(tracks, tl, ev, cfg)
    ev["entity"] = ev.event_id.map(entities(ev, gr))
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
        entities_up=int(ev[ev.direction == "UP"].entity.nunique()), entities_down=int(ev[ev.direction == "DOWN"].entity.nunique()),
        solo_up=int(ev[(ev.direction == "UP") & ev.entity.str.startswith("S")].shape[0]),
        solo_down=int(ev[(ev.direction == "DOWN") & ev.entity.str.startswith("S")].shape[0]),
        groups_total=int(len(gr)), groups_up=cnt(gr, "UP"), groups_down=cnt(gr, "DOWN"),
        group_sizes_up=sizes(gr, "UP"), group_sizes_down=sizes(gr, "DOWN"),
        per_camera=per_cam)
    write_json(fd / "summary.json", summary)
    log(f"[analyze] people={summary['unique_people']} UP={summary['up']} DOWN={summary['down']} "
        f"groups UP={summary['groups_up']} DOWN={summary['groups_down']}")
    return summary
