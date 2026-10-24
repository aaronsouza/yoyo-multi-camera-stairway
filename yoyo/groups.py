"""Stage 5: groups = people who stay close, move in the same direction and keep doing so over time.

Crossing groups (counted UP/DOWN) are built from crossing events; standing groups (people close together for seconds without going
anywhere) are reported separately and never counted as UP/DOWN."""
import numpy as np
import pandas as pd


def _pairs_in_cam(df, gid_map, g, with_time=False, standing=False):
    """Tracklet pairs of one camera that walk together (and, with standing=True, pairs that stay together without moving)."""
    d = df[df.tid.isin(gid_map.index)].copy()
    d["fx"], d["fy"], d["h"] = (d.x1 + d.x2) / 2, d.y2, d.y2 - d.y1
    m = d.merge(d, on="frame", suffixes=("", "_b"))
    m = m[m.tid < m.tid_b]
    m["dist"] = np.hypot(m.fx - m.fx_b, m.fy - m.fy_b) / ((m.h + m.h_b) / 2)
    found, stand = [], []
    min_frames = min(g["min_common_frames"], g.get("standing_min_frames", 10 ** 9)) if standing else g["min_common_frames"]
    for (a, b), x in m.groupby(["tid", "tid_b"]):
        if len(x) < min_frames:
            continue
        x = x.sort_values("frame")
        va = np.array([x.fx.iloc[-1] - x.fx.iloc[0], x.fy.iloc[-1] - x.fy.iloc[0]])
        vb = np.array([x.fx_b.iloc[-1] - x.fx_b.iloc[0], x.fy_b.iloc[-1] - x.fy_b.iloc[0]])
        hm = x.h.mean()
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        walking = (len(x) >= g["min_common_frames"] and (x.dist <= g["prox_heights"]).mean() >= g["frac_close"]
                   and na / hm >= g["min_move_heights"] and nb / hm >= g["min_move_heights"]
                   and va @ vb / (na * nb) >= g["min_cos"])
        if walking:
            found.append((a, b, float(x.t.median())) if with_time else (a, b))
        elif (standing and len(x) >= g.get("standing_min_frames", 10 ** 9)
              and (x.dist <= g.get("standing_prox_heights", 1.2)).mean() >= g.get("standing_frac_close", 0.8)
              and na / hm < g["min_move_heights"] and nb / hm < g["min_move_heights"]):
            stand.append((a, b, float(x.t.min()), float(x.t.max()), float(x.fx.mean()), float(x.fy.mean())))
    return (found, stand) if standing else found


def _home_cam(cams, priority):
    """The camera whose view a crossing event is grouped in: the highest-priority camera that saw it."""
    cs = cams.split(";")
    return min(cs, key=lambda c: priority.index(c) if c in priority else len(priority))


def compute_crossing_groups(tracks, tracklets, events, cfg):
    """Groups of crossings (see compute_all_groups); returns only the crossing groups."""
    return compute_all_groups(tracks, tracklets, events, cfg)[0]


def compute_all_groups(tracks, tracklets, events, cfg):
    """Returns (crossing groups, standing groups).

    Crossing groups: two same-direction crossing events belong to one group when the two people walked together (a close, co-moving
    tracklet pair in the same camera) AND crossed within a few seconds of each other. The union is time-local, so an identity that
    was re-used for a different person seconds later (merged ID) cannot chain separate crossings into one group. A group lives in
    one camera's view (the highest-priority camera that saw each event), holds each person once, is not a long chain, and is a set
    of events (`event_ids`); size = people who crossed together."""
    g, off = cfg["groups"], cfg.get("offsets_s", {})
    pair_t = {}
    stand_rows = []
    for cam, df in tracks.items():
        sub = tracklets[tracklets.cam == cam].set_index("tid")
        walk, stand = _pairs_in_cam(df, sub.gid, g, with_time=True, standing=True)
        for a, b, tm in walk:
            ga, gb = int(sub.gid[a]), int(sub.gid[b])
            if ga != gb:
                pair_t.setdefault((min(ga, gb), max(ga, gb)), []).append(tm + off.get(cam, 0))
        stand_rows += [dict(cam=cam, a=a, b=b, gid_a=int(sub.gid[a]), gid_b=int(sub.gid[b]), t0=t0 + off.get(cam, 0),
                            t1=t1 + off.get(cam, 0), x=x, y=y) for a, b, t0, t1, x, y in stand]
    ev = events.reset_index(drop=True)
    home = [_home_cam(c, g.get("camera_priority", [])) for c in ev.cams]
    edges = []
    for i in range(len(ev)):
        for j in range(i + 1, len(ev)):
            a, b = ev.iloc[i], ev.iloc[j]
            if a.direction != b.direction or abs(a.t_end - b.t_end) > g.get("pair_gap_s", 4.0) or home[i] != home[j] or a.gid == b.gid:
                continue
            tm = pair_t.get((min(a.gid, b.gid), max(a.gid, b.gid)), [])
            mid = (a.t_end + b.t_end) / 2
            if any(abs(t - mid) <= g.get("pair_window_s", 6.0) for t in tm):
                edges.append((abs(a.t_end - b.t_end), i, j))
    members = {i: [i] for i in range(len(ev))}
    owner = list(range(len(ev)))
    linked = set()
    for _, i, j in sorted(edges):
        ri, rj = owner[i], owner[j]
        if ri == rj:
            linked.add((i, j))
            continue
        both = members[ri] + members[rj]
        te = ev.t_end.values[both]
        n_links = sum(1 for (x, y) in linked | {(i, j)} if x in both and y in both)
        if (len(both) > g.get("max_size", 8) or te.max() - te.min() > g.get("max_span_s", 8.0)
                or len(set(ev.gid.values[both])) < len(both)
                or (len(both) >= 4 and n_links < g.get("min_link_density", 0.5) * len(both) * (len(both) - 1) / 2)):
            continue
        linked.add((i, j))
        for k in members[rj]:
            owner[k] = ri
        members[ri] = both
        del members[rj]
    rows = []
    for idx in sorted((m for m in members.values() if len(m) >= 2), key=lambda m: ev.t_start.values[m].min()):
        sel = ev.iloc[idx]
        rows.append(dict(size=len(idx), members=";".join(map(str, sorted(set(sel.gid)))), direction=sel.direction.iloc[0],
                         n_crossing_members=len(idx), t=round(float(sel.t_start.median()), 2),
                         cams=";".join(sorted(set(";".join(sel.cams).split(";")))), event_ids=";".join(map(str, sorted(sel.event_id)))))
    out = pd.DataFrame(rows, columns=["size", "members", "direction", "n_crossing_members", "t", "cams", "event_ids"])
    out.insert(0, "group_id", np.arange(1, len(out) + 1))
    return out, standing_groups(stand_rows, g)


def standing_groups(rows, g):
    """Components of standing pairs (same camera), capped in size, one row per group; every person appears once."""
    cols = ["cam", "size", "tids", "gids", "t0", "t1", "x", "y"]
    if not rows:
        return pd.DataFrame(columns=["sg_id"] + cols)
    df = pd.DataFrame(rows)
    out = []
    for cam, d in df.groupby("cam"):
        par = {}

        def f(x):
            par.setdefault(x, x)
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for r in d.itertuples():
            par[f(r.a)] = f(r.b)
        comp = {}
        for r in d.itertuples():
            comp.setdefault(f(r.a), []).append(r)
        for rs in comp.values():
            tids = sorted({r.a for r in rs} | {r.b for r in rs})
            gids = sorted({r.gid_a for r in rs} | {r.gid_b for r in rs})
            if (2 <= len(tids) <= g.get("standing_max_size", 6) and len(gids) == len(tids)
                    and len(rs) >= g.get("min_link_density", 0.5) * len(tids) * (len(tids) - 1) / 2):
                out.append(dict(cam=cam, size=len(tids), tids=";".join(map(str, tids)), gids=";".join(map(str, gids)),
                                t0=round(min(r.t0 for r in rs), 2), t1=round(max(r.t1 for r in rs), 2),
                                x=round(float(np.mean([r.x for r in rs])), 1), y=round(float(np.mean([r.y for r in rs])), 1)))
    res = pd.DataFrame(out, columns=cols).sort_values(["cam", "t0"]).reset_index(drop=True)
    res.insert(0, "sg_id", np.arange(1, len(res) + 1))
    return res


def compute_groups(tracks, tracklets, events, cfg):
    """Legacy identity-chained groups (groups.mode != 'crossing'); in 'crossing' mode see compute_all_groups."""
    g = cfg["groups"]
    if g.get("mode") == "crossing":
        return compute_crossing_groups(tracks, tracklets, events, cfg)
    tl = tracklets.assign(tc=(tracklets.t0 + tracklets.t1) / 2)
    recs = []
    for cam, df in tracks.items():
        sub = tl[tl.cam == cam].set_index("tid")
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in _pairs_in_cam(df, sub.gid, g):
            parent[find(a)] = find(b)
        comps = {}
        for x in list(parent):
            comps.setdefault(find(x), []).append(x)
        for tids in comps.values():
            if 2 <= len(tids) <= g.get("max_size", 8):
                recs.append(dict(cam=cam, n=len(tids), gids=set(sub.loc[tids].gid), t=float(sub.loc[tids].tc.median())))
    par = list(range(len(recs)))

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            a, b = recs[i], recs[j]
            if a["cam"] != b["cam"] and abs(a["t"] - b["t"]) <= g["event_window_s"] and len(a["gids"] & b["gids"]) >= 2:
                par[f(i)] = f(j)
    merged = {}
    for i, r in enumerate(recs):
        merged.setdefault(f(i), []).append(r)
    rows = []
    for rs in sorted(merged.values(), key=lambda rs: min(r["t"] for r in rs)):
        gids = set().union(*[r["gids"] for r in rs])
        tg = float(np.median([r["t"] for r in rs]))
        cams = {r["cam"] for r in rs}
        ev = events[events.gid.isin(gids) & ((events.t_end - tg).abs() <= 2 * g["event_window_s"])]
        ev = ev.loc[np.array([bool(cams & set(c.split(";"))) for c in ev.cams], dtype=bool)]
        best = dict(direction="NONE", n=0, t=np.nan, cams="")
        for d in ("UP", "DOWN"):
            e = ev[ev.direction == d]
            if e.gid.nunique() >= 2 and e.gid.nunique() > best["n"]:
                best = dict(direction=d, n=e.gid.nunique(), t=round(e.t_start.median(), 2),
                            cams=";".join(sorted(set(";".join(e.cams).split(";")))))
        rows.append(dict(size=max(r["n"] for r in rs), members=";".join(map(str, sorted(gids))),
                         direction=best["direction"], n_crossing_members=best["n"],
                         t=best["t"] if best["direction"] != "NONE" else round(tg, 2), cams=best["cams"] or ";".join(sorted(cams))))
    out = pd.DataFrame(rows, columns=["size", "members", "direction", "n_crossing_members", "t", "cams"])
    out.insert(0, "group_id", np.arange(1, len(out) + 1))
    return out
