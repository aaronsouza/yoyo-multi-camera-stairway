"""Stage 5: groups = people who stay close, move in the same direction and keep doing so over time."""
import numpy as np
import pandas as pd


def _pairs_in_cam(df, gid_map, g, with_time=False):
    d = df[df.tid.isin(gid_map.index)].copy()
    d["fx"], d["fy"], d["h"] = (d.x1 + d.x2) / 2, d.y2, d.y2 - d.y1
    m = d.merge(d, on="frame", suffixes=("", "_b"))
    m = m[m.tid < m.tid_b]
    m["dist"] = np.hypot(m.fx - m.fx_b, m.fy - m.fy_b) / ((m.h + m.h_b) / 2)
    found = []
    for (a, b), x in m.groupby(["tid", "tid_b"]):
        if len(x) < g["min_common_frames"] or (x.dist <= g["prox_heights"]).mean() < g["frac_close"]:
            continue
        x = x.sort_values("frame")
        va = np.array([x.fx.iloc[-1] - x.fx.iloc[0], x.fy.iloc[-1] - x.fy.iloc[0]])
        vb = np.array([x.fx_b.iloc[-1] - x.fx_b.iloc[0], x.fy_b.iloc[-1] - x.fy_b.iloc[0]])
        hm = x.h.mean()
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na / hm < g["min_move_heights"] or nb / hm < g["min_move_heights"]:
            continue  # standing together is not "walking together"
        if va @ vb / (na * nb) < g["min_cos"]:
            continue
        found.append((a, b, float(x.t.median())) if with_time else (a, b))
    return found


def compute_crossing_groups(tracks, tracklets, events, cfg):
    """Groups of crossings: two same-direction crossing events belong to one group when the two people walked together
    (a close, co-moving tracklet pair in the same camera) AND crossed within a few seconds of each other. The union is
    time-local, so an identity that was re-used for a different person seconds later (merged ID) cannot chain two
    separate crossings into one group. Every group is a set of events (`event_ids`), size = people who crossed together."""
    g, off = cfg["groups"], cfg.get("offsets_s", {})
    pair_t = {}  # (gid, gid) -> times at which the two were walking together
    for cam, df in tracks.items():
        sub = tracklets[tracklets.cam == cam].set_index("tid")
        for a, b, tm in _pairs_in_cam(df, sub.gid, g, with_time=True):
            ga, gb = int(sub.gid[a]), int(sub.gid[b])
            if ga != gb:
                pair_t.setdefault((min(ga, gb), max(ga, gb)), []).append(tm + off.get(cam, 0))
    ev = events.reset_index(drop=True)
    cams = [set(c.split(";")) for c in ev.cams]
    edges = []
    for i in range(len(ev)):
        for j in range(i + 1, len(ev)):
            a, b = ev.iloc[i], ev.iloc[j]
            if a.direction != b.direction or abs(a.t_end - b.t_end) > g.get("pair_gap_s", 4.0) or not cams[i] & cams[j]:
                continue
            tm = pair_t.get((min(a.gid, b.gid), max(a.gid, b.gid)), [])
            mid = (a.t_end + b.t_end) / 2
            if any(abs(t - mid) <= g.get("pair_window_s", 6.0) for t in tm):
                edges.append((abs(a.t_end - b.t_end), i, j))
    par = list(range(len(ev)))

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    members = {i: [i] for i in range(len(ev))}
    for _, i, j in sorted(edges):
        ri, rj = f(i), f(j)
        if ri == rj:
            continue
        both = members[ri] + members[rj]
        te = ev.t_end.values[both]
        if len(both) > g.get("max_size", 8) or te.max() - te.min() > g.get("max_span_s", 8.0):
            continue
        par[rj] = ri
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
    return out


def compute_groups(tracks, tracklets, events, cfg):
    if cfg["groups"].get("mode") == "crossing":
        return compute_crossing_groups(tracks, tracklets, events, cfg)
    """Groups are built per camera from tracklet pairs (no cross-time chaining through identities), then the same
    group seen by several cameras is merged by shared identities + time. Size = most members any one camera saw."""
    g = cfg["groups"]
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
            if 2 <= len(tids) <= g.get("max_size", 8):  # bigger blobs are crowds, not groups
                recs.append(dict(cam=cam, n=len(tids), gids=set(sub.loc[tids].gid), t=float(sub.loc[tids].tc.median())))
    # merge records of the same group seen from several cameras
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
