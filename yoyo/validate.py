"""Compare predicted events/groups against a manually verified segment.

ground_truth.csv columns: time_s, direction (UP|DOWN), group_id (blank = walked alone; same id = walked together)
Segment is [start_s, end_s] on the common timeline.
"""
import pandas as pd
from .common import final_dir, write_json


def match(pred, gt, tol):
    used, tp, fn = set(), [], []
    for i, g in gt.iterrows():
        best, bd = None, tol + 1e-9
        for j, p in pred.iterrows():
            d = abs(p.t_start - g.time_s)
            if j not in used and p.direction == g.direction and d <= bd:
                best, bd = j, d
        if best is None:
            fn.append(g.to_dict())
        else:
            used.add(best)
            tp.append((i, best))
    fp = [pred.loc[j].to_dict() for j in pred.index if j not in used]
    return tp, fp, fn


def run(cfg, gt_path, start_s, end_s, tol=8.0, log=print):
    fd = final_dir(cfg)
    gt = pd.read_csv(gt_path).dropna(subset=["time_s", "direction"])
    gt["direction"] = gt.direction.str.strip().str.upper()
    gt = gt[(gt.time_s >= start_s) & (gt.time_s <= end_s)]
    ev = pd.read_csv(fd / "events.csv")
    ev = ev[(ev.t_start >= start_s - tol) & (ev.t_start <= end_s + tol)]
    tp, fp, fn = match(ev, gt, tol)
    P = len(tp) / max(1, len(tp) + len(fp))
    R = len(tp) / max(1, len(tp) + len(fn))
    res = dict(segment=[start_s, end_s], tolerance_s=tol,
               gt_up=int((gt.direction == "UP").sum()), pred_up=int((ev.direction == "UP").sum()),
               gt_down=int((gt.direction == "DOWN").sum()), pred_down=int((ev.direction == "DOWN").sum()),
               tp=len(tp), fp=len(fp), fn=len(fn), precision=round(P, 3), recall=round(R, 3),
               false_positives=fp, missed=fn)
    if "group_id" in gt:  # groups: ground-truth groups with >=2 members vs predicted groups in the segment
        gg = gt.dropna(subset=["group_id"]).groupby("group_id").agg(n=("time_s", "size"), d=("direction", "first"),
                                                                     t=("time_s", "median")).reset_index()
        pg = pd.read_csv(fd / "groups.csv")
        pg = pg[(pg.direction != "NONE") & (pg.t >= start_s - tol) & (pg.t <= end_s + tol)]
        pred_g = [dict(t=r.t, direction=r.direction, size=r.size) for r in pg.itertuples()]
        ok_dir = ok_size = 0
        left = list(pred_g)
        for r in gg.itertuples():
            m = [p for p in left if p["direction"] == r.d and abs(p["t"] - r.t) <= tol]
            if m:
                p = min(m, key=lambda p: abs(p["t"] - r.t))
                left.remove(p)
                ok_dir += 1
                ok_size += int(p["size"] == r.n)
        res.update(gt_groups=len(gg), pred_groups=len(pred_g), groups_matched=ok_dir, group_size_correct=ok_size)
    write_json(fd / "validation.json", res)
    log({k: v for k, v in res.items() if k not in ("false_positives", "missed")})
    return res
