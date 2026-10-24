"""Checklist 4 evidence: 30 s clip with coloured crossing-group boxes (group id, size, UP/DOWN) and dashed standing-group boxes,
from saved results (no inference), plus a review sheet of the groups in the interval. Usage: python tools/make_group_evidence.py cam5 1500 1620"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yoyo.common import ROOT, final_dir, load_config, videos  # noqa: E402
from yoyo.render import render_camera  # noqa: E402

DUR = 30


def main(cam, lo, hi):
    cfg = load_config()
    fd = final_dir(cfg)
    gr, ev = pd.read_csv(fd / "groups.csv"), pd.read_csv(fd / "events.csv").set_index("event_id")
    sg = pd.read_csv(fd / "standing_groups.csv")
    gr = gr[gr.cams.str.split(";").apply(lambda c: cam in c)].copy()
    gr["video_time"] = gr.t.apply(lambda t: f"{int(t // 60):02d}:{t % 60:04.1f}")
    gr["person_ids"] = gr.members
    gr["event_ids_in_group"] = gr.event_ids
    out_dir = ROOT / "evidence" / "checklist4_groups"
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = gr[gr.t.between(lo, hi)].copy()
    sheet["manual_size"] = ""
    sheet["verdict (OK / WRONG_SIZE / NOT_A_GROUP)"] = ""
    sheet[["group_id", "video_time", "direction", "size", "person_ids", "event_ids_in_group", "manual_size",
           "verdict (OK / WRONG_SIZE / NOT_A_GROUP)"]].to_csv(out_dir / f"{cam}_{lo}-{hi}_group_review_sheet.csv", index=False)
    stand = sg[(sg.cam == cam) & (sg.t0 < hi) & (sg.t1 > lo)]

    def score(s):  # busiest window: crossing groups of both directions first, then standing groups
        g = gr[gr.t.between(s, s + DUR)]
        return 3 * len(g) + 2 * (g.direction.nunique() == 2) + 0.5 * ((stand.t0 < s + DUR) & (stand.t1 > s)).sum()
    s0 = max(range(lo, hi - DUR + 1), key=score)
    out = out_dir / f"{cam}_groups_{s0}-{s0 + DUR}s.mp4"
    render_camera(cfg, cam, videos(cfg)[cam], s0, s0 + DUR, force=True, out_path=out)
    g = gr[gr.t.between(s0, s0 + DUR)]
    g[["group_id", "video_time", "direction", "size", "person_ids"]].to_csv(out_dir / f"{cam}_groups_{s0}-{s0 + DUR}s_groups.csv", index=False)
    print(out, "crossing groups in clip:", len(g), g.direction.value_counts().to_dict(), "sizes", list(g["size"]),
          "| standing groups visible:", int(((stand.t0 < s0 + DUR) & (stand.t1 > s0)).sum()))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
