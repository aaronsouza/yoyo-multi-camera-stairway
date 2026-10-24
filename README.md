# Multi-camera stairway people & group counting

YOLO26n (OpenVINO, CPU) → ByteTrack → OSNet-x0.25 ReID → global identities → two-zone stair crossing state machine →
spatiotemporal group detection → Streamlit. Every stage checkpoints to `results/`; the app only reads saved results.

## Setup
```
pip install -r requirements.txt
```
Put the 7 videos in `data/videos/` (camera id = file name). Edit `config.yaml` if needed.

## Run
```
python -m yoyo.cli probe                       # video info + one still per camera -> results/frame_<cam>.jpg
python tools/annotate.py --cam cam1 --mode zones   # draw 'bottom' then 'top' stair zone (press N between, S to save)
python -m yoyo.cli detect --limit-s 120 --cams cam1   # 2-minute benchmark; rerun without --limit-s to resume
python -m yoyo.cli all                         # detect -> embed -> identities -> analyze -> render (resumable)
python -m yoyo.cli analyze                     # re-run counting/groups only (seconds, no inference)
python -m yoyo.cli analyze --force             # also re-do identity matching from saved embeddings
python -m yoyo.cli validate --gt ground_truth.csv --start-s 600 --end-s 900
streamlit run app.py
```

## Checkpointing / storage
| Stage | Saved to | Re-run cost |
|---|---|---|
| detect+track | `results/cams/<cam>/tracks.csv` + `progress.json` (every 300 frames; resumes mid-video) | hours |
| ReID | `results/cams/<cam>/embeddings.npz` (≤8 crops/track, no images stored) | minutes |
| identities, events, groups | `results/final/*.csv`, `summary.json` | seconds |
| annotated video | `results/videos/<cam>.mp4` (960 px, H.264 CRF 30) | minutes |

Tuning crossing or group thresholds in `config.yaml` only needs `analyze`.

## How counting works
- **Crossing**: a person must *dwell* in one stair-end zone and later in the other within `max_cross_s`. Stopping, turning
  around mid-stairs, or touching only one zone creates no event.
- **Cross-camera dedup**: tracklets are merged into global person IDs (cosine ReID similarity, average linkage;
  cannot-link between simultaneous tracklets of one camera; time-gap limits). Zone visits of all cameras are merged
  per identity before transitions are extracted, so a crossing seen by 3 cameras is **one** event, while a person
  who later walks back down is a separate DOWN event. A camera that only sees the top and another only the bottom
  still produce one crossing. `summary.json` reports the naive per-camera sum next to the de-duplicated count.
- **Groups**: two identities are linked if for ≥`min_common_frames` frames they stay within `prox_heights` person
  heights ≥`frac_close` of the time, both actually move, and in the same direction; links form connected components.
  A group counts UP/DOWN if ≥2 members complete the same-direction crossing within `event_window_s`.

## Validation
`ground_truth.csv`: `time_s,direction,group_id` (one row per person crossing; same `group_id` = walked together).
`validate` writes `results/final/validation.json` (precision/recall, missed crossings, false positives, group size
accuracy). Fill the table below after running it and explain each error.

| Segment | GT UP/DOWN | Pred UP/DOWN | Precision | Recall | Groups GT/pred | Notes |
|---|---|---|---|---|---|---|
| _todo_ | | | | | | |

## Bonus: floor-plan stitching
`tools/annotate.py --mode calib` / `--mode floor` write `homographies.yaml` (≥4 matching ground points per camera);
`python -m yoyo.cli calibrate` projects identity tracks to `results/final/floor_tracks.png`. Homographies are valid on
one plane only, so positions on the stairs are approximate.

## Known limitations
- OSNet x0.25 loaded with `weights: null` is only ImageNet-initialised; for real cross-camera ReID supply a
  ReID-trained checkpoint (e.g. MSMT17) via `reid.weights`. Without torchreid it falls back to colour histograms.
- Similar clothing, heavy occlusion and large view changes cause identity merges/splits; thresholds are in `config.yaml`.
- Video timelines are assumed synchronised; use `offsets_s` otherwise.

`python -m pytest tests` runs synthetic checks of the crossing, dedup, group and clustering logic.
