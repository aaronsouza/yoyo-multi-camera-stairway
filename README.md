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

## Bonus: multi-camera calibration and track stitching
All seven cameras are placed on one floor plan (metres, plaza ground plane) and tracks are stitched across cameras. It reuses the saved tracks and
global IDs; no detector or ReID is run. Code: `yoyo/geometry.py`, `register.py`, `calibrate.py`, `floorplan.py`, `stitch.py`, `calibration_check.py`,
`bonus_view.py` (the dashboard tab). Results of the calibration itself are versioned in `calibration/`.

```
python -m yoyo.cli calibrate   # seams -> ground orientation, heights, cameras tied together  -> calibration/calibration.json   (~15 min)
python -m yoyo.cli floorplan   # plan picture + coverage                                       -> results/calibration/
python -m yoyo.cli stitch      # trajectories across cameras                                   -> results/calibration/trajectories.csv.gz
python -m yoyo.cli calcheck    # landmark + time-shift + third-camera validation               -> calibration/validation.json
streamlit run app.py           # tab "Multi-Camera Calibration & Track Stitching (Bonus)"
```
**Method.** No floor plan or camera geometry was available, so none is assumed. (1) The plaza is paved with rectangular slabs: the seam lines form two
perpendicular families. Fitting them in all cameras with one shared focal length gives each camera's ground-plane orientation (shared focal length 1566 px;
median seam error 0.27-0.37 degrees). (2) Camera height comes from people's pixel height (adults taken as 1.7 m, about +-15%), which makes each plan
metric. (3) The plans differ only by a rotation (multiple of 90 degrees), a small scale and a translation; they are found by letting every pair of foot
points seen at the same instant vote for a translation, then refining all cameras jointly. This uses no identities, so the stitching test below is independent.
(4) Tracklets of one global ID become one trajectory only if they agree in space and time (<= 5 m where two cameras overlap, a possible walk across gaps);
otherwise the ID is split. Gaps stay gaps: positions are never interpolated, and stairs / far-field points are not mapped.

| Check | Result |
|---|---|
| Landmarks (3 static objects read by eye, 13 observations, not used for fitting) | agree within rms 0.61 m, median 0.27 m, worst 1.24 m |
| Same-instant foot points, all 21 camera pairs | every pair beats time-shifted chance: 9-41% of matches are real |
| Third camera (predicted spot of a person seen by two cameras) | a detected person within 1 m in 77% of 600 cases (chance 61%; the plaza is crowded) |
| Stitching vs reviewed identities (round 3, out-of-sample) | 16 of 16 decidable joins are the same person (8 more "cannot tell"); 6 of 7 decidable splits were really different people |
| Result | 5,848 trajectories from 4,460 global IDs, 2,915 seen by 2+ cameras, 1,174 IDs split because they mixed people |

**Limits.** The identity review was done by the AI assistant from crops, not by a human, on small samples (see `calibration/stitch_review_summary.json`);
earlier rounds (greedy rule) were worse and were used to improve the rule. Lens distortion is not modelled, positions beyond ~6 camera heights are only
approximate, and the staircase (not on the plaza plane) is left unmapped. Scale is approximate (+-15%) and the plan axes follow cam1's seams (rotated so cam5 looks up).

## Known limitations
- OSNet x0.25 loaded with `weights: null` is only ImageNet-initialised; for real cross-camera ReID supply a
  ReID-trained checkpoint (e.g. MSMT17) via `reid.weights`. Without torchreid it falls back to colour histograms.
- Similar clothing, heavy occlusion and large view changes cause identity merges/splits; thresholds are in `config.yaml`.
- Video timelines are assumed synchronised; use `offsets_s` otherwise.

`python -m pytest tests` runs synthetic checks of the crossing, dedup, group and clustering logic.
