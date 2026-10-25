# Multi-Camera Stairway People & Group Counting

A CPU-only computer-vision system for seven synchronized street CCTV cameras (1920×1080, 59.94 fps, ~35 min each) overlooking one plaza and a staircase.
It tracks every person, counts **completed** UP/DOWN stair crossings once across cameras, detects walking groups by direction and size, and presents the
results in a Streamlit dashboard. A bonus module places all seven cameras on one metric floor plan and stitches trajectories across cameras.

**Live dashboard:** https://yoyo-multi-camera-stairway-qbbevhnrbrthuaxvqyaczz.streamlit.app/

## Highlights
| Requirement | Status |
|---|---|
| Detection + tracking with IDs, annotated video (all 7 cameras) | ✅ |
| Completed UP/DOWN crossings: stops, turn-arounds and partial walks are not counted | ✅ |
| Same person / same crossing across cameras counted once | ✅ |
| Groups of 2+ by direction and size; standing groups reported separately | ✅ |
| Streamlit demo | ✅ |
| Bonus: multi-camera calibration, floor plan, track stitching | ✅ |

## Architecture
```
videos ─► YOLO26n (OpenVINO, 960 px, 10 fps) ─► ByteTrack ─► tracks
                   └► OSNet-x0.25 (MSMT17) + HSV ─► embeddings ─► global IDs (average linkage, cannot-links)
stair zones ─► two-zone crossing state machine ─► events ─► cross-camera dedup ─► crossing groups ─► Streamlit
bonus: paving-seam calibration ─► common floor plan ─► stitched trajectories ─► Streamlit
```
- **Efficient on a laptop CPU (i7-10510U, no GPU):** OpenVINO detector at ~6–7 processed frames/s. Detection and ReID run once and are checkpointed
  (resumable, every 300 frames); counting, grouping and calibration re-run from saved tracks without inference.
- **Robust crossings:** a person must dwell in one stair-end zone and reach the other within 25 s on a continuous, correctly directed path. Cam5 adds
  widened zones, frame-edge exit inference and direction-flip suppression.
- **Cross-camera counting:** ReID-based global IDs, plus one-to-one time/direction matching of cam5↔cam2 crossings. 794 raw per-track transitions become
  614 counted crossings.
- **Groups:** same-direction crossings of people tracked walking together within 4 s, built in one camera's view (so one person is one member).
- **Bonus calibration without a floor plan:** one shared focal length (1566 px) and each camera's ground orientation come from the paving seams (median
  fit error 0.27–0.37°); cameras are tied together by time-synchronized foot-point voting, with no ReID, and refined jointly.

## Setup & run
```
pip install -r requirements-pipeline.txt   # full pipeline; videos in data/videos/, OSNet MSMT17 weights at models/osnet_x0_25_msmt17.pth
                                           # the dashboard alone only needs requirements.txt
python -m yoyo.cli all              # detect -> embed -> identities -> analyze -> render
python -m yoyo.cli analyze          # counting + groups only (no inference)
python -m yoyo.cli calibrate && python -m yoyo.cli floorplan && python -m yoyo.cli stitch   # bonus
streamlit run app.py
python -m pytest tests              # 25 tests
```
Stair zones: `zones.yaml` (`tools/annotate.py`). Thresholds: `config.yaml`.

## Results
Whole recording: **207 UP / 407 DOWN crossings, 36 UP / 74 DOWN groups**, 15,505 tracklets in 5,024 global IDs (an upper bound on distinct people).

**Ground truth comparison** — Cam5, 25:00–27:00, manual count vs system (totals; agreement = system / manual):

| Metric | Manual | Final | Agreement |
|---|---|---|---|
| Persons UP | 12 | 11 | 92% |
| Persons DOWN | 15 | 14 | 93% |
| Persons total | 27 | 25 | 93% |
| Solo UP | 5 | 4 | 80% |
| Solo DOWN | 7 | 6 | 86% |
| Groups UP | 3 (sizes 2, 2, 3) | 3 (sizes 2, 2, 3) | exact |
| Groups DOWN | 3 (sizes 2, 2, 4) | 3 (sizes 2, 3, 3) | count exact, sizes differ |
| Entities UP (group = 1) | 8 | 7 | 88% |
| Entities DOWN (group = 1) | 10 | 9 | 90% |
| Entities total | 18 | 16 | 89% |

**Event-level check** (same window, each of the 25 counted crossings reviewed by hand): precision **100%** (25/25 correct, no duplicates, wrong directions or false counts), recall **92.6%** (25 of 27 real crossings; UP 11/12, DOWN 14/15).

**Bonus (`calibration/validation.json`)**

| Check | Result |
|---|---|
| Independent landmarks, 3 static objects, 13 observations | agree within rms 0.61 m (median 0.27 m) |
| Camera pairs vs time-shifted chance | all 21 pairs show real matches (9–41%) |
| Third-camera prediction check | person within 1 m of the predicted spot in 77% of 600 cases (chance 61%) |
| Trajectories | 5,848 from 4,460 global IDs; 2,915 seen by 2+ cameras |
| Stitching vs blind identity review (AI-judged, small sample) | 16/16 decidable joins correct; 6/7 decidable splits correct |

## Dashboard (`app.py`)
- **Proof of concept:** 30 s annotated clips for tracking (all cameras), crossings, cross-camera matching, groups, and all four together. Clips are generated
  by `tools/make_*.py` into `evidence/`.
- **Results:** headline counts, ground-truth comparison, crossing and group tables.
- **Bonus tab:** interactive floor plan with camera coverage, person search, trajectory replay, camera views at a chosen moment, calibration status.

## Repository
| Path | Purpose |
|---|---|
| `yoyo/` | pipeline (`detect`, `embed`, `identities`, `events`, `groups`, `report`, `render`, `validate`) and bonus (`geometry`, `register`, `calibrate`, `stitch`, `floorplan`, `bonus_view`) |
| `config.yaml`, `zones.yaml` | thresholds, stair zones |
| `calibration/` | calibration, landmarks, validation, review results |
| `tools/` | zone annotation and clip generation |
| `tests/` | synthetic tests |
