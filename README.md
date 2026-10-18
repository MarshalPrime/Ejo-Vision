# Ejo-Vision

Camera-only localization of a race car on Yas Marina. The result is a pose file: time, `x`, `y`, and `yaw` in the map frame. No GNSS and no lidar are used while the car is moving.

No public dataset is used. No pretrained model is used. The daylight example bag was not used to train, tune, or score anything. `scripts/train_pose_net.py` is an unused trial trained only on this same night run.

The pose file is already in this repository. You only need the large recordings if you want to rebuild it.

## 1. Read the result

Open these two files. Nothing else has to be installed.

| File | What it is |
|---|---|
| `output/submission_retrieval.csv` | Pose for the full training log. Columns: `timestamp_ns`, `x`, `y`, `yaw`. |
| `output/retrieval_validation_laps.png` | Estimated path (orange) on the ground-truth path (blue) for the held-out laps. |

The held-out laps are 896 s to 1121 s. On that window the score is about **4.2 m** position, **0.9 m** sideways, **0.7°** heading, with **100%** coverage. The same slice is saved as `output/submission_retrieval_validation.csv` and `output/gt_validation_laps.csv`.

The first 896 s are about **78 m**. Some night views match the wrong part of the track, and the file is not corrected there.

## 2. Set up Python

Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

On Windows, activate with `.venv\Scripts\activate`.

Checks that do not need the recordings:

```bash
python3 -m pytest
```

The notebook `yas_marina_camera_localization.ipynb` is the same work with the pictures inline. The scripts below are the run path.

## 3. Rebuild the pose file

Skip this section if you only need the files already in `output/`.

Put these three folders next to this README. They are not in git because the training bag is about 7 GB.

- `c0_2026-10-13_22-53-42_camloc/` — training recording
- `23-00-38_cut_120m_update/` — map and the organizers' scoring script
- `rosbag_2026-10-13_16-08-46_cameras_lap/` — daylight example, only if you want to look at that bag

`output/gt.csv` is already in the repository. From the project folder:

```bash
python scripts/build_picture_book.py
python scripts/retrieve_localize.py
```

The first command reads the training bag and writes `output/picture_book.npz` (1792 pages from the first 896 s). The second command loads that book, drives the bag, and writes `output/submission_retrieval.csv`. It then prints development, validation, and tail errors.

Ground truth is used to label the book and to print those errors. It is not read while the car is moving. The first pose of the training log is taken from `output/gt.csv`.

On an Apple Silicon Mac with Python 3.11, building the book took **21.6 s** and the localize pass took **24.8 s**, including reading the bag. That is about **2 ms** per front-camera frame.

Score only the held-out laps with the organizers' script:

```bash
python 23-00-38_cut_120m_update/tools/evaluate.py \
  output/gt_validation_laps.csv \
  output/submission_retrieval_validation.csv
```

## 4. Run a recording that has no starting pose

The problem statement allows the first pose of a run. A later message from the organizers says the hidden test removes the pose fields and does not give a start. This command covers that case. It does not read position or orientation, and it does not take `--first-x`.

```bash
python scripts/retrieve_localize.py \
  --mcap PATH_TO_TEST.mcap \
  --find-start \
  --output output/submission_test.csv
```

Leave out `--find-start` and the script still finds the start, as long as you do not pass `--first-x`, `--first-y`, and `--first-yaw`. If you do pass them, that pose is used.

The start comes from speed, yaw rate, and the map. A straight fits every part of the lap, so it is refused. The first place that fits, and whose next-best place is much worse, becomes the start. On this training log that happens at **42.9 s**, on the pit line. After that, a front-camera match may move the car when it lies on the same part of the lap, including from the pit line onto the racing line.

Scored on the training log with no starting pose:

| Window | Position | Sideways | Heading | Inside 1 m |
|---|---|---|---|---|
| First 42.9 s, before the turn is unique | 462 m | 438 m | 57° | 4% |
| After that, until 896 s | 21 m | 16 m | 5.4° | 17% |
| Held-out laps, 896 s to 1121 s | 3.6 m | 1.0 m | 1.7° | 16% |
| After 1121 s | 2.5 m | 0.3 m | 0.7° | 38% |

The held-out file for the run that does use the first pose is still `output/submission_retrieval.csv` (about 4.2 m, 0.9 m sideways, inside 1 m on 33% of frames). `output/submission_find_start.csv` is the run with no starting pose. The code reads `vx_mps`, `vy_mps`, and `wz_radps` only. It does not read `x`, `y`, yaw, `s`, or `n` from `/constructor0/state_estimation`.

## What the method does

The front camera `fl`, speed, and yaw rate are the inputs. Every 0.5 s a 48×24 thumbnail is compared with the picture book. On this same training log a page must be at least 8 s old, so a future frame of this drive is never used. On another recording the book is earlier training, not a future frame, so every page is allowed. A match is kept only when similarity is at least 0.90 and it lies within 30 m, along the lap, of the map place. Otherwise the car stays on the racing line or the pit line.

There is no trained network. `output/picture_book.npz` is the saved lookup.
