# Ejo-Vision

Camera-only localization of a race car on Yas Marina. The result is a pose file: time, `x`, `y`, and `yaw` in the map frame. No GNSS and no lidar are used while the car is moving.

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

## 4. Run a different recording

Use the book already in the repository. Pass that recording's own first pose. Do not pass its ground truth.

```bash
python scripts/retrieve_localize.py \
  --mcap PATH_TO_TEST.mcap \
  --first-x X --first-y Y --first-yaw YAW \
  --output output/submission_test.csv
```

`X`, `Y`, and `YAW` are the car's pose at the first speed sample, in the map frame, yaw in radians.

## What the method does

The front camera `fl`, speed, yaw rate, and the first pose are the inputs. Every 0.5 s a 48×24 thumbnail is compared with earlier pages of the picture book. A match is kept only when similarity is at least 0.90 and the page is at least 8 s old. Otherwise the car is moved by speed and yaw rate alone.

There is no trained network. `output/picture_book.npz` is the saved lookup. `scripts/train_pose_net.py` was a trial and is not the pose file above.
