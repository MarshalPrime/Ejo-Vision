# Ejo-Vision

This project explores camera-based localization for autonomous racing. We aim to estimate a vehicle’s position, movement, and heading from onboard camera images without relying on GPS/GNSS or LiDAR. Our mission is to develop, test, and improve robust visual localization methods for high-speed racing environments at the SchubKI National AI Hackathon 2026.

Reusable code lives in `src/ejo_vision/`. The working notebook is `yas_marina_camera_localization.ipynb`.

## Current method (Yas Marina)

Estimate `x, y, yaw` in the map frame. Option B (`s, n`) is the same pose converted with `23-00-38_cut_120m_update/map.mcap`.

The localizer uses the front camera `fl`, speed and yaw rate, and the first pose. Every 0.5 s it compares a 48×24 thumbnail with earlier frames from the first 896 s of this training run. A match is kept only when similarity is at least 0.90 and the page is at least 8 s old. Otherwise the car is moved by speed and yaw rate alone. Live ground truth is not an input. Past training poses are the labels of the picture book.

Held-out laps are 896 s to 1121 s. On those laps the last run was about **4.2 m** position RMSE, **0.9 m** lateral, **0.7°** yaw, with **100%** coverage. The early laps are worse (about 78 m) because some night views match the wrong part of the track.

Two commands, after the training bag is at `c0_2026-10-13_22-53-42_camloc/` and `output/gt.csv` exists:

```bash
python scripts/build_picture_book.py
python scripts/retrieve_localize.py
```

The first command writes `output/picture_book.npz` (1792 pages from the first 896 s). The second loads that file, drives the bag, and writes `output/submission_retrieval.csv`. It prints development, validation, and tail errors. Ground truth is not read while the car is moving. On the training log it is used for the first pose and for the printed scores.

Building the book took **21.6 s**. The localize pass took **24.8 s** on an Apple Silicon Mac (Python 3.11), including reading the 7 GB bag. That is about **2 ms** per front-camera frame, under the 100 ms camera period. The pose file matches the earlier single-pass result: held-out laps stay at **4.2 m** / **0.9 m** / **0.7°**.

A different bag uses the same book and that bag's own first pose. Do not pass its ground truth:

```bash
python scripts/retrieve_localize.py \
  --mcap PATH_TO_TEST.mcap \
  --first-x X --first-y Y --first-yaw YAW \
  --output output/submission_test.csv
```

Build `output/gt.csv` once from the organizers' script if it is missing:

```bash
python c0_2026-10-13_22-53-42_camloc/tools/extract_gt.py \
  c0_2026-10-13_22-53-42_camloc/c0_2026-10-13_22-53-42_camloc_0.mcap \
  output/gt.csv
```

Ground truth in that file is for the picture-book labels and for scoring. Do not feed `/tf` or live pose into a new estimator.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Run checks that do **not** need OpenCV, Matplotlib, or a dataset:

```bash
python3 -m pytest
```

`output/` is in the repository. It contains the pose files, the validation plot, the picture book, and `gt.csv`.

Three folders stay on the machine that holds the recordings, because they are too large for git:

- `c0_2026-10-13_22-53-42_camloc/` — training bag
- `23-00-38_cut_120m_update/` — map and the organizers' scoring tools
- `rosbag_2026-10-13_16-08-46_cameras_lap/` — daylight example bag

With those folders present locally, the two commands above rebuild the book and the pose file. Without them, the committed `output/submission_retrieval.csv` and `output/retrieval_validation_laps.png` are the results to read.
