"""Save the development-lap picture book from the training bag.

Ground truth is used here only to label pages. The localizer loads the saved
file and does not open this ground-truth table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ejo_vision.evaluate import interpolate_pose  # noqa: E402
from ejo_vision.retrieve import BOOK_EVERY, DEV_END_S, save_picture_book, thumbnail_descriptor  # noqa: E402
from ejo_vision.stream import iter_speed_and_front  # noqa: E402

MCAP = ROOT / "c0_2026-10-13_22-53-42_camloc" / "c0_2026-10-13_22-53-42_camloc_0.mcap"
GT_CSV = ROOT / "output" / "gt.csv"
BOOK_PATH = ROOT / "output" / "picture_book.npz"


def load_gt(path: Path):
    gt = np.genfromtxt(path, delimiter=",", names=True)
    return (
        gt["timestamp_ns"].astype(np.int64),
        gt["x"].astype(float),
        gt["y"].astype(float),
        np.unwrap(gt["yaw"].astype(float)),
    )


def build(mcap: Path = MCAP, gt_csv: Path = GT_CSV, book_path: Path = BOOK_PATH) -> int:
    if not mcap.is_file():
        raise FileNotFoundError(f"Training bag not found: {mcap}")
    if not gt_csv.is_file():
        raise FileNotFoundError(
            "output/gt.csv is missing. Build it with "
            "c0_2026-10-13_22-53-42_camloc/tools/extract_gt.py"
        )

    tg, gx, gy, gyaw = load_gt(gt_csv)
    dev_end = int(tg[0]) + int(DEV_END_S * 1e9)
    descriptors, times, xs, ys, yaws = [], [], [], [], []
    started = False
    n_fl = 0

    for kind, ts, *payload in iter_speed_and_front(mcap):
        if kind == "speed":
            started = True
            continue
        if not started:
            continue
        n_fl += 1
        if n_fl % BOOK_EVERY != 0 or ts > dev_end:
            continue
        image = cv2.imdecode(np.frombuffer(payload[0], np.uint8), cv2.IMREAD_GRAYSCALE)
        query = thumbnail_descriptor(image)
        if query is None:
            continue
        labeled = interpolate_pose(ts, tg, gx, gy, gyaw)
        if labeled is None:
            continue
        descriptors.append(query)
        times.append(ts)
        xs.append(labeled[0])
        ys.append(labeled[1])
        yaws.append(labeled[2])

    save_picture_book(book_path, np.stack(descriptors), times, xs, ys, yaws)
    print(f"book pages {len(times)}")
    print(f"wrote {book_path.relative_to(ROOT)}")
    return len(times)


if __name__ == "__main__":
    import time

    started = time.perf_counter()
    build()
    print(f"elapsed {time.perf_counter() - started:.1f} s")
