"""Localize a bag from a frozen picture book.

The book is built by scripts/build_picture_book.py. This script reads the front
camera, speed, yaw rate, and the first pose. It does not read ground truth
while it is driving. A ground-truth file is used only when you ask to score
the training log.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ejo_vision.evaluate import interpolate_pose  # noqa: E402
from ejo_vision.retrieve import (  # noqa: E402
    BOOK_EVERY,
    DEV_END_S,
    MIN_AGE_NS,
    SIM_MIN,
    VAL_END_S,
    best_past_match,
    load_picture_book,
    thumbnail_descriptor,
)
from ejo_vision.stream import iter_speed_and_front  # noqa: E402

TRAIN_MCAP = ROOT / "c0_2026-10-13_22-53-42_camloc" / "c0_2026-10-13_22-53-42_camloc_0.mcap"
GT_CSV = ROOT / "output" / "gt.csv"
BOOK_PATH = ROOT / "output" / "picture_book.npz"
OUT_CSV = ROOT / "output" / "submission_retrieval.csv"


def _gt_arrays(path: Path):
    gt = np.genfromtxt(path, delimiter=",", names=True)
    return (
        gt["timestamp_ns"].astype(np.int64),
        gt["x"].astype(float),
        gt["y"].astype(float),
        np.unwrap(gt["yaw"].astype(float)),
    )


def window_metrics(name, rows, tg, gx, gy, gyaw, t0, t_start_s, t_end_s):
    t_lo, t_hi = t0 + t_start_s * 1e9, t0 + t_end_s * 1e9
    sel = [(t, x, y, yaw) for t, x, y, yaw in rows if t_lo <= t < t_hi]
    if not sel:
        print(f"{name}: no poses")
        return
    ts = np.array([r[0] for r in sel], dtype=np.int64)
    sx = np.array([r[1] for r in sel])
    sy = np.array([r[2] for r in sel])
    syaw = np.array([r[3] for r in sel])
    i = np.searchsorted(tg, ts)
    ok = (i > 0) & (i < len(tg))
    i = np.clip(i, 1, len(tg) - 1)
    ok &= (tg[i] - tg[i - 1]) <= 50_000_000
    ts, sx, sy, syaw, i = ts[ok], sx[ok], sy[ok], syaw[ok], i[ok]
    a = (ts - tg[i - 1]) / (tg[i] - tg[i - 1])
    gxx = gx[i - 1] + a * (gx[i] - gx[i - 1])
    gyy = gy[i - 1] + a * (gy[i] - gy[i - 1])
    gyyaw = gyaw[i - 1] + a * (gyaw[i] - gyaw[i - 1])
    dx, dy = sx - gxx, sy - gyy
    lon = np.cos(gyyaw) * dx + np.sin(gyyaw) * dy
    lat = -np.sin(gyyaw) * dx + np.cos(gyyaw) * dy
    pos = np.hypot(dx, dy)
    dyaw = np.degrees(np.angle(np.exp(1j * (syaw - gyyaw))))
    rms = lambda v: float(np.sqrt(np.mean(v**2)))
    print(
        f"{name:12s}  n={len(pos):5d}  pos {rms(pos):6.1f} m  "
        f"lat {rms(lat):6.1f} m  lon {rms(lon):6.1f} m  "
        f"yaw {rms(dyaw):5.1f} deg  within 1 m {100 * np.mean(pos < 1):4.1f}%"
    )


def localize(mcap: Path, book_path: Path, first_pose: tuple[float, float, float]) -> list[tuple]:
    book = load_picture_book(book_path)
    desc = book["descriptors"]
    book_t = book["timestamp_ns"]
    book_x, book_y, book_yaw = book["x"], book["y"], book["yaw"]

    rows = []
    query_t = []
    reset_t = []
    x = y = yaw = None
    last = None
    pending = []
    pending_i = 0
    n_fl = 0

    def apply_pending(until_ns: int) -> None:
        nonlocal x, y, yaw, last, pending_i
        while pending_i < len(pending) and pending[pending_i][0] <= until_ns:
            ts, vx, vy, wz = pending[pending_i]
            pending_i += 1
            dt = (ts - last[0]) / 1e9
            c, s = np.cos(yaw), np.sin(yaw)
            x += (c * last[1] - s * last[2]) * dt
            y += (s * last[1] + c * last[2]) * dt
            yaw += last[3] * dt
            last = (ts, vx, vy, wz)

    for kind, ts, *payload in iter_speed_and_front(mcap):
        if kind == "speed":
            vx, vy, wz = payload
            if x is None:
                x, y, yaw = first_pose
                last = (ts, vx, vy, wz)
                continue
            pending.append((ts, vx, vy, wz))
            continue

        if x is None:
            continue
        apply_pending(ts)
        n_fl += 1
        if n_fl % BOOK_EVERY == 0:
            image = cv2.imdecode(np.frombuffer(payload[0], np.uint8), cv2.IMREAD_GRAYSCALE)
            query = thumbnail_descriptor(image)
            if query is not None:
                index, sim = best_past_match(desc, book_t, query, ts, MIN_AGE_NS)
                query_t.append(ts)
                if index is not None and sim >= SIM_MIN:
                    x = float(book_x[index])
                    y = float(book_y[index])
                    yaw = float(book_yaw[index])
                    reset_t.append(ts)
        rows.append((ts, x, y, yaw))

    print(f"book pages {len(book_t)}")
    return rows, query_t, reset_t


def write_csv(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_ns", "x", "y", "yaw"])
        for ts, px, py, pyaw in rows:
            writer.writerow([ts, f"{px:.4f}", f"{py:.4f}", f"{pyaw:.6f}"])
    print(f"wrote {len(rows)} rows -> {path.name}")


def first_pose_from_gt(mcap: Path, gt_csv: Path) -> tuple[float, float, float]:
    tg, gx, gy, gyaw = _gt_arrays(gt_csv)
    for kind, ts, *_payload in iter_speed_and_front(mcap):
        if kind == "speed":
            pose = interpolate_pose(ts, tg, gx, gy, gyaw)
            if pose is None:
                raise RuntimeError("First speed sample has no ground-truth pose within 50 ms.")
            return pose
    raise RuntimeError(f"No speed topic in {mcap}")


def _split_counts(times: list[int], dev_end: int) -> tuple[int, int]:
    dev = sum(ts <= dev_end for ts in times)
    return dev, len(times) - dev


def score(rows, query_t, reset_t, gt_csv: Path) -> None:
    tg, gx, gy, gyaw = _gt_arrays(gt_csv)
    t0 = int(tg[0])
    dev_end = t0 + int(DEV_END_S * 1e9)
    q_dev, q_rest = _split_counts(query_t, dev_end)
    r_dev, r_rest = _split_counts(reset_t, dev_end)
    print(f"resets dev {r_dev}/{q_dev}   after dev {r_rest}/{q_rest}")
    window_metrics("development", rows, tg, gx, gy, gyaw, t0, 0.0, DEV_END_S)
    window_metrics("validation", rows, tg, gx, gy, gyaw, t0, DEV_END_S, VAL_END_S)
    window_metrics("tail", rows, tg, gx, gy, gyaw, t0, VAL_END_S, 1e7)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Localize from a frozen picture book.")
    parser.add_argument("--mcap", type=Path, default=TRAIN_MCAP)
    parser.add_argument("--book", type=Path, default=BOOK_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--gt", type=Path, default=None, help="Score the run. Also supplies the first pose when --first-x is omitted.")
    parser.add_argument("--first-x", type=float)
    parser.add_argument("--first-y", type=float)
    parser.add_argument("--first-yaw", type=float)
    args = parser.parse_args(argv)
    given = (args.first_x, args.first_y, args.first_yaw)
    if any(value is not None for value in given) and any(value is None for value in given):
        parser.error("Pass --first-x, --first-y, and --first-yaw together.")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    training = args.mcap.resolve() == TRAIN_MCAP.resolve()
    if not args.book.is_file():
        raise FileNotFoundError("Picture book missing. Run: python scripts/build_picture_book.py")
    if not args.mcap.is_file():
        raise FileNotFoundError(f"Bag not found: {args.mcap}")

    output = args.output
    if output is None:
        if not training:
            raise SystemExit("Pass --output when the bag is not the training log.")
        output = OUT_CSV

    gt_path = args.gt
    if gt_path is None and training and GT_CSV.is_file():
        gt_path = GT_CSV

    given = (args.first_x, args.first_y, args.first_yaw)
    if all(value is not None for value in given):
        first_pose = given
    elif gt_path is not None:
        first_pose = first_pose_from_gt(args.mcap, gt_path)
    else:
        raise SystemExit(
            "Pass --first-x, --first-y, and --first-yaw. The picture book does not contain the test pose."
        )

    rows, query_t, reset_t = localize(args.mcap, args.book, first_pose)
    write_csv(output, rows)
    if gt_path is not None:
        score(rows, query_t, reset_t, gt_path)
    else:
        print(f"resets {len(reset_t)}/{len(query_t)}")


if __name__ == "__main__":
    started = time.perf_counter()
    main()
    print(f"elapsed {time.perf_counter() - started:.1f} s")
