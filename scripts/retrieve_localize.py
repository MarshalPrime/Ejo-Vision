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
from ejo_vision.place_on_line import (  # noqa: E402
    AGREE_M,
    COST_LOCAL,
    LOCAL_M,
    TURN_LOCAL,
    best_hypothesis,
    choose_lock,
    load_profiles,
    match_heading,
)
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
MAP_MCAP = ROOT / "23-00-38_cut_120m_update" / "map.mcap"


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


def _wrap_m(ds: float, lap: float) -> float:
    return (ds + lap / 2.0) % lap - lap / 2.0


def localize_find_start(mcap: Path, book_path: Path, map_mcap: Path) -> tuple:
    """Drive a bag that has speed and yaw rate, and no starting pose.

    Position and orientation are not read. The pose at each frame uses only
    samples up to that frame.
    """
    if not map_mcap.is_file():
        raise FileNotFoundError(f"Map not found: {map_mcap}")
    book = load_picture_book(book_path)
    desc = book["descriptors"]
    book_t = book["timestamp_ns"]
    book_x, book_y, book_yaw = book["x"], book["y"], book["yaw"]
    book_lo, book_hi = int(book_t[0]), int(book_t[-1])
    profiles = load_profiles(map_mcap)

    rows = []
    query_t = []
    reset_t = []
    samples = []
    dist = 0.0
    yaw_i = 0.0
    last = None
    x = y = yaw = None
    locked = None
    s = None
    n_fl = 0
    t_first = None
    pending_before = 0

    def trace():
        dist_now = dist
        yaw_now = yaw_i
        if last is not None and samples and samples[-1][0] != last[0]:
            dist_now = dist
            yaw_now = yaw_i
        return (
            np.array([item[1] for item in samples] + [dist_now]),
            np.array([item[2] for item in samples] + [yaw_now]),
        )

    for kind, ts, *payload in iter_speed_and_front(mcap):
        if kind == "speed":
            vx, vy, wz = payload
            if last is None:
                t_first = ts
                last = (ts, vx, vy, wz)
                samples.append((ts, 0.0, 0.0))
                continue
            dt = (ts - last[0]) / 1e9
            step = float(np.hypot(last[1], last[2]) * dt)
            dist += step
            yaw_i += last[3] * dt
            if x is not None and locked is not None:
                c, sn = np.cos(yaw), np.sin(yaw)
                x += (c * last[1] - sn * last[2]) * dt
                y += (sn * last[1] + c * last[2]) * dt
                yaw += last[3] * dt
                s = (s + step) % locked.lap
            last = (ts, vx, vy, wz)
            if ts - samples[-1][0] >= 1e8:
                samples.append((ts, dist, yaw_i))
            continue

        if last is None:
            continue
        dt = (ts - last[0]) / 1e9
        step = float(np.hypot(last[1], last[2]) * dt)
        dist += step
        yaw_i += last[3] * dt
        if x is not None and locked is not None:
            c, sn = np.cos(yaw), np.sin(yaw)
            x += (c * last[1] - sn * last[2]) * dt
            y += (sn * last[1] + c * last[2]) * dt
            yaw += last[3] * dt
            s = (s + step) % locked.lap
        last = (ts, last[1], last[2], last[3])
        if ts - samples[-1][0] >= 1e8:
            samples.append((ts, dist, yaw_i))

        n_fl += 1
        photo = None
        if n_fl % BOOK_EVERY == 0:
            image = cv2.imdecode(np.frombuffer(payload[0], np.uint8), cv2.IMREAD_GRAYSCALE)
            query = thumbnail_descriptor(image)
            if query is not None:
                other_run = ts < book_lo or ts > book_hi
                index, sim = best_past_match(
                    desc, book_t, query, ts, MIN_AGE_NS, ignore_time=other_run
                )
                query_t.append(ts)
                if index is not None and sim >= SIM_MIN:
                    photo = (float(book_x[index]), float(book_y[index]), float(book_yaw[index]))

        if locked is None:
            if photo is not None:
                x, y, yaw = photo
                gaps = []
                for profile in profiles:
                    s_b, n_b, _, _ = profile.line.xy_to_sn([x], [y], [yaw])
                    gaps.append((abs(float(n_b[0])), profile, float(s_b[0])))
                _, locked, s = min(gaps)
                reset_t.append(ts)
                print(f"start from a picture at {(ts - t_first) / 1e9:.1f} s on {locked.name}")
            else:
                travelled_d, travelled_y = trace()
                unique = choose_lock(profiles, travelled_d, travelled_y) if len(samples) > 20 else None
                if unique is not None:
                    locked, s, cost = unique
                    xx, yy, yyaw = locked.line.sn_to_xy([s], [0.0], [0.0])
                    x, y, yaw = float(xx[0]), float(yy[0]), float(yyaw[0])
                    print(
                        f"start from the map at {(ts - t_first) / 1e9:.1f} s "
                        f"on {locked.name}  cost {cost:.4f}"
                    )
                else:
                    hypo = best_hypothesis(profiles, travelled_d, travelled_y) if len(samples) > 20 else None
                    if hypo is not None:
                        profile, s_end, _cost = hypo
                        xx, yy, yyaw = profile.line.sn_to_xy([s_end], [0.0], [0.0])
                        x, y, yaw = float(xx[0]), float(yy[0]), float(yyaw[0])
                    else:
                        x, y, yaw = 0.0, 0.0, 0.0
                    pending_before += 1
        else:
            window = [item for item in samples if ts - item[0] <= 30e9]
            if photo is not None:
                placed = []
                for profile in profiles:
                    s_b, n_b, _, _ = profile.line.xy_to_sn([photo[0]], [photo[1]], [photo[2]])
                    placed.append((abs(float(n_b[0])), profile, float(s_b[0])))
                on_active = [item for item in placed if item[1] is locked][0]
                if abs(_wrap_m(on_active[2] - s, locked.lap)) <= AGREE_M:
                    _, locked, s = min(placed)
                    x, y, yaw = photo
                    reset_t.append(ts)
            elif len(window) > 10 and window[-1][0] - window[0][0] >= 8e9:
                travelled_d = np.array([item[1] for item in window])
                travelled_y = np.array([item[2] for item in window])
                if np.max(travelled_y) - np.min(travelled_y) > TURN_LOCAL:
                    origin = (s - (dist - travelled_d[0])) % locked.lap
                    s_hat, cost, _second, _turn = match_heading(
                        locked.heading, locked.step, locked.lap, travelled_d, travelled_y, origin, LOCAL_M
                    )
                    if cost < COST_LOCAL:
                        s = s_hat
                        xx, yy, yyaw = locked.line.sn_to_xy([s], [0.0], [0.0])
                        x, y, yaw = float(xx[0]), float(yy[0]), float(yyaw[0])
        rows.append((ts, x, y, yaw))

    print(f"book pages {len(book_t)}   frames before a unique start {pending_before}")
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
    parser.add_argument("--gt", type=Path, default=None, help="Score the run. On the training log, also supplies the first pose.")
    parser.add_argument("--map", type=Path, default=MAP_MCAP)
    parser.add_argument("--find-start", action="store_true", help="Do not use a starting pose. Find it from speed, yaw rate, and the map.")
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
    if args.find_start:
        first_pose = None
    elif all(value is not None for value in given):
        first_pose = given
    elif gt_path is not None and training:
        first_pose = first_pose_from_gt(args.mcap, gt_path)
    else:
        first_pose = None

    if first_pose is None:
        print("no starting pose was given; finding it from speed, yaw rate, and the map")
        rows, query_t, reset_t = localize_find_start(args.mcap, args.book, args.map)
    else:
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
