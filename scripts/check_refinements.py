"""Two changes checked against the front-only result. Neither replaces it.

- dense: save and compare a front photo on every frame, then copy that page's pose.
- slide: copy the page when the guess is far from it. When the guess is already
  within 25 m, keep its along-track distance (at most 12 m from the page) and
  take the sideways position and heading from the page.

The development laps decide whether a change helps. The locked laps are printed
once, with the rules fixed beforehand.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ejo_vision.evaluate import interpolate_pose  # noqa: E402
from ejo_vision.retrieve import (  # noqa: E402
    BOOK_EVERY,
    DEV_END_S,
    MIN_AGE_NS,
    SIM_MIN,
    THUMB_H,
    THUMB_W,
    VAL_END_S,
    best_past_match,
    thumbnail_descriptor,
)
from ejo_vision.stream import iter_speed_and_front  # noqa: E402
from retrieve_localize import GT_CSV, OUT_CSV, TRAIN_MCAP, _gt_arrays, window_metrics  # noqa: E402

NEAR_M = 25.0
SLIDE_MAX_M = 12.0
DESC = THUMB_W * THUMB_H


class Drive:
    def __init__(self, pose: tuple[float, float, float]):
        self.x, self.y, self.yaw = pose
        self.last = None
        self.pending_i = 0

    def start(self, ts: int, vx: float, vy: float, wz: float) -> None:
        self.last = (ts, vx, vy, wz)

    def apply_until(self, pending: list, until_ns: int) -> None:
        while self.pending_i < len(pending) and pending[self.pending_i][0] <= until_ns:
            ts, vx, vy, wz = pending[self.pending_i]
            self.pending_i += 1
            dt = (ts - self.last[0]) / 1e9
            c, s = np.cos(self.yaw), np.sin(self.yaw)
            self.x += (c * self.last[1] - s * self.last[2]) * dt
            self.y += (s * self.last[1] + c * self.last[2]) * dt
            self.yaw += self.last[3] * dt
            self.last = (ts, vx, vy, wz)

    def reset(self, pose: tuple[float, float, float]) -> None:
        self.x, self.y, self.yaw = pose

    @property
    def pose(self) -> tuple[float, float, float]:
        return (float(self.x), float(self.y), float(self.yaw))


class Book:
    def __init__(self, capacity: int):
        self.desc = np.empty((capacity, DESC), dtype=np.float32)
        self.t = np.empty(capacity, dtype=np.int64)
        self.x = np.empty(capacity, dtype=np.float64)
        self.y = np.empty(capacity, dtype=np.float64)
        self.yaw = np.empty(capacity, dtype=np.float64)
        self.n = 0

    def add(self, query: np.ndarray, ts: int, pose: tuple[float, float, float]) -> None:
        i = self.n
        self.desc[i] = query
        self.t[i] = ts
        self.x[i], self.y[i], self.yaw[i] = pose
        self.n = i + 1

    def match(self, query: np.ndarray, ts: int) -> tuple[float, float, float] | None:
        if self.n == 0:
            return None
        index, sim = best_past_match(self.desc[: self.n], self.t[: self.n], query, ts, MIN_AGE_NS)
        if index is None or sim < SIM_MIN:
            return None
        return (float(self.x[index]), float(self.y[index]), float(self.yaw[index]))


def slide_pose(predicted: tuple[float, float, float], booked: tuple[float, float, float]):
    px, py, _pyaw = predicted
    bx, by, byaw = booked
    dx, dy = px - bx, py - by
    if dx * dx + dy * dy > NEAR_M * NEAR_M:
        return booked
    hx, hy = float(np.cos(byaw)), float(np.sin(byaw))
    lon = float(np.clip(dx * hx + dy * hy, -SLIDE_MAX_M, SLIDE_MAX_M))
    return (bx + lon * hx, by + lon * hy, byaw)


def run() -> None:
    tg, gx, gy, gyaw = _gt_arrays(GT_CSV)
    t0 = int(tg[0])
    dev_end = t0 + int(DEV_END_S * 1e9)
    coarse = Book(4_000)
    dense = Book(12_000)
    pending: list = []
    front = slide = every = None
    rows = {"front": [], "slide": [], "dense": []}
    n_fl = 0

    for kind, ts, *payload in iter_speed_and_front(TRAIN_MCAP):
        if kind == "speed":
            vx, vy, wz = payload
            if front is None:
                pose = interpolate_pose(ts, tg, gx, gy, gyaw)
                front = Drive(pose)
                slide = Drive(pose)
                every = Drive(pose)
                front.start(ts, vx, vy, wz)
                slide.start(ts, vx, vy, wz)
                every.start(ts, vx, vy, wz)
                continue
            pending.append((ts, vx, vy, wz))
            continue

        if front is None:
            continue
        jpeg = payload[0]
        for drive in (front, slide, every):
            drive.apply_until(pending, ts)
        n_fl += 1
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
        query = thumbnail_descriptor(image)
        if query is not None:
            dense_pose = dense.match(query, ts)
            if dense_pose is not None:
                every.reset(dense_pose)
            if n_fl % BOOK_EVERY == 0:
                booked = coarse.match(query, ts)
                if booked is not None:
                    front.reset(booked)
                    slide.reset(slide_pose(slide.pose, booked))
            if ts <= dev_end:
                labeled = interpolate_pose(ts, tg, gx, gy, gyaw)
                if labeled is not None:
                    dense.add(query, ts, labeled)
                    if n_fl % BOOK_EVERY == 0:
                        coarse.add(query, ts, labeled)
        rows["front"].append((ts, front.x, front.y, front.yaw))
        rows["slide"].append((ts, slide.x, slide.y, slide.yaw))
        rows["dense"].append((ts, every.x, every.y, every.yaw))

    print(f"coarse pages {coarse.n}   dense pages {dense.n}")
    for label in ("front", "slide", "dense"):
        print(f"\n{label}")
        window_metrics("development", rows[label], tg, gx, gy, gyaw, t0, 0.0, DEV_END_S)
        window_metrics("validation", rows[label], tg, gx, gy, gyaw, t0, DEV_END_S, VAL_END_S)
        window_metrics("tail", rows[label], tg, gx, gy, gyaw, t0, VAL_END_S, 1e7)

    saved = [tuple(line.split(",")) for line in OUT_CSV.read_text().splitlines()[1:]]
    fresh = [(str(ts), f"{x:.4f}", f"{y:.4f}", f"{yaw:.6f}") for ts, x, y, yaw in rows["front"]]
    print("front rows match submission_retrieval.csv" if fresh == saved else "FRONT ROWS DIFFER")


if __name__ == "__main__":
    import time

    started = time.perf_counter()
    run()
    print(f"elapsed {time.perf_counter() - started:.1f} s")
