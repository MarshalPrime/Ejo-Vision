"""See whether the other six cameras change the front-only result.

Each camera has its own picture book, built from the first 896 s only.
Three drives share the same speed and yaw rate:

- front: only the narrow front camera may reset the pose. This should match
  output/submission_retrieval.csv.
- any: any camera with a strong match may reset the pose.
- confirmed: the front match is kept only when another camera, within 100 ms,
  found a page within 15 m of that same place.

Nothing here overwrites the front-only pose file.
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
    VAL_END_S,
    best_past_match,
    thumbnail_descriptor,
)
from ejo_vision.stream import CAMERAS, iter_all_cameras  # noqa: E402
from retrieve_localize import GT_CSV, OUT_CSV, TRAIN_MCAP, _gt_arrays, window_metrics  # noqa: E402

AGREE_WINDOW_NS = 100_000_000
AGREE_RADIUS_M = 15.0


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


def _close(pose, other) -> bool:
    return (pose[0] - other[0]) ** 2 + (pose[1] - other[1]) ** 2 <= AGREE_RADIUS_M**2


def run() -> None:
    tg, gx, gy, gyaw = _gt_arrays(GT_CSV)
    t0 = int(tg[0])
    dev_end = t0 + int(DEV_END_S * 1e9)

    books = {
        name: {"desc": [], "t": [], "pose": []}
        for name in CAMERAS
    }
    pending: list = []
    front = any_drive = confirmed = None
    front_rows = []
    any_rows = []
    confirmed_rows = []
    resets = {name: 0 for name in CAMERAS}
    confirmed_resets = 0
    votes: list[tuple] = []
    pending_front = None

    for kind, *payload in iter_all_cameras(TRAIN_MCAP, BOOK_EVERY):
        if kind == "speed":
            ts, vx, vy, wz = payload
            if front is None:
                pose = interpolate_pose(ts, tg, gx, gy, gyaw)
                front = Drive(pose)
                any_drive = Drive(pose)
                confirmed = Drive(pose)
                front.start(ts, vx, vy, wz)
                any_drive.start(ts, vx, vy, wz)
                confirmed.start(ts, vx, vy, wz)
                continue
            pending.append((ts, vx, vy, wz))
            continue

        name, ts, jpeg = payload
        if front is None:
            continue

        proposal = None
        if jpeg is not None:
            image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
            query = thumbnail_descriptor(image)
            book = books[name]
            if query is not None and book["desc"]:
                index, sim = best_past_match(
                    np.stack(book["desc"]),
                    np.asarray(book["t"]),
                    query,
                    ts,
                    MIN_AGE_NS,
                )
                if index is not None and sim >= SIM_MIN:
                    proposal = book["pose"][index]
                    resets[name] += 1
            if ts <= dev_end and query is not None:
                labeled = interpolate_pose(ts, tg, gx, gy, gyaw)
                if labeled is not None:
                    book["desc"].append(query)
                    book["t"].append(ts)
                    book["pose"].append(labeled)

        if proposal is not None:
            any_drive.apply_until(pending, ts)
            any_drive.reset(proposal)
            votes.append((name, ts, proposal))
            votes = [vote for vote in votes if ts - vote[1] <= AGREE_WINDOW_NS]
            if name == "fl":
                pending_front = (ts, proposal)
                if any(vote[0] != "fl" and _close(proposal, vote[2]) for vote in votes):
                    confirmed.apply_until(pending, ts)
                    confirmed.reset(proposal)
                    confirmed_resets += 1
                    pending_front = None
            elif pending_front is not None and ts - pending_front[0] <= AGREE_WINDOW_NS and _close(
                proposal, pending_front[1]
            ):
                confirmed.apply_until(pending, ts)
                confirmed.reset(pending_front[1])
                confirmed_resets += 1
                pending_front = None

        if name == "fl":
            front.apply_until(pending, ts)
            any_drive.apply_until(pending, ts)
            confirmed.apply_until(pending, ts)
            if proposal is not None:
                front.reset(proposal)
            front_rows.append((ts, front.x, front.y, front.yaw))
            any_rows.append((ts, any_drive.x, any_drive.y, any_drive.yaw))
            confirmed_rows.append((ts, confirmed.x, confirmed.y, confirmed.yaw))

    print("book pages", {name: len(books[name]["t"]) for name in CAMERAS})
    print("strong matches", resets)
    print(f"front resets kept by another camera {confirmed_resets}")
    for label, rows in (
        ("front", front_rows),
        ("any camera", any_rows),
        ("front confirmed", confirmed_rows),
    ):
        print(f"\n{label}")
        window_metrics("development", rows, tg, gx, gy, gyaw, t0, 0.0, DEV_END_S)
        window_metrics("validation", rows, tg, gx, gy, gyaw, t0, DEV_END_S, VAL_END_S)
        window_metrics("tail", rows, tg, gx, gy, gyaw, t0, VAL_END_S, 1e7)

    saved = [
        tuple(line.split(","))
        for line in OUT_CSV.read_text().splitlines()[1:]
    ]
    fresh = [(str(ts), f"{x:.4f}", f"{y:.4f}", f"{yaw:.6f}") for ts, x, y, yaw in front_rows]
    print("front rows match submission_retrieval.csv" if fresh == saved else "FRONT ROWS DIFFER")


if __name__ == "__main__":
    import time

    started = time.perf_counter()
    run()
    print(f"elapsed {time.perf_counter() - started:.1f} s")
