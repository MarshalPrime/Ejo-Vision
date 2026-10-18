"""Find the place along the lap from speed, turn rate, and the racing line.

The recent turn shape is matched only near the distance already travelled, so a
straight cannot throw the car to another part of the circuit. A front-camera
match may change the sideways offset only when it lies on that same place.

The picture book is built from the first 896 s only. The locked laps are scored
once, with these same settings, and are not used to change them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "23-00-38_cut_120m_update" / "tools"))

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
from ejo_vision.stream import iter_speed_and_front  # noqa: E402
from retrieve_localize import GT_CSV, TRAIN_MCAP, _gt_arrays, window_metrics  # noqa: E402
from track_frame import load_lines  # noqa: E402

WINDOW_S = 30.0
LOCAL_M = 40.0
TURN_MIN = np.deg2rad(40.0)
COST_MAX = 0.03
AGREE_M = 30.0
MAP_MCAP = ROOT / "23-00-38_cut_120m_update" / "map.mcap"


def _wrap_m(ds: float, lap: float) -> float:
    return (ds + lap / 2.0) % lap - lap / 2.0


class RacingLine:
    def __init__(self, line):
        self.line = line
        self.step = 1.0
        s_tab = np.arange(0.0, line.lap, self.step)
        _, _, heading = line.sn_to_xy(s_tab, np.zeros_like(s_tab), np.zeros_like(s_tab))
        self.heading = heading
        self.lap = line.lap

    def match(self, s_start: float, dist: np.ndarray, yaw: np.ndarray) -> tuple[float, float]:
        sel = np.linspace(0, len(dist) - 1, 40).astype(int)
        travelled = dist[sel] - dist[sel][0]
        turned = yaw[sel]
        s0 = np.arange(s_start - LOCAL_M, s_start + LOCAL_M + 1.0, 2.0)
        pos = np.mod(s0[:, None] + travelled[None, :], self.lap)
        idx = np.floor(pos / self.step).astype(int) % len(self.heading)
        psi = self.heading[idx]
        dpsi = np.angle(np.exp(1j * (psi - psi[:, :1])))
        dyaw = np.angle(np.exp(1j * (turned - turned[0])))
        err = np.angle(np.exp(1j * (dyaw[None, :] - dpsi)))
        cost = np.mean(err**2, axis=1)
        j = int(np.argmin(cost))
        return float((s0[j] + travelled[-1]) % self.lap), float(cost[j])


def run() -> None:
    tg, gx, gy, gyaw = _gt_arrays(GT_CSV)
    t0 = int(tg[0])
    dev_end = t0 + int(DEV_END_S * 1e9)
    track = RacingLine(load_lines(MAP_MCAP)["track"])
    book_desc, book_t, book_pose = [], [], []
    map_rows, photo_rows = [], []
    samples = []
    dist = 0.0
    yaw = None
    s_map = None
    s_photo = None
    last = None
    n_fl = 0
    relocks = 0
    photo_used = 0
    n_hold = 0.0
    epsi_hold = 0.0
    photo_until = 0

    for kind, ts, *payload in iter_speed_and_front(TRAIN_MCAP):
        if kind == "speed":
            vx, vy, wz = payload
            if yaw is None:
                pose = interpolate_pose(ts, tg, gx, gy, gyaw)
                s0, _, _, _ = track.line.xy_to_sn([pose[0]], [pose[1]])
                s_map = s_photo = float(s0[0])
                yaw = pose[2]
                last = (ts, vx, vy, wz)
                samples.append((ts, dist, yaw))
                continue
            dt = (ts - last[0]) / 1e9
            dist += float(np.hypot(last[1], last[2]) * dt)
            yaw += last[3] * dt
            step = float(np.hypot(last[1], last[2]) * dt)
            s_map = (s_map + step) % track.lap
            s_photo = (s_photo + step) % track.lap
            last = (ts, vx, vy, wz)
            if ts - samples[-1][0] >= 1e8:
                samples.append((ts, dist, yaw))
            continue

        if yaw is None:
            continue
        # Bring the along-track estimate up to this photo.
        dt = (ts - last[0]) / 1e9
        step = float(np.hypot(last[1], last[2]) * dt)
        dist_now = dist + step
        yaw_now = yaw + last[3] * dt
        s_map_now = (s_map + step) % track.lap
        s_photo_now = (s_photo + step) % track.lap
        window = [item for item in samples if ts - item[0] <= WINDOW_S * 1e9]
        if len(window) > 10 and window[-1][0] - window[0][0] >= 8e9:
            travelled = np.array([item[1] for item in window] + [dist_now])
            turned = np.array([item[2] for item in window] + [yaw_now])
            if np.max(turned) - np.min(turned) > TURN_MIN:
                s_hat, cost = track.match(
                    (s_map_now - (dist_now - travelled[0])) % track.lap, travelled, turned
                )
                if cost < COST_MAX:
                    s_map_now = s_hat
                    relocks += 1
                if ts >= photo_until:
                    s_hat, cost = track.match(
                        (s_photo_now - (dist_now - travelled[0])) % track.lap, travelled, turned
                    )
                    if cost < COST_MAX:
                        s_photo_now = s_hat
        x, y, map_yaw = track.line.sn_to_xy([s_map_now], [0.0], [0.0])
        pose_map = (float(x[0]), float(y[0]), float(map_yaw[0]))

        n_fl += 1
        if n_fl % BOOK_EVERY == 0:
            image = cv2.imdecode(np.frombuffer(payload[0], np.uint8), cv2.IMREAD_GRAYSCALE)
            query = thumbnail_descriptor(image)
            if query is not None and book_desc:
                index, sim = best_past_match(
                    np.stack(book_desc), np.asarray(book_t), query, ts, MIN_AGE_NS
                )
                if index is not None and sim >= SIM_MIN:
                    bx, by, byaw = book_pose[index]
                    s_b, n_b, epsi_b, _ = track.line.xy_to_sn([bx], [by], [byaw])
                    if abs(_wrap_m(float(s_b[0]) - s_photo_now, track.lap)) <= AGREE_M:
                        s_photo_now = float(s_b[0])
                        n_hold = float(n_b[0])
                        epsi_hold = float(epsi_b[0])
                        photo_until = ts + int(2e9)
                        photo_used += 1
            if ts <= dev_end and query is not None:
                labeled = interpolate_pose(ts, tg, gx, gy, gyaw)
                if labeled is not None:
                    book_desc.append(query)
                    book_t.append(ts)
                    book_pose.append(labeled)
        xn, yn, ynaw = track.line.sn_to_xy([s_photo_now], [n_hold], [epsi_hold])
        pose_out = (float(xn[0]), float(yn[0]), float(ynaw[0]))
        map_rows.append((ts, *pose_map))
        photo_rows.append((ts, *pose_out))
        dist, yaw, s_map, s_photo = dist_now, yaw_now, s_map_now, s_photo_now
        last = (ts, last[1], last[2], last[3])

    print(f"corner corrections {relocks}   photo nudges {photo_used}   book {len(book_desc)}")
    for label, rows in (("map only", map_rows), ("map, photo only when it agrees", photo_rows)):
        print(f"\n{label}")
        window_metrics("development", rows, tg, gx, gy, gyaw, t0, 0.0, DEV_END_S)
        window_metrics("validation", rows, tg, gx, gy, gyaw, t0, DEV_END_S, VAL_END_S)
        window_metrics("tail", rows, tg, gx, gy, gyaw, t0, VAL_END_S, 1e7)


if __name__ == "__main__":
    import time

    started = time.perf_counter()
    run()
    print(f"elapsed {time.perf_counter() - started:.1f} s")
