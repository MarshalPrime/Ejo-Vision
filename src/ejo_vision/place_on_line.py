"""Place the car on a map line when nobody gives the starting pose.

Speed and yaw rate build a heading-versus-distance trace. That trace is
compared with the racing line and the pit line. A straight fits every part
of the lap, so it is refused. A turn is accepted only when one place fits
and the next place, at least 120 m away, fits much worse.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TURN_LOCK = np.deg2rad(45.0)
COST_LOCK = 0.01
RATIO_LOCK = 5.0
SEPARATION_M = 120.0
LOCAL_M = 40.0
TURN_LOCAL = np.deg2rad(40.0)
COST_LOCAL = 0.03
SEARCH_STEP_M = 2.0
AGREE_M = 30.0


class LineProfile:
    def __init__(self, name: str, line, step: float, heading: np.ndarray):
        self.name = name
        self.line = line
        self.step = step
        self.lap = float(line.lap)
        self.heading = np.asarray(heading, dtype=float)


def match_heading(
    heading: np.ndarray,
    step: float,
    lap: float,
    dist: np.ndarray,
    yaw: np.ndarray,
    origin: float | None = None,
    radius: float | None = None,
) -> tuple[float, float, float, float]:
    """Return place at the end of the trace, best cost, next cost, and turn."""
    dist = np.asarray(dist, dtype=float)
    yaw = np.asarray(yaw, dtype=float)
    sel = np.linspace(0, len(dist) - 1, min(60, len(dist))).astype(int)
    travelled = dist[sel] - dist[sel][0]
    unwrapped = np.unwrap(yaw[sel])
    turn = float(np.max(unwrapped) - np.min(unwrapped))
    turned = unwrapped - unwrapped[0]
    if origin is None:
        s0 = np.arange(0.0, lap, SEARCH_STEP_M)
    else:
        s0 = np.arange(origin - radius, origin + radius + SEARCH_STEP_M, SEARCH_STEP_M)
    pos = np.mod(s0[:, None] + travelled[None, :], lap)
    idx = np.floor(pos / step).astype(int) % len(heading)
    psi = heading[idx]
    dpsi = np.angle(np.exp(1j * (psi - psi[:, :1])))
    err = np.angle(np.exp(1j * (turned[None, :] - dpsi)))
    cost = np.mean(err**2, axis=1)
    j = int(np.argmin(cost))
    away = np.abs((s0 - s0[j] + lap / 2.0) % lap - lap / 2.0) > SEPARATION_M
    second = float(np.min(cost[away])) if np.any(away) else np.inf
    s_end = float((s0[j] + travelled[-1]) % lap)
    return s_end, float(cost[j]), second, turn


def lock_ok(cost: float, second: float, turn: float) -> bool:
    return turn > TURN_LOCK and cost < COST_LOCK and second > RATIO_LOCK * max(cost, 1e-12)


def load_profiles(map_mcap: str | Path) -> list[LineProfile]:
    map_mcap = Path(map_mcap)
    tools = str(map_mcap.resolve().parent / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from track_frame import load_lines

    profiles = []
    for name, line in load_lines(map_mcap).items():
        step = 2.0
        s_tab = np.arange(0.0, line.lap, step)
        _, _, heading = line.sn_to_xy(s_tab, np.zeros_like(s_tab), np.zeros_like(s_tab))
        profiles.append(LineProfile(name, line, step, heading))
    return profiles


def choose_lock(profiles: list[LineProfile], dist: np.ndarray, yaw: np.ndarray):
    """The unique line place, or None when the trace is still ambiguous."""
    chosen = None
    for profile in profiles:
        s_end, cost, second, turn = match_heading(profile.heading, profile.step, profile.lap, dist, yaw)
        if lock_ok(cost, second, turn) and (chosen is None or cost < chosen[2]):
            chosen = (profile, s_end, cost)
    return chosen


def best_hypothesis(profiles: list[LineProfile], dist: np.ndarray, yaw: np.ndarray):
    """Lowest-cost place, including an ambiguous one. Keeps a pose on every frame."""
    chosen = None
    for profile in profiles:
        s_end, cost, _second, _turn = match_heading(profile.heading, profile.step, profile.lap, dist, yaw)
        if chosen is None or cost < chosen[2]:
            chosen = (profile, s_end, cost)
    return chosen
