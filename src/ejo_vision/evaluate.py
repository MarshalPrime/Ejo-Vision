"""Offline evaluation helpers.

These metrics assume the two trajectories already share a coordinate frame
and scale. For monocular VO you typically need an alignment step (for example
Umeyama / Sim(3)) before reporting metres. Do not treat unaligned RMSE as
metric localization accuracy.
"""

from __future__ import annotations

import numpy as np


def trajectory_rmse(gt: np.ndarray, est: np.ndarray) -> float:
    """Pointwise RMSE on the overlapping prefix of two (N, D) paths."""
    gt = np.asarray(gt, dtype=float)
    est = np.asarray(est, dtype=float)
    n = min(len(gt), len(est))
    if n == 0:
        raise ValueError("Both trajectories must contain at least one point.")
    return float(np.sqrt(np.mean(np.sum((gt[:n] - est[:n]) ** 2, axis=1))))


def interpolate_pose(
    timestamp_ns: int,
    t_ns: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    yaw: np.ndarray,
    max_gap_ns: int = 50_000_000,
) -> tuple[float, float, float] | None:
    """Pose at timestamp_ns. Returns None when the surrounding samples are too far apart."""
    i = int(np.searchsorted(t_ns, timestamp_ns))
    i = min(max(i, 1), len(t_ns) - 1)
    if int(t_ns[i]) - int(t_ns[i - 1]) > max_gap_ns:
        return None
    span = float(t_ns[i] - t_ns[i - 1])
    weight = (int(timestamp_ns) - int(t_ns[i - 1])) / span
    return (
        float(x[i - 1] + weight * (x[i] - x[i - 1])),
        float(y[i - 1] + weight * (y[i] - y[i - 1])),
        float(yaw[i - 1] + weight * (yaw[i] - yaw[i - 1])),
    )


def tracking_failure_rate(diagnostics: list[dict]) -> float:
    """Fraction of frames whose estimator did not return a usable motion."""
    if not diagnostics:
        return 0.0
    failures = sum(1 for item in diagnostics if item.get("reason") != "ok")
    return failures / len(diagnostics)
