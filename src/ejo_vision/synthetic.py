"""Synthetic 2D paths for plotting and metric plumbing — not a camera simulator."""

from __future__ import annotations

import numpy as np


def make_synthetic_trajectory(n: int = 300, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    x_gt = 80.0 * t
    y_gt = 5.0 * np.sin(2.0 * np.pi * t) + 2.0 * np.sin(6.0 * np.pi * t)
    gt = np.column_stack([x_gt, y_gt])
    noise = np.cumsum(rng.normal(0.0, 0.12, size=(n, 2)), axis=0)
    est = gt + noise
    return gt, est
