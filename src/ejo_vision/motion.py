"""Dead reckoning from allowed car-state signals.

Uses body velocity (vx forward, vy left) and yaw rate. Does not read position.
Map heading yaw is 0 along +x and increases toward +y.
"""

from __future__ import annotations

import numpy as np


def integrate_body_velocity(
    t_ns: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    wz: np.ndarray,
    x0: float,
    y0: float,
    yaw0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate one causal trajectory. Sample i uses rates from sample i-1."""
    t_ns = np.asarray(t_ns)
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    wz = np.asarray(wz, dtype=float)
    n = len(t_ns)
    if n == 0 or not (n == len(vx) == len(vy) == len(wz)):
        raise ValueError("t_ns, vx, vy, and wz must have the same non-zero length.")

    x = np.empty(n)
    y = np.empty(n)
    yaw = np.empty(n)
    x[0], y[0], yaw[0] = x0, y0, yaw0
    if n == 1:
        return x, y, yaw

    dt = np.diff(t_ns.astype(np.float64)) / 1e9
    yaw[1:] = yaw0 + np.cumsum(wz[:-1] * dt)
    c = np.cos(yaw[:-1])
    s = np.sin(yaw[:-1])
    x[1:] = x0 + np.cumsum((c * vx[:-1] - s * vy[:-1]) * dt)
    y[1:] = y0 + np.cumsum((s * vx[:-1] + c * vy[:-1]) * dt)
    return x, y, yaw
