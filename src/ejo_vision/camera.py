"""Map points into a camera image.

ROS tf convention: p_parent = R @ p_child + t.
Camera frames are optical: x right, y down, z forward.
"""

from __future__ import annotations

import numpy as np


def quat_to_R(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = float(np.sqrt(x * x + y * y + z * z + w * w))
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def parent_to_child(points_parent: np.ndarray, R_parent_child: np.ndarray, t_parent_child: np.ndarray) -> np.ndarray:
    """Invert p_parent = R @ p_child + t."""
    points_parent = np.asarray(points_parent, dtype=float)
    t = np.asarray(t_parent_child, dtype=float).reshape(3)
    return (np.asarray(R_parent_child, dtype=float).T @ (points_parent - t).T).T


def map_to_camera(
    points_map: np.ndarray,
    R_map_base: np.ndarray,
    t_map_base: np.ndarray,
    R_base_cam: np.ndarray,
    t_base_cam: np.ndarray,
) -> np.ndarray:
    """Border points in the map frame, expressed in the camera frame."""
    points_base = parent_to_child(points_map, R_map_base, t_map_base)
    return parent_to_child(points_base, R_base_cam, t_base_cam)
