"""Coordinate-frame helpers for monocular visual odometry.

OpenCV `recoverPose` returns R and t such that a 3D point maps as:

    X_cam2 = R @ X_cam1 + t

That is a world-to-camera style relative pose of camera 2 expressed in
camera 1. The camera-center displacement of camera 2, written in camera-1
coordinates, is therefore:

    C2_in_cam1 = -R.T @ t

We store world pose as:
- `R_world_cam`: maps camera coordinates into the first camera's frame
- `position`: the camera center in that same frame

Heading is the yaw of the optical axis (+Z) in the XZ ground plane, which
matches the usual optical-frame convention (X right, Y down, Z forward).
"""

from __future__ import annotations

import numpy as np

def camera_center_displacement(R_rel: np.ndarray, t_rel: np.ndarray) -> np.ndarray:
    """Return camera-2 center in camera-1 coordinates from recoverPose (R, t)."""
    R_rel = np.asarray(R_rel, dtype=float)
    t_rel = np.asarray(t_rel, dtype=float).reshape(3)
    return -R_rel.T @ t_rel


def compose_world_pose(
    R_world_cam: np.ndarray,
    position: np.ndarray,
    R_rel: np.ndarray,
    t_rel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Chain one recoverPose step onto the current world pose.

    Translation from a monocular pair is a *direction* (and here a unit
    vector). The resulting trajectory therefore has arbitrary scale.
    """
    R_world_cam = np.asarray(R_world_cam, dtype=float)
    position = np.asarray(position, dtype=float).reshape(3)
    delta_cam = camera_center_displacement(R_rel, t_rel)
    new_position = position + R_world_cam @ delta_cam
    new_R_world_cam = R_world_cam @ np.asarray(R_rel, dtype=float).T
    return new_R_world_cam, new_position


def yaw_from_R_world_cam(R_world_cam: np.ndarray) -> float:
    """Yaw of the camera optical axis in the XZ plane, in radians.

    0 means looking along +Z. Positive yaw rotates toward +X.
    """
    R_world_cam = np.asarray(R_world_cam, dtype=float)
    forward_world = R_world_cam @ np.array([0.0, 0.0, 1.0])
    return float(np.arctan2(forward_world[0], forward_world[2]))
