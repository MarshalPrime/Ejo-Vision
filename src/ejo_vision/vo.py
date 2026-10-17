"""Monocular ORB + essential-matrix visual odometry baseline.

This estimates rotation and a *unit* translation direction. It cannot recover
metric scale from camera images alone. `K` must be the real 3x3 intrinsics
for the camera that produced the images.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ejo_vision.geometry import compose_world_pose, yaw_from_R_world_cam


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Install OpenCV with: pip install opencv-python") from exc
    return cv2


def read_gray(path: str | Path):
    cv2 = _require_cv2()
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def estimate_relative_motion(img1, img2, K, nfeatures: int = 2500, orb=None):
    """Estimate relative R and translation direction between two monocular frames."""
    cv2 = _require_cv2()
    if orb is None:
        orb = cv2.ORB_create(nfeatures=nfeatures)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None, {"reason": "Not enough descriptors/keypoints", "matches": 0, "inliers": 0}

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(des1, des2, k=2)
    good = [
        m
        for pair in pairs
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < 0.75 * n.distance
    ]

    if len(good) < 8:
        return None, {"reason": "Not enough good matches", "matches": len(good), "inliers": 0}

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None:
        return None, {
            "reason": "Essential matrix estimation failed",
            "matches": len(good),
            "inliers": 0,
        }

    inliers, R, t, _pose_mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    inliers = int(inliers)
    if inliers < 8:
        return None, {
            "reason": "Too few recoverPose inliers",
            "matches": len(good),
            "inliers": inliers,
        }

    result = {"R": R, "t_direction": t.reshape(3), "inliers": inliers}
    info = {
        "reason": "ok",
        "matches": len(good),
        "inliers": inliers,
        "inlier_ratio": float(inliers / max(len(good), 1)),
    }
    return result, info


def run_vo(image_paths, K, max_frames=None, nfeatures: int = 2500):
    """Chain relative motions. Positions and headings have arbitrary scale/units."""
    cv2 = _require_cv2()
    paths = list(image_paths)
    if max_frames is not None:
        paths = paths[:max_frames]
    if len(paths) < 2:
        raise ValueError("Provide at least two sequential images.")

    orb = cv2.ORB_create(nfeatures=nfeatures)
    R_world_cam = np.eye(3)
    position = np.zeros(3)
    positions = [position.copy()]
    yaws = [yaw_from_R_world_cam(R_world_cam)]
    diagnostics = []

    previous = read_gray(paths[0])
    for idx, path in enumerate(paths[1:], start=1):
        current = read_gray(path)
        motion, info = estimate_relative_motion(previous, current, K, orb=orb)
        info["frame"] = idx
        diagnostics.append(info)

        if motion is not None:
            R_world_cam, position = compose_world_pose(
                R_world_cam, position, motion["R"], motion["t_direction"]
            )

        positions.append(position.copy())
        yaws.append(yaw_from_R_world_cam(R_world_cam))
        previous = current

    return {
        "positions": np.asarray(positions),
        "yaws": np.asarray(yaws),
        "diagnostics": diagnostics,
    }
