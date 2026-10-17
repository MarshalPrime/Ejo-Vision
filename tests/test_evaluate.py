import numpy as np
import pytest

from ejo_vision.evaluate import interpolate_pose, tracking_failure_rate, trajectory_rmse


def test_rmse_identical_paths_is_zero():
    path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    assert trajectory_rmse(path, path) == 0.0


def test_rmse_uses_shorter_prefix():
    gt = np.array([[0.0, 0.0], [3.0, 4.0]])
    est = np.array([[0.0, 0.0]])
    assert trajectory_rmse(gt, est) == 0.0


def test_rmse_rejects_empty():
    with pytest.raises(ValueError):
        trajectory_rmse(np.zeros((0, 2)), np.zeros((0, 2)))


def test_interpolate_pose_midpoint_and_gap():
    t_ns = np.array([0, 100], dtype=np.int64)
    pose = interpolate_pose(50, t_ns, np.array([0.0, 10.0]), np.array([0.0, 4.0]), np.array([0.0, 1.0]))
    assert pose == (5.0, 2.0, 0.5)
    assert interpolate_pose(50, t_ns, np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0]), max_gap_ns=10) is None


def test_failure_rate():
    diagnostics = [
        {"reason": "ok"},
        {"reason": "Not enough good matches"},
        {"reason": "ok"},
    ]
    assert tracking_failure_rate(diagnostics) == pytest.approx(1 / 3)
    assert tracking_failure_rate([]) == 0.0
