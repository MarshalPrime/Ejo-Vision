import numpy as np

from ejo_vision.geometry import (
    camera_center_displacement,
    compose_world_pose,
    yaw_from_R_world_cam,
)


def test_forward_motion_uses_camera_center_not_raw_t():
    # X2 = I @ X1 + t with t = [0, 0, -1] means camera 2 sits at +Z in cam1.
    R = np.eye(3)
    t = np.array([0.0, 0.0, -1.0])
    np.testing.assert_allclose(camera_center_displacement(R, t), [0.0, 0.0, 1.0])

    R_world, pos = compose_world_pose(np.eye(3), np.zeros(3), R, t)
    np.testing.assert_allclose(pos, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(R_world, np.eye(3))
    assert yaw_from_R_world_cam(R_world) == 0.0


def test_raw_t_is_not_the_camera_path():
    # Adding R @ t would send this example backward.
    t = np.array([0.0, 0.0, -1.0])
    notebook_style = np.eye(3) @ t
    correct = camera_center_displacement(np.eye(3), t)
    assert not np.allclose(notebook_style, correct)
    np.testing.assert_allclose(correct, -notebook_style)


def test_yaw_rotation_about_camera_y():
    # 90° about +Y (down): optical axis +Z rotates toward -X in a right-handed frame.
    theta = np.pi / 2
    c, s = np.cos(theta), np.sin(theta)
    R_rel = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    R_world, pos = compose_world_pose(np.eye(3), np.zeros(3), R_rel, np.zeros(3))
    np.testing.assert_allclose(pos, [0.0, 0.0, 0.0])
    yaw = yaw_from_R_world_cam(R_world)
    np.testing.assert_allclose(yaw, -np.pi / 2, atol=1e-12)
