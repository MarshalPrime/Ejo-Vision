import numpy as np

from ejo_vision.camera import map_to_camera, parent_to_child, quat_to_R


def test_identity_quaternion():
    np.testing.assert_allclose(quat_to_R(0, 0, 0, 1), np.eye(3), atol=1e-12)


def test_parent_to_child_undoes_translation():
    points = np.array([[1.0, 2.0, 3.0]])
    child = parent_to_child(points, np.eye(3), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(child, [[0.0, 0.0, 0.0]])


def test_map_point_straight_ahead_of_camera():
    # Car at the map origin, camera 1 m ahead of base_link, point 10 m ahead in the map.
    # Optical frame is rotated: base x (forward) -> camera z, base y (left) -> camera -x, base z (up) -> camera -y.
    R_base_cam = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    p_cam = map_to_camera(
        np.array([[10.0, 0.0, 0.0]]),
        np.eye(3),
        np.zeros(3),
        R_base_cam,
        np.array([1.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(p_cam[0], [0.0, 0.0, 9.0], atol=1e-9)
