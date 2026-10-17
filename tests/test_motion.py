import numpy as np

from ejo_vision.motion import integrate_body_velocity


def test_straight_line_one_metre():
    t = np.array([0, 1_000_000_000], dtype=np.int64)
    x, y, yaw = integrate_body_velocity(t, [1.0, 1.0], [0.0, 0.0], [0.0, 0.0], 0.0, 0.0, 0.0)
    np.testing.assert_allclose(x[-1], 1.0)
    np.testing.assert_allclose(y[-1], 0.0)
    np.testing.assert_allclose(yaw[-1], 0.0)


def test_positive_yaw_rate_turns_left():
    # 90 degrees in 1 s, no translation. Heading should become +pi/2.
    t = np.array([0, 1_000_000_000], dtype=np.int64)
    x, y, yaw = integrate_body_velocity(
        t, [0.0, 0.0], [0.0, 0.0], [np.pi / 2, np.pi / 2], 0.0, 0.0, 0.0
    )
    np.testing.assert_allclose(yaw[-1], np.pi / 2)
    np.testing.assert_allclose([x[-1], y[-1]], [0.0, 0.0])
