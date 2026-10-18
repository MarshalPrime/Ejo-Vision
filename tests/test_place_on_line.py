import numpy as np

from ejo_vision.place_on_line import lock_ok, match_heading


def _bent_heading():
    lap = 1000.0
    step = 1.0
    s = np.arange(0.0, lap, step)
    heading = np.zeros_like(s)
    bend = (s >= 400.0) & (s < 460.0)
    heading[bend] = np.linspace(0.0, np.pi / 2.0, int(np.sum(bend)))
    heading[s >= 460.0] = np.pi / 2.0
    return heading, step, lap


def test_a_real_turn_lands_on_that_turn():
    heading, step, lap = _bent_heading()
    dist = np.linspace(0.0, 80.0, 40)
    yaw = np.interp(400.0 + dist, np.arange(0.0, lap, step), heading)
    s_end, cost, second, turn = match_heading(heading, step, lap, dist, yaw)
    assert abs((s_end - 480.0 + lap / 2) % lap - lap / 2) < 4.0
    assert cost < 0.01
    assert lock_ok(cost, second, turn)


def test_a_straight_is_not_a_lock():
    heading = np.zeros(1000)
    dist = np.linspace(0.0, 100.0, 20)
    yaw = np.zeros_like(dist)
    _s_end, cost, second, turn = match_heading(heading, 1.0, 1000.0, dist, yaw)
    assert not lock_ok(cost, second, turn)
