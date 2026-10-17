from ejo_vision.evaluate import trajectory_rmse
from ejo_vision.synthetic import make_synthetic_trajectory


def test_synthetic_shapes_and_reproducibility():
    gt_a, est_a = make_synthetic_trajectory(n=50, seed=0)
    gt_b, est_b = make_synthetic_trajectory(n=50, seed=0)
    assert gt_a.shape == (50, 2)
    assert est_a.shape == (50, 2)
    assert (gt_a == gt_b).all()
    assert (est_a == est_b).all()
    assert trajectory_rmse(gt_a, est_a) > 0.0
