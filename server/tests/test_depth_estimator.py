import numpy as np
from server.spatial.depth_estimator import DepthEstimator


def test_depth_to_metric():
    estimator = DepthEstimator()
    estimator.depth_scale = 100.0
    relative_depth = np.array([[0.5, 1.0], [0.25, 2.0]], dtype=np.float32)
    metric = estimator.to_metric(relative_depth)
    np.testing.assert_allclose(metric, [[200.0, 100.0], [400.0, 50.0]])


def test_calibrate_scale():
    estimator = DepthEstimator()
    relative_depth = np.ones((100, 100), dtype=np.float32) * 1.5
    estimator.calibrate_scale(relative_depth, pixel=(50, 50), real_distance_cm=75.0)
    assert abs(estimator.depth_scale - 112.5) < 0.01
