import numpy as np

from hashbench import metrics


def test_perfect_separation():
    genuine = np.array([0.0, 0.05, 0.1])
    impostor = np.array([0.4, 0.5, 0.6])
    assert metrics.auc(genuine, impostor) == 1.0
    assert metrics.equal_error_rate(genuine, impostor)[0] == 0.0
    t = metrics.threshold_at_fpr(impostor, 0.0)
    assert 0.1 <= t < 0.4
    assert metrics.match_rate(genuine, t) == 1.0


def test_threshold_respects_fpr():
    impostor = np.linspace(0, 1, 101)
    t = metrics.threshold_at_fpr(impostor, 0.1)
    assert metrics.false_match_rate(impostor, t) <= 0.1
    # and it is the largest such threshold
    larger = impostor[impostor > t].min()
    assert metrics.false_match_rate(impostor, larger) > 0.1


def test_identical_distributions_auc_half():
    rng = np.random.default_rng(0)
    x = rng.random(2000)
    y = rng.random(2000)
    assert abs(metrics.auc(x, y) - 0.5) < 0.05


def test_wilson_interval():
    lo, hi = metrics.wilson_interval(50, 100)
    assert lo < 0.5 < hi and abs((lo + hi) / 2 - 0.5) < 1e-12
    assert abs(lo - 0.4038) < 1e-3                      # textbook value
    lo, hi = metrics.wilson_interval(100, 100)
    assert hi == 1.0 and 0.95 < lo < 1.0                # stays inside [0, 1]
    lo, hi = metrics.wilson_interval(10, 1000)
    assert lo > 0.0 and hi < 0.02
