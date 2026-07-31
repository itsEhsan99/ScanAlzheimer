"""Unit tests for classification metrics and bootstrap intervals.

Uses hand-constructed prediction vectors with known correct answers, so
every expected value can be verified by hand.
"""

import numpy as np
import pytest

from ScanAlzheimer.evaluation.metrics import (
    auc,
    balanced_accuracy,
    beats_chance,
    bootstrap_ci,
    classification_report,
    confusion,
    format_ci,
    sensitivity,
    specificity,
)

# ----- point metrics -----


def test_perfect_prediction_scores_one():
    y = np.array([0, 0, 1, 1])
    assert balanced_accuracy(y, y) == 1.0
    assert sensitivity(y, y) == 1.0
    assert specificity(y, y) == 1.0


def test_inverted_prediction_scores_zero():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    assert balanced_accuracy(y_true, y_pred) == 0.0


def test_balanced_accuracy_punishes_constant_prediction():
    """The core reason we use balanced accuracy: predicting all-positive on
    an imbalanced set must not look good."""
    y_true = np.array([1] * 90 + [0] * 10)
    y_pred = np.ones(100, dtype=int)

    assert (y_true == y_pred).mean() == 0.9
    assert balanced_accuracy(y_true, y_pred) == 0.5


def test_sensitivity_and_specificity_are_computed_correctly():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 1])
    assert np.isclose(sensitivity(y_true, y_pred), 2 / 3)
    assert np.isclose(specificity(y_true, y_pred), 2 / 3)


def test_sensitivity_is_nan_without_positives():
    y_true = np.zeros(5, dtype=int)
    assert np.isnan(sensitivity(y_true, y_true))


def test_metrics_reject_empty_input():
    with pytest.raises(ValueError, match="empty"):
        balanced_accuracy(np.array([]), np.array([]))


def test_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch"):
        balanced_accuracy(np.array([0, 1]), np.array([0, 1, 1]))


def test_metrics_reject_non_binary_labels():
    with pytest.raises(ValueError, match="only 0 and 1"):
        balanced_accuracy(np.array([0, 1, 2]), np.array([0, 1, 1]))


# ----- AUC -----


def test_auc_of_perfect_ranking_is_one():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert auc(y_true, y_score) == 1.0


def test_auc_of_random_ranking_is_near_half():
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 500 + [1] * 500)
    y_score = rng.random(1000)
    assert 0.42 < auc(y_true, y_score) < 0.58


def test_auc_is_nan_for_single_class():
    rng = np.random.default_rng(0)
    assert np.isnan(auc(np.ones(5, dtype=int), rng.random(5)))


# ----- confusion matrix -----


def test_confusion_matrix_layout():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    tn, fp, fn, tp = confusion(y_true, y_pred).ravel()
    assert (tn, fp, fn, tp) == (1, 1, 1, 1)


def test_confusion_always_two_by_two():
    """Even when a class is entirely absent from predictions."""
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    assert confusion(y_true, y_pred).shape == (2, 2)


# ----- report -----


def test_report_contains_expected_keys():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    report = classification_report(y_true, y_pred, y_score=np.array([0.2, 0.6, 0.7, 0.9]))
    expected = {
        "n",
        "n_positive",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "auc",
    }
    assert set(report) == expected


def test_report_omits_auc_without_scores():
    y = np.array([0, 1, 0, 1])
    assert "auc" not in classification_report(y, y)


# ----- bootstrap -----


def test_bootstrap_interval_brackets_point_estimate():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=200)
    y_pred = np.where(rng.random(200) < 0.75, y_true, 1 - y_true)

    point, lo, hi = bootstrap_ci(y_true, y_pred, balanced_accuracy, n_boot=500)
    assert lo <= point <= hi


def test_bootstrap_is_deterministic_for_same_seed():
    y_true = np.array([0, 1] * 50)
    y_pred = np.array([0, 1] * 40 + [1, 0] * 10)

    a = bootstrap_ci(y_true, y_pred, balanced_accuracy, n_boot=200, seed=3)
    b = bootstrap_ci(y_true, y_pred, balanced_accuracy, n_boot=200, seed=3)
    assert a == b


def test_bootstrap_interval_narrows_with_more_data():
    """More subjects must yield a tighter interval -- the whole point of
    reporting intervals at this sample size."""
    rng = np.random.default_rng(2)

    def width(n):
        y_true = rng.integers(0, 2, size=n)
        y_pred = np.where(rng.random(n) < 0.75, y_true, 1 - y_true)
        _, lo, hi = bootstrap_ci(y_true, y_pred, balanced_accuracy, n_boot=400)
        return hi - lo

    assert width(1000) < width(60)


def test_bootstrap_of_random_model_includes_chance():
    """A coin-flip classifier must not be declared better than chance."""
    rng = np.random.default_rng(4)
    y_true = rng.integers(0, 2, size=200)
    y_pred = rng.integers(0, 2, size=200)

    _, lo, hi = bootstrap_ci(y_true, y_pred, balanced_accuracy, n_boot=500)
    assert lo <= 0.5 <= hi
    assert not beats_chance(lo)


def test_bootstrap_of_strong_model_excludes_chance():
    rng = np.random.default_rng(5)
    y_true = rng.integers(0, 2, size=300)
    y_pred = np.where(rng.random(300) < 0.9, y_true, 1 - y_true)

    _, lo, _ = bootstrap_ci(y_true, y_pred, balanced_accuracy, n_boot=500)
    assert beats_chance(lo)


def test_bootstrap_works_with_auc():
    rng = np.random.default_rng(6)
    y_true = np.array([0] * 100 + [1] * 100)
    y_score = np.concatenate([rng.normal(0.3, 0.2, 100), rng.normal(0.7, 0.2, 100)])

    point, lo, hi = bootstrap_ci(y_true, y_score, auc, n_boot=300)
    assert lo <= point <= hi
    assert point > 0.8


def test_bootstrap_rejects_bad_arguments():
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_ci(y, y, balanced_accuracy, n_boot=0)
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_ci(y, y, balanced_accuracy, alpha=1.5)


# ----- formatting -----


def test_format_ci_renders_all_three_numbers():
    assert format_ci(0.723, 0.651, 0.798) == "0.723 [0.651, 0.798]"


def test_format_ci_handles_nan():
    assert format_ci(float("nan"), 0.1, 0.2) == "n/a"


def test_beats_chance_is_false_for_nan():
    assert not beats_chance(float("nan"))
