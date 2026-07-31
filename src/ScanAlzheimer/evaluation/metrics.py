"""Classification metrics with bootstrap confidence intervals.

A single point estimate from ~200 subjects is close to meaningless, so
every headline number in this project is reported with an interval. If
the interval for balanced accuracy includes 0.5, the model has not been
shown to beat chance -- and we say so rather than quoting the midpoint.
"""

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

CHANCE_BALANCED_ACCURACY = 0.5


def _validate(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on empty arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")

    allowed = {0, 1}
    if not set(np.unique(y_true)).issubset(allowed):
        raise ValueError("y_true must contain only 0 and 1")
    if not set(np.unique(y_pred)).issubset(allowed):
        raise ValueError("y_pred must contain only 0 and 1")

    return y_true, y_pred


def sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall for the positive (demented) class -- the share of true cases found.

    Returns NaN when no positive cases exist, since the quantity is undefined.
    """
    y_true, y_pred = _validate(y_true, y_pred)
    n_positive = int((y_true == 1).sum())
    if n_positive == 0:
        return float("nan")
    return float(((y_pred == 1) & (y_true == 1)).sum() / n_positive)


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall for the negative (cognitively normal) class."""
    y_true, y_pred = _validate(y_true, y_pred)
    n_negative = int((y_true == 0).sum())
    if n_negative == 0:
        return float("nan")
    return float(((y_pred == 0) & (y_true == 0)).sum() / n_negative)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean of sensitivity and specificity -- the headline metric here.

    Unlike plain accuracy, a model that predicts one class for everyone
    scores 0.5 no matter how the classes are distributed.
    """
    y_true, y_pred = _validate(y_true, y_pred)
    return float(balanced_accuracy_score(y_true, y_pred))


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve from continuous scores.

    Threshold-free, so it measures ranking quality rather than the effect
    of one arbitrary cutoff. Undefined (NaN) if only one class is present.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    if y_true.shape != y_score.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_score.shape}")
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    """Full metric set for one evaluation, as a flat dictionary."""
    y_true, y_pred = _validate(y_true, y_pred)

    report = {
        "n": float(len(y_true)),
        "n_positive": float((y_true == 1).sum()),
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "sensitivity": sensitivity(y_true, y_pred),
        "specificity": specificity(y_true, y_pred),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None:
        report["auc"] = auc(y_true, y_score)
    return report


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """2x2 confusion matrix as [[TN, FP], [FN, TP]]."""
    y_true, y_pred = _validate(y_true, y_pred)
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred_or_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval for any metric.

    Resamples subjects with replacement `n_boot` times and takes the
    empirical percentiles of the resulting metric distribution. Resamples
    that end up single-class (where the metric is undefined) are skipped.

    Returns (point_estimate, lower_bound, upper_bound).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred_or_score = np.asarray(y_pred_or_score).ravel()

    if y_true.shape != y_pred_or_score.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred_or_score.shape}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1; got {n_boot}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")

    point = metric_fn(y_true, y_pred_or_score)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    values: list[float] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        resampled_true = y_true[idx]
        if len(np.unique(resampled_true)) < 2:
            continue
        value = metric_fn(resampled_true, y_pred_or_score[idx])
        if not np.isnan(value):
            values.append(value)

    if not values:
        return point, float("nan"), float("nan")

    lower = float(np.percentile(values, 100 * alpha / 2))
    upper = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return point, lower, upper


def beats_chance(lower_bound: float, chance: float = CHANCE_BALANCED_ACCURACY) -> bool:
    """Whether a confidence interval excludes the chance level.

    The honest test for 'does this model work at all' -- a point estimate
    above chance means nothing if the interval straddles it.
    """
    if np.isnan(lower_bound):
        return False
    return bool(lower_bound > chance)


def format_ci(point: float, lower: float, upper: float, digits: int = 3) -> str:
    """Render a metric and its interval for printing or the README."""
    if np.isnan(point):
        return "n/a"
    if np.isnan(lower) or np.isnan(upper):
        return f"{point:.{digits}f}"
    return f"{point:.{digits}f} [{lower:.{digits}f}, {upper:.{digits}f}]"


def bootstrap_difference(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Paired bootstrap for the difference between two models (b minus a).

    Both models are resampled on the *same* subjects each iteration, which
    accounts for the correlation between them. Comparing two separate
    confidence intervals for overlap is not a valid test of a difference;
    this is.

    Returns (difference, lower_bound, upper_bound). If the interval
    excludes zero, the difference is supported by the data.
    """
    y_true = np.asarray(y_true).ravel()
    pred_a = np.asarray(pred_a).ravel()
    pred_b = np.asarray(pred_b).ravel()

    if not (y_true.shape == pred_a.shape == pred_b.shape):
        raise ValueError("All three arrays must have the same shape")

    point = metric_fn(y_true, pred_b) - metric_fn(y_true, pred_a)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs: list[float] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        resampled_true = y_true[idx]
        if len(np.unique(resampled_true)) < 2:
            continue
        value = metric_fn(resampled_true, pred_b[idx]) - metric_fn(resampled_true, pred_a[idx])
        if not np.isnan(value):
            diffs.append(value)

    if not diffs:
        return point, float("nan"), float("nan")

    return (
        point,
        float(np.percentile(diffs, 100 * alpha / 2)),
        float(np.percentile(diffs, 100 * (1 - alpha / 2))),
    )
