"""Nested cross-validation for unbiased performance estimates.

Selecting a hyperparameter by looking at out-of-fold scores and then
reporting the best of those scores is optimistic: the evaluation folds
influenced the choice. No subject moved between train and test, but
information did.

Nested CV separates the two jobs. An inner loop, run only on the outer
training folds, picks the hyperparameter; the outer fold is touched once,
for scoring. The resulting estimate is usually lower than the selected
one -- that gap is the selection bias made visible.

The inner loop reuses the existing subject-level fold assignment rather
than resplitting, so the leakage guarantees carry through unchanged.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ScanAlzheimer.evaluation.metrics import balanced_accuracy
from ScanAlzheimer.evaluation.splits import FOLD_COLUMN, assert_no_subject_leakage
from ScanAlzheimer.inference.aggregate import aggregate_predictions

ModelBuilder = Callable[[dict], Any]
MetricFn = Callable[[np.ndarray, np.ndarray], float]


def _fit_predict(
    model: Any,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
) -> pd.DataFrame:
    """Fit on train, return test rows with scores attached."""
    x_train = train[feature_columns].to_numpy()
    y_train = train[label_column].to_numpy()

    if len(np.unique(y_train)) < 2:
        raise ValueError("Training split contains a single class")

    model.fit(x_train, y_train)
    scores = model.predict_proba(test[feature_columns].to_numpy())[:, 1]

    result = test.copy()
    result["y_score"] = scores
    return result


def select_hyperparameters(
    train_pool: pd.DataFrame,
    feature_columns: list[str],
    build_model: ModelBuilder,
    param_grid: list[dict],
    label_column: str = "label",
    metric_fn: MetricFn = balanced_accuracy,
    aggregation: str = "mean",
) -> tuple[dict, list[dict]]:
    """Pick the best parameters using only the outer training folds.

    Scores are computed after aggregating slices to subjects, matching how
    the model is ultimately judged: selecting on slice-level scores would
    optimise for the wrong unit.
    """
    if not param_grid:
        raise ValueError("param_grid must contain at least one configuration")

    inner_folds = sorted(train_pool[FOLD_COLUMN].unique())
    if len(inner_folds) < 2:
        raise ValueError("Need at least two folds in the training pool for inner CV")

    trace: list[dict] = []
    for params in param_grid:
        fold_scores: list[float] = []

        for inner_fold in inner_folds:
            inner_test = train_pool[train_pool[FOLD_COLUMN] == inner_fold]
            inner_train = train_pool[train_pool[FOLD_COLUMN] != inner_fold]

            predicted = _fit_predict(
                build_model(params), inner_train, inner_test, feature_columns, label_column
            )
            subjects = aggregate_predictions(predicted, method=aggregation)
            fold_scores.append(
                metric_fn(subjects["label"].to_numpy(), subjects["y_pred"].to_numpy())
            )

        trace.append({"params": params, "mean_score": float(np.mean(fold_scores))})

    best = max(trace, key=lambda entry: entry["mean_score"])
    return best["params"], trace


def run_nested_cross_validation(
    frame: pd.DataFrame,
    feature_columns: list[str],
    build_model: ModelBuilder,
    param_grid: list[dict],
    label_column: str = "label",
    metric_fn: MetricFn = balanced_accuracy,
    aggregation: str = "mean",
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, list[dict]]:
    """Nested CV producing out-of-fold predictions and the choices made.

    Returns (subject_predictions, selections), where `selections` records
    which parameters won in each outer fold. Disagreement between folds is
    itself informative: it means the choice is not well determined by this
    much data, and the reported number carries that instability.
    """
    assert_no_subject_leakage(frame)

    missing = set(feature_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Frame is missing feature columns: {sorted(missing)}")

    outer_folds = sorted(frame[FOLD_COLUMN].unique())
    if len(outer_folds) < 3:
        raise ValueError("Nested CV needs at least three folds")

    predictions: list[pd.DataFrame] = []
    selections: list[dict] = []

    for outer_fold in outer_folds:
        train_pool = frame[frame[FOLD_COLUMN] != outer_fold]
        outer_test = frame[frame[FOLD_COLUMN] == outer_fold]

        best_params, _ = select_hyperparameters(
            train_pool,
            feature_columns,
            build_model,
            param_grid,
            label_column,
            metric_fn,
            aggregation,
        )
        selections.append({"outer_fold": int(outer_fold), **best_params})

        predicted = _fit_predict(
            build_model(best_params), train_pool, outer_test, feature_columns, label_column
        )
        predictions.append(predicted)

    combined = pd.concat(predictions, ignore_index=True)
    return aggregate_predictions(combined, method=aggregation, threshold=threshold), selections


def selection_stability(selections: list[dict]) -> pd.DataFrame:
    """How often each configuration was chosen across outer folds.

    A single configuration winning every fold suggests a real optimum;
    a different winner each time means the data cannot resolve the choice.
    """
    frame = pd.DataFrame(selections).drop(columns=["outer_fold"])
    counts = frame.value_counts().reset_index(name="times_selected")
    counts["share"] = (counts["times_selected"] / len(selections)).round(2)
    return counts
