"""Subject-level cross-validation producing out-of-fold predictions.

Every subject is predicted exactly once, by a model that never saw them
during training. Metrics are then computed on the pooled out-of-fold
predictions, which at this sample size is more stable than averaging
per-fold scores and lets us bootstrap over subjects directly.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ScanAlzheimer.evaluation.splits import (
    FOLD_COLUMN,
    assert_no_subject_leakage,
    train_test_masks,
)

ModelFactory = Callable[[], Any]


def run_cross_validation(
    frame: pd.DataFrame,
    feature_columns: list[str],
    model_factory: ModelFactory,
    label_column: str = "label",
) -> pd.DataFrame:
    """Fit one model per fold and collect out-of-fold predictions.

    A fresh model is built for every fold, so no state carries over. The
    leakage guard runs first: if the split is broken, we stop rather than
    produce a number that looks fine and means nothing.
    """
    assert_no_subject_leakage(frame)

    missing = set(feature_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Frame is missing feature columns: {sorted(missing)}")

    if frame[feature_columns].isna().any().any():
        raise ValueError("Feature columns contain NaN values")

    result = frame.copy()
    result["y_pred"] = np.nan
    result["y_score"] = np.nan

    for fold in sorted(frame[FOLD_COLUMN].unique()):
        train_mask, test_mask = train_test_masks(frame, fold)

        x_train = frame.loc[train_mask, feature_columns].to_numpy()
        y_train = frame.loc[train_mask, label_column].to_numpy()
        x_test = frame.loc[test_mask, feature_columns].to_numpy()

        if len(np.unique(y_train)) < 2:
            raise ValueError(f"Training split for fold {fold} contains a single class")

        model = model_factory()
        model.fit(x_train, y_train)

        result.loc[test_mask, "y_score"] = model.predict_proba(x_test)[:, 1]
        result.loc[test_mask, "y_pred"] = model.predict(x_test)

    if result["y_pred"].isna().any():
        raise RuntimeError("Some rows received no out-of-fold prediction")

    result["y_pred"] = result["y_pred"].astype(int)
    return result


def shuffle_labels(frame: pd.DataFrame, seed: int = 0, label_column: str = "label") -> pd.DataFrame:
    """Return a copy with labels randomly permuted across subjects.

    Used as a negative control: an honest pipeline scores about chance on
    shuffled labels. Published work has shown that slice-level splitting
    reaches ~96% accuracy on randomly labelled brain MRI, so this check is
    the difference between a result and an artefact.
    """
    rng = np.random.default_rng(seed)
    shuffled = frame.copy()
    shuffled[label_column] = rng.permutation(frame[label_column].to_numpy())
    return shuffled
