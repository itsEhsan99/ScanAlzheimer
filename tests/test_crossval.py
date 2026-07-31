"""Unit tests for the cross-validation loop and its guards.

Includes a negative control: a pipeline that scores well on shuffled
labels is broken, no matter how good its real numbers look.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ScanAlzheimer.evaluation.crossval import run_cross_validation, shuffle_labels
from ScanAlzheimer.evaluation.metrics import balanced_accuracy
from ScanAlzheimer.models.baseline import build_baseline_model, feature_columns


def _make_frame(n_per_class=50, separation=2.0, seed=0):
    """Synthetic cohort with a controllable class signal."""
    rng = np.random.default_rng(seed)
    rows = []
    for label in (0, 1):
        for i in range(n_per_class):
            idx = label * n_per_class + i
            rows.append(
                {
                    "subject_id": f"S{idx:04d}",
                    "label": label,
                    "fold": idx % 5,
                    "feat_a": rng.normal(label * separation, 1.0),
                    "feat_b": rng.normal(0.0, 1.0),
                }
            )
    return pd.DataFrame(rows)


def _factory():
    return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])


# ----- basic behaviour -----


def test_every_row_gets_a_prediction():
    frame = _make_frame()
    result = run_cross_validation(frame, ["feat_a", "feat_b"], _factory)
    assert result["y_pred"].notna().all()
    assert len(result) == len(frame)


def test_predictions_are_binary_and_scores_are_probabilities():
    result = run_cross_validation(_make_frame(), ["feat_a"], _factory)
    assert set(result["y_pred"].unique()).issubset({0, 1})
    assert result["y_score"].between(0.0, 1.0).all()


def test_separable_data_is_classified_well():
    frame = _make_frame(separation=4.0)
    result = run_cross_validation(frame, ["feat_a"], _factory)
    score = balanced_accuracy(result["label"], result["y_pred"])
    assert score > 0.9


def test_pure_noise_scores_near_chance():
    """No signal must mean no performance -- our floor sanity check."""
    frame = _make_frame(separation=0.0)
    result = run_cross_validation(frame, ["feat_a", "feat_b"], _factory)
    score = balanced_accuracy(result["label"], result["y_pred"])
    assert 0.3 < score < 0.7


def test_input_frame_is_not_mutated():
    frame = _make_frame()
    run_cross_validation(frame, ["feat_a"], _factory)
    assert "y_pred" not in frame.columns


# ----- guards -----


def test_rejects_missing_feature_columns():
    with pytest.raises(ValueError, match="missing feature columns"):
        run_cross_validation(_make_frame(), ["does_not_exist"], _factory)


def test_rejects_nan_features():
    frame = _make_frame()
    frame.loc[0, "feat_a"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        run_cross_validation(frame, ["feat_a"], _factory)


def test_rejects_leaked_split():
    """The leakage guard must fire before any model is fitted."""
    frame = _make_frame()
    frame.loc[0, "subject_id"] = frame.loc[1, "subject_id"]
    frame.loc[0, "fold"] = 0
    frame.loc[1, "fold"] = 1
    with pytest.raises(ValueError, match="leakage detected"):
        run_cross_validation(frame, ["feat_a"], _factory)


def test_rejects_single_class_training_fold():
    frame = _make_frame(n_per_class=10)
    frame["fold"] = (frame["label"] == 1).astype(int)
    with pytest.raises(ValueError, match="single class"):
        run_cross_validation(frame, ["feat_a"], _factory)


# ----- negative control -----


def test_shuffle_labels_preserves_class_counts():
    frame = _make_frame()
    shuffled = shuffle_labels(frame, seed=1)
    assert shuffled["label"].sum() == frame["label"].sum()
    assert len(shuffled) == len(frame)


def test_shuffle_labels_is_deterministic():
    frame = _make_frame()
    a = shuffle_labels(frame, seed=5)
    b = shuffle_labels(frame, seed=5)
    pd.testing.assert_frame_equal(a, b)


def test_shuffled_labels_score_near_chance():
    """The decisive negative control: destroy the label-feature relationship
    and the pipeline must collapse to chance. If it does not, something is
    leaking."""
    frame = _make_frame(separation=4.0)
    shuffled = shuffle_labels(frame, seed=2)
    result = run_cross_validation(shuffled, ["feat_a", "feat_b"], _factory)
    score = balanced_accuracy(result["label"], result["y_pred"])
    assert 0.3 < score < 0.7


# ----- baseline model -----


def test_baseline_model_is_a_pipeline_with_scaler():
    """Scaling must live inside the model, not be applied beforehand."""
    model = build_baseline_model()
    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["scaler"], StandardScaler)


def test_baseline_model_rejects_invalid_c():
    with pytest.raises(ValueError, match="must be positive"):
        build_baseline_model(c=0.0)


def test_feature_sets_are_defined():
    assert "age" in feature_columns("age_only")
    assert "grey_matter_fraction" in feature_columns("tissue_only")
    assert len(feature_columns("tissue_plus_age")) > len(feature_columns("tissue_only"))


def test_feature_columns_rejects_unknown_set():
    with pytest.raises(ValueError, match="Unknown feature set"):
        feature_columns("nonexistent")
