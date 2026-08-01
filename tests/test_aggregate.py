"""Unit tests for slice-to-subject aggregation."""

import numpy as np
import pandas as pd
import pytest

from ScanAlzheimer.inference.aggregate import (
    aggregate_predictions,
    aggregate_scores,
    slice_agreement,
)


def _slice_frame(scores_by_subject, labels=None, folds=None):
    """Build a slice-level frame from {subject_id: [scores]}."""
    labels = labels or {}
    folds = folds or {}
    rows = []
    for subject_id, scores in scores_by_subject.items():
        for i, score in enumerate(scores):
            rows.append(
                {
                    "subject_id": subject_id,
                    "label": labels.get(subject_id, 0),
                    "fold": folds.get(subject_id, 0),
                    "position": i,
                    "y_score": score,
                }
            )
    return pd.DataFrame(rows)


# ----- score aggregation -----


def test_mean_aggregation():
    assert aggregate_scores(np.array([0.2, 0.4, 0.6]), "mean") == pytest.approx(0.4)


def test_median_resists_a_single_outlier():
    """One badly wrong slice must not dominate the subject's verdict."""
    scores = np.array([0.4, 0.45, 0.5, 0.99])
    assert aggregate_scores(scores, "median") < aggregate_scores(scores, "mean")


def test_max_aggregation_is_the_most_sensitive():
    scores = np.array([0.1, 0.2, 0.85])
    assert aggregate_scores(scores, "max") == pytest.approx(0.85)


def test_aggregate_scores_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        aggregate_scores(np.array([]))


def test_aggregate_scores_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown aggregation"):
        aggregate_scores(np.array([0.5]), "bogus")


# ----- frame aggregation -----


def test_aggregation_yields_one_row_per_subject():
    frame = _slice_frame({"S1": [0.1, 0.2, 0.3], "S2": [0.7, 0.8, 0.9]})
    result = aggregate_predictions(frame)
    assert len(result) == 2
    assert set(result["subject_id"]) == {"S1", "S2"}
    assert (result["n_slices"] == 3).all()


def test_aggregation_thresholds_correctly():
    frame = _slice_frame({"S1": [0.1, 0.2], "S2": [0.8, 0.9]})
    result = aggregate_predictions(frame).set_index("subject_id")
    assert result.loc["S1", "y_pred"] == 0
    assert result.loc["S2", "y_pred"] == 1


def test_aggregation_preserves_label_and_fold():
    frame = _slice_frame({"S1": [0.9, 0.9]}, labels={"S1": 1}, folds={"S1": 3})
    result = aggregate_predictions(frame)
    assert result["label"].item() == 1
    assert result["fold"].item() == 3


def test_aggregation_rejects_inconsistent_labels():
    """A subject with two different labels means the frame was built wrong."""
    frame = _slice_frame({"S1": [0.5, 0.5]})
    frame.loc[0, "label"] = 1
    with pytest.raises(ValueError, match="inconsistent 'label'"):
        aggregate_predictions(frame)


def test_aggregation_rejects_inconsistent_folds():
    frame = _slice_frame({"S1": [0.5, 0.5]})
    frame.loc[0, "fold"] = 1
    with pytest.raises(ValueError, match="inconsistent 'fold'"):
        aggregate_predictions(frame)


def test_aggregation_rejects_bad_threshold():
    frame = _slice_frame({"S1": [0.5]})
    with pytest.raises(ValueError, match="threshold"):
        aggregate_predictions(frame, threshold=1.5)


def test_aggregation_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        aggregate_predictions(pd.DataFrame({"subject_id": ["S1"]}))


def test_methods_can_disagree_on_the_same_subject():
    """Borderline cases are exactly where the aggregation choice matters."""
    frame = _slice_frame({"S1": [0.2, 0.3, 0.95]})
    mean_pred = aggregate_predictions(frame, method="mean")["y_pred"].item()
    max_pred = aggregate_predictions(frame, method="max")["y_pred"].item()
    assert mean_pred == 0
    assert max_pred == 1


def test_aggregation_does_not_mutate_input():
    frame = _slice_frame({"S1": [0.4, 0.6]})
    original = frame.copy()
    aggregate_predictions(frame)
    pd.testing.assert_frame_equal(frame, original)


# ----- agreement -----


def test_full_agreement_when_all_slices_concur():
    frame = _slice_frame({"S1": [0.9, 0.85, 0.95]})
    result = slice_agreement(frame)
    assert result["agreement"].item() == 1.0
    assert result["n_positive_slices"].item() == 3


def test_split_decision_lowers_agreement():
    frame = _slice_frame({"S1": [0.9, 0.9, 0.1, 0.1]})
    assert slice_agreement(frame)["agreement"].item() < 1.0


def test_agreement_reports_score_spread():
    tight = slice_agreement(_slice_frame({"S1": [0.5, 0.51, 0.49]}))
    wide = slice_agreement(_slice_frame({"S1": [0.1, 0.5, 0.9]}))
    assert tight["score_std"].item() < wide["score_std"].item()
