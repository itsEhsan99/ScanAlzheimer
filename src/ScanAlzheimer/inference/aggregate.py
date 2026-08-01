"""Aggregate slice-level predictions into one prediction per subject.

The model classifies slices, but the clinical claim is about people. How
those slice scores combine is a modelling decision, so the strategies live
here explicitly rather than being hard-coded into a training loop.

Metrics must always be computed after aggregation. Scoring at slice level
counts each subject several times, which artificially narrows confidence
intervals -- a milder version of the leakage this project guards against.
"""

import numpy as np
import pandas as pd

AGGREGATIONS = ("mean", "median", "max")
DEFAULT_AGGREGATION = "mean"
DEFAULT_THRESHOLD = 0.5


def aggregate_scores(scores: np.ndarray, method: str = DEFAULT_AGGREGATION) -> float:
    """Combine one subject's slice scores into a single score."""
    scores = np.asarray(scores).ravel()
    if scores.size == 0:
        raise ValueError("Cannot aggregate an empty score array")
    if method not in AGGREGATIONS:
        raise ValueError(f"Unknown aggregation {method!r}. Available: {list(AGGREGATIONS)}")

    if method == "mean":
        return float(np.mean(scores))
    if method == "median":
        return float(np.median(scores))
    return float(np.max(scores))


def aggregate_predictions(
    slice_frame: pd.DataFrame,
    method: str = DEFAULT_AGGREGATION,
    threshold: float = DEFAULT_THRESHOLD,
    score_column: str = "y_score",
    label_column: str = "label",
) -> pd.DataFrame:
    """Collapse slice-level predictions to one row per subject.

    Raises if a subject carries more than one label or fold, which would
    mean the slice frame was built incorrectly.
    """
    required = {"subject_id", score_column, label_column}
    missing = required - set(slice_frame.columns)
    if missing:
        raise ValueError(f"Slice frame is missing columns: {sorted(missing)}")

    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1); got {threshold}")

    for column in (label_column, "fold"):
        if column in slice_frame.columns:
            inconsistent = slice_frame.groupby("subject_id")[column].nunique()
            conflicting = inconsistent[inconsistent > 1].index.tolist()
            if conflicting:
                raise ValueError(f"Subjects have inconsistent {column!r}: {conflicting[:5]}")

    rows = []
    for subject_id, group in slice_frame.groupby("subject_id", sort=True):
        score = aggregate_scores(group[score_column].to_numpy(), method)
        row = {
            "subject_id": subject_id,
            "label": int(group[label_column].iloc[0]),
            "y_score": score,
            "y_pred": int(score >= threshold),
            "n_slices": len(group),
        }
        if "fold" in group.columns:
            row["fold"] = int(group["fold"].iloc[0])
        if "age" in group.columns:
            row["age"] = group["age"].iloc[0]
        rows.append(row)

    return pd.DataFrame(rows)


def slice_agreement(
    slice_frame: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
    score_column: str = "y_score",
) -> pd.DataFrame:
    """Per-subject agreement among slice-level decisions.

    Low agreement means the evidence is spread thin across slices, which is
    worth surfacing in the dashboard: a subject where 3 of 5 slices disagree
    deserves less confidence than one where all 5 agree.
    """
    rows = []
    for subject_id, group in slice_frame.groupby("subject_id", sort=True):
        votes = (group[score_column].to_numpy() >= threshold).astype(int)
        majority = int(round(votes.mean()))
        rows.append(
            {
                "subject_id": subject_id,
                "n_slices": len(votes),
                "n_positive_slices": int(votes.sum()),
                "agreement": float(np.mean(votes == majority)),
                "score_std": float(np.std(group[score_column].to_numpy())),
            }
        )
    return pd.DataFrame(rows)
