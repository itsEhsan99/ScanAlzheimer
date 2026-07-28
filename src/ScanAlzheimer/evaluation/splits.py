"""Subject-level cross-validation splits and leakage guards.

Folds are always assigned at the *subject* level and then attached to
whatever row-level frame is being used (one row per session now, one row
per slice later). Assigning folds directly to expanded rows would place
slices from the same brain in both train and test, which is the single
most common source of inflated accuracy in this literature.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

FOLD_COLUMN = "fold"
GROUP_COLUMN = "subject_id"


def assign_subject_folds(
    manifest: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign each subject to exactly one cross-validation fold.

    Returns a lookup table with columns [subject_id, label, fold], one row
    per subject. Stratification keeps the class ratio comparable across
    folds; grouping keeps each subject confined to a single fold.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2; got {n_splits}")

    subjects = (
        manifest[[GROUP_COLUMN, "label"]]
        .drop_duplicates(subset=GROUP_COLUMN)
        .sort_values(GROUP_COLUMN)
        .reset_index(drop=True)
    )

    inconsistent = manifest.groupby(GROUP_COLUMN)["label"].nunique()
    conflicting = inconsistent[inconsistent > 1].index.tolist()
    if conflicting:
        raise ValueError(f"Subjects have conflicting labels: {conflicting}")

    counts = subjects["label"].value_counts()
    if counts.min() < n_splits:
        raise ValueError(
            f"Cannot make {n_splits} stratified folds: smallest class has "
            f"only {counts.min()} subjects"
        )

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.empty(len(subjects), dtype=int)

    for fold_index, (_, test_idx) in enumerate(
        splitter.split(subjects, subjects["label"], groups=subjects[GROUP_COLUMN])
    ):
        folds[test_idx] = fold_index

    subjects[FOLD_COLUMN] = folds
    return subjects


def attach_folds(frame: pd.DataFrame, subject_folds: pd.DataFrame) -> pd.DataFrame:
    """Attach subject-level fold assignments to a row-level frame.

    Works identically whether `frame` has one row per session or one row
    per slice -- the fold always follows the subject.
    """
    lookup = subject_folds[[GROUP_COLUMN, FOLD_COLUMN]]

    missing = set(frame[GROUP_COLUMN]) - set(lookup[GROUP_COLUMN])
    if missing:
        raise ValueError(f"No fold assigned for subjects: {sorted(missing)}")

    merged = frame.merge(lookup, on=GROUP_COLUMN, how="left", validate="many_to_one")
    if len(merged) != len(frame):
        raise RuntimeError("Row count changed during fold attachment")
    return merged


def assert_no_subject_leakage(frame: pd.DataFrame) -> None:
    """Raise if any subject appears in more than one fold.

    This is a runtime guard, not just a test helper: call it before every
    training run so a broken split fails loudly instead of quietly
    producing an impressive but meaningless score.
    """
    for column in (GROUP_COLUMN, FOLD_COLUMN):
        if column not in frame.columns:
            raise KeyError(f"Frame is missing required column {column!r}")

    fold_counts = frame.groupby(GROUP_COLUMN)[FOLD_COLUMN].nunique()
    leaked = fold_counts[fold_counts > 1]
    if not leaked.empty:
        raise ValueError(
            f"Subject-level leakage detected: {len(leaked)} subject(s) appear in "
            f"multiple folds, e.g. {leaked.index[:5].tolist()}"
        )


def train_test_masks(frame: pd.DataFrame, fold: int) -> tuple[pd.Series, pd.Series]:
    """Return boolean (train_mask, test_mask) for the given held-out fold."""
    if fold not in set(frame[FOLD_COLUMN]):
        raise ValueError(f"Fold {fold} not present in frame")

    test_mask = frame[FOLD_COLUMN] == fold
    return ~test_mask, test_mask


def summarize_folds(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-fold summary of size, class balance, and age -- used to confirm
    that folds are actually comparable to one another."""
    rows = []
    for fold, group in frame.groupby(FOLD_COLUMN):
        n_total = len(group)
        n_positive = int((group["label"] == 1).sum())
        rows.append(
            {
                "fold": int(fold),
                "n_subjects": group[GROUP_COLUMN].nunique(),
                "n_rows": n_total,
                "n_cn": int((group["label"] == 0).sum()),
                "n_demented": n_positive,
                "pct_demented": round(100 * n_positive / n_total, 1),
                "mean_age": round(float(group["age"].mean()), 1),
            }
        )
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
