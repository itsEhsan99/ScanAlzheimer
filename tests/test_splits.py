"""Unit tests for subject-level splitting and the leakage guard.

The most important test here is the one that deliberately constructs a
leaky split and verifies the guard catches it -- a guard that has never
been shown to fire is not a guard.
"""

import pandas as pd
import pytest

from ScanAlzheimer.evaluation.splits import (
    assert_no_subject_leakage,
    assign_subject_folds,
    attach_folds,
    summarize_folds,
    train_test_masks,
)


def _make_manifest(n_per_class=20):
    """One row per subject, balanced classes, varied ages."""
    rows = []
    for i in range(n_per_class):
        rows.append(
            {"subject_id": f"OAS1_{i:04d}", "label": 0, "age": 65 + (i % 20), "raw_id": f"r{i}"}
        )
    for i in range(n_per_class):
        j = i + n_per_class
        rows.append(
            {"subject_id": f"OAS1_{j:04d}", "label": 1, "age": 70 + (i % 20), "raw_id": f"r{j}"}
        )
    return pd.DataFrame(rows)


def _expand_to_slices(manifest, n_slices=4):
    """Simulate the stage-4 situation: several rows per subject."""
    return manifest.loc[manifest.index.repeat(n_slices)].reset_index(drop=True)


# ----- fold assignment -----


def test_every_subject_gets_exactly_one_fold():
    folds = assign_subject_folds(_make_manifest(), n_splits=5)
    assert len(folds) == 40
    assert folds["subject_id"].nunique() == 40
    assert folds["fold"].nunique() == 5


def test_folds_are_deterministic_for_same_seed():
    manifest = _make_manifest()
    a = assign_subject_folds(manifest, n_splits=5, seed=7)
    b = assign_subject_folds(manifest, n_splits=5, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_folds_change_with_seed():
    manifest = _make_manifest(n_per_class=50)
    a = assign_subject_folds(manifest, n_splits=5, seed=1)
    b = assign_subject_folds(manifest, n_splits=5, seed=2)
    assert not a["fold"].equals(b["fold"])


def test_class_balance_roughly_preserved_across_folds():
    manifest = _make_manifest(n_per_class=50)
    folds = assign_subject_folds(manifest, n_splits=5)
    for _, group in folds.groupby("fold"):
        share = (group["label"] == 1).mean()
        assert 0.3 < share < 0.7


def test_rejects_too_few_subjects_for_requested_splits():
    manifest = _make_manifest(n_per_class=3)
    with pytest.raises(ValueError, match="smallest class"):
        assign_subject_folds(manifest, n_splits=5)


def test_rejects_subject_with_conflicting_labels():
    manifest = pd.DataFrame(
        {
            "subject_id": ["OAS1_0001", "OAS1_0001"],
            "label": [0, 1],
            "age": [70, 70],
        }
    )
    with pytest.raises(ValueError, match="conflicting labels"):
        assign_subject_folds(manifest, n_splits=2)


def test_rejects_invalid_n_splits():
    with pytest.raises(ValueError, match="n_splits"):
        assign_subject_folds(_make_manifest(), n_splits=1)


# ----- attaching folds to expanded rows -----


def test_attach_folds_preserves_row_count_when_expanded():
    manifest = _make_manifest()
    folds = assign_subject_folds(manifest, n_splits=5)
    slices = _expand_to_slices(manifest, n_slices=4)

    result = attach_folds(slices, folds)
    assert len(result) == len(slices) == 160


def test_attach_folds_keeps_all_slices_of_a_subject_together():
    """The whole point: expanding to slices must not split a brain."""
    manifest = _make_manifest()
    folds = assign_subject_folds(manifest, n_splits=5)
    slices = attach_folds(_expand_to_slices(manifest, n_slices=6), folds)

    assert_no_subject_leakage(slices)


def test_attach_folds_raises_for_unknown_subject():
    manifest = _make_manifest()
    folds = assign_subject_folds(manifest, n_splits=5)
    stray = pd.DataFrame({"subject_id": ["OAS1_9999"], "label": [1], "age": [80]})

    with pytest.raises(ValueError, match="No fold assigned"):
        attach_folds(stray, folds)


# ----- the leakage guard -----


def test_guard_passes_on_a_clean_split():
    manifest = _make_manifest()
    folds = assign_subject_folds(manifest, n_splits=5)
    clean = attach_folds(manifest, folds)
    assert_no_subject_leakage(clean)  # must not raise


def test_guard_catches_deliberately_leaked_split():
    """Construct the exact failure mode we are defending against: the same
    subject's slices scattered across different folds."""
    leaked = pd.DataFrame(
        {
            "subject_id": ["OAS1_0001", "OAS1_0001", "OAS1_0002", "OAS1_0002"],
            "label": [1, 1, 0, 0],
            "age": [75, 75, 68, 68],
            "fold": [0, 1, 0, 0],  # subject 0001 straddles two folds
        }
    )
    with pytest.raises(ValueError, match="leakage detected"):
        assert_no_subject_leakage(leaked)


def test_guard_reports_how_many_subjects_leaked():
    leaked = pd.DataFrame(
        {
            "subject_id": ["A", "A", "B", "B"],
            "label": [1, 1, 0, 0],
            "age": [70, 70, 70, 70],
            "fold": [0, 1, 0, 1],
        }
    )
    with pytest.raises(ValueError, match="2 subject"):
        assert_no_subject_leakage(leaked)


def test_guard_requires_fold_column():
    frame = pd.DataFrame({"subject_id": ["A"], "label": [1]})
    with pytest.raises(KeyError, match="fold"):
        assert_no_subject_leakage(frame)


# ----- masks and summary -----


def test_train_test_masks_are_complementary():
    manifest = _make_manifest()
    folds = assign_subject_folds(manifest, n_splits=5)
    frame = attach_folds(manifest, folds)

    train, test = train_test_masks(frame, fold=0)
    assert (train ^ test).all()
    assert test.sum() > 0


def test_train_test_masks_share_no_subjects():
    manifest = _make_manifest()
    folds = assign_subject_folds(manifest, n_splits=5)
    frame = attach_folds(_expand_to_slices(manifest), folds)

    train, test = train_test_masks(frame, fold=2)
    overlap = set(frame.loc[train, "subject_id"]) & set(frame.loc[test, "subject_id"])
    assert overlap == set()


def test_train_test_masks_reject_missing_fold():
    manifest = _make_manifest()
    folds = assign_subject_folds(manifest, n_splits=5)
    frame = attach_folds(manifest, folds)
    with pytest.raises(ValueError, match="not present"):
        train_test_masks(frame, fold=99)


def test_summarize_folds_covers_all_subjects():
    manifest = _make_manifest()
    folds = assign_subject_folds(manifest, n_splits=5)
    frame = attach_folds(manifest, folds)

    summary = summarize_folds(frame)
    assert len(summary) == 5
    assert summary["n_subjects"].sum() == 40
