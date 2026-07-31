"""Unit tests for tissue-volume feature extraction, using synthetic
segmentation maps with known composition."""

import numpy as np
import pytest

from ScanAlzheimer.features.tissue import (
    derived_ratios,
    extract_tissue_features,
    feature_names,
    tissue_fractions,
    tissue_voxel_counts,
)


def _make_segmentation(n_csf=100, n_gm=300, n_wm=200, n_background=400):
    """Build a 1D segmentation with an exactly known tissue composition."""
    return np.concatenate(
        [
            np.zeros(n_background, dtype=np.int16),
            np.ones(n_csf, dtype=np.int16),
            np.full(n_gm, 2, dtype=np.int16),
            np.full(n_wm, 3, dtype=np.int16),
        ]
    )


def test_voxel_counts_are_exact():
    seg = _make_segmentation(n_csf=100, n_gm=300, n_wm=200, n_background=400)
    counts = tissue_voxel_counts(seg)
    assert counts == {
        "background": 400,
        "csf": 100,
        "grey_matter": 300,
        "white_matter": 200,
    }


def test_voxel_counts_reject_unknown_labels():
    seg = np.array([0, 1, 2, 3, 7], dtype=np.int16)
    with pytest.raises(ValueError, match="unknown labels"):
        tissue_voxel_counts(seg)


def test_fractions_sum_to_one():
    counts = tissue_voxel_counts(_make_segmentation())
    fractions = tissue_fractions(counts)
    assert np.isclose(sum(fractions.values()), 1.0)


def test_fractions_ignore_background():
    """Doubling the background must not change any tissue fraction."""
    a = tissue_fractions(tissue_voxel_counts(_make_segmentation(n_background=400)))
    b = tissue_fractions(tissue_voxel_counts(_make_segmentation(n_background=800)))
    assert a == b


def test_fractions_handle_empty_brain():
    counts = {"background": 100, "csf": 0, "grey_matter": 0, "white_matter": 0}
    fractions = tissue_fractions(counts)
    assert all(v == 0.0 for v in fractions.values())


def test_gm_csf_ratio_falls_with_atrophy():
    """The core biological claim: atrophy lowers the GM/CSF ratio."""
    healthy = tissue_voxel_counts(_make_segmentation(n_gm=400, n_csf=100))
    atrophic = tissue_voxel_counts(_make_segmentation(n_gm=250, n_csf=250))
    assert derived_ratios(healthy)["gm_csf_ratio"] > derived_ratios(atrophic)["gm_csf_ratio"]


def test_derived_ratios_handle_zero_denominator():
    counts = {"background": 10, "csf": 0, "grey_matter": 100, "white_matter": 0}
    ratios = derived_ratios(counts)
    assert ratios["gm_csf_ratio"] == 0.0
    assert ratios["gm_wm_ratio"] == 0.0


def test_extract_returns_all_declared_features():
    features = extract_tissue_features(_make_segmentation())
    assert set(features) == set(feature_names())


def test_feature_names_are_stable_and_unique():
    names = feature_names()
    assert len(names) == len(set(names))
    assert feature_names() == names


def test_extract_is_scale_invariant_for_fractions():
    """Same composition at different resolutions gives identical fractions."""
    small = extract_tissue_features(_make_segmentation(100, 300, 200, 400))
    large = extract_tissue_features(_make_segmentation(200, 600, 400, 800))
    for name in ("csf_fraction", "grey_matter_fraction", "white_matter_fraction"):
        assert np.isclose(small[name], large[name])
    assert large["intracranial_voxels"] == 2 * small["intracranial_voxels"]


def test_extract_does_not_mutate_input():
    seg = _make_segmentation()
    original = seg.copy()
    extract_tissue_features(seg)
    np.testing.assert_array_equal(seg, original)
