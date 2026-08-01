"""Unit tests for GMM-based tissue segmentation, using synthetic volumes
whose true composition is known by construction."""

import numpy as np
import pytest

from ScanAlzheimer.preprocessing.segmentation import (
    BACKGROUND,
    CSF,
    GREY_MATTER,
    WHITE_MATTER,
    dice_coefficient,
    fit_tissue_mixture,
    foreground_mask,
    intensity_ordered_labels,
    segment_volume,
    segmentation_agreement,
)


@pytest.fixture
def synthetic_brain():
    """Volume with three well-separated intensity clusters plus background.

    Returns (volume, ground_truth_segmentation).
    """
    rng = np.random.default_rng(0)
    volume = np.zeros((20, 20, 20), dtype=np.float32)
    truth = np.zeros((20, 20, 20), dtype=np.int16)

    regions = [
        (slice(2, 8), CSF, 0.15),
        (slice(8, 14), GREY_MATTER, 0.50),
        (slice(14, 18), WHITE_MATTER, 0.85),
    ]
    for region, label, centre in regions:
        shape = volume[region, 2:18, 2:18].shape
        volume[region, 2:18, 2:18] = rng.normal(centre, 0.02, shape).astype(np.float32)
        truth[region, 2:18, 2:18] = label

    volume = np.clip(volume, 0.01, 1.0) * (truth > 0)
    return volume, truth


# ----- foreground -----


def test_foreground_excludes_exact_zeros():
    volume = np.zeros((4, 4, 4), dtype=np.float32)
    volume[1:3, 1:3, 1:3] = 0.5
    assert foreground_mask(volume).sum() == 8


def test_foreground_rejects_non_3d():
    with pytest.raises(ValueError, match="3D volume"):
        foreground_mask(np.zeros((4, 4)))


# ----- component ordering -----


def test_components_are_relabelled_by_intensity():
    """The crux of the module: mixture component order is arbitrary, so
    darkest must map to CSF regardless of which index it received."""
    rng = np.random.default_rng(1)
    intensities = np.concatenate(
        [
            rng.normal(0.8, 0.02, 500),  # brightest first, on purpose
            rng.normal(0.1, 0.02, 500),
            rng.normal(0.5, 0.02, 500),
        ]
    )
    model = fit_tissue_mixture(intensities)
    mapping = intensity_ordered_labels(model)

    darkest = int(np.argmin(model.means_.ravel()))
    brightest = int(np.argmax(model.means_.ravel()))
    assert mapping[darkest] == CSF
    assert mapping[brightest] == WHITE_MATTER


def test_mapping_covers_all_three_tissues():
    rng = np.random.default_rng(2)
    intensities = np.concatenate([rng.normal(m, 0.02, 300) for m in (0.15, 0.5, 0.85)])
    mapping = intensity_ordered_labels(fit_tissue_mixture(intensities))
    assert set(mapping.tolist()) == {CSF, GREY_MATTER, WHITE_MATTER}


def test_fit_rejects_too_few_voxels():
    with pytest.raises(ValueError, match="at least 3 voxels"):
        fit_tissue_mixture(np.array([0.5]))


# ----- segmentation -----


def test_segmentation_recovers_known_tissues(synthetic_brain):
    volume, truth = synthetic_brain
    segmentation = segment_volume(volume)

    for label in (CSF, GREY_MATTER, WHITE_MATTER):
        assert dice_coefficient(segmentation, truth, label) > 0.9


def test_segmentation_preserves_background(synthetic_brain):
    volume, truth = synthetic_brain
    segmentation = segment_volume(volume)
    np.testing.assert_array_equal(segmentation == BACKGROUND, truth == BACKGROUND)


def test_segmentation_uses_only_valid_labels(synthetic_brain):
    volume, _ = synthetic_brain
    labels = set(np.unique(segment_volume(volume)).tolist())
    assert labels.issubset({BACKGROUND, CSF, GREY_MATTER, WHITE_MATTER})


def test_segmentation_is_deterministic(synthetic_brain):
    volume, _ = synthetic_brain
    np.testing.assert_array_equal(segment_volume(volume), segment_volume(volume))


def test_segmentation_handles_empty_volume():
    segmentation = segment_volume(np.zeros((10, 10, 10), dtype=np.float32))
    assert (segmentation == BACKGROUND).all()


def test_segmentation_does_not_mutate_input(synthetic_brain):
    volume, _ = synthetic_brain
    original = volume.copy()
    segment_volume(volume)
    np.testing.assert_array_equal(volume, original)


def test_segmentation_rejects_non_3d():
    with pytest.raises(ValueError, match="3D volume"):
        segment_volume(np.zeros((4, 4)))


# ----- dice -----


def test_dice_of_identical_masks_is_one():
    a = np.array([1, 1, 2, 2])
    assert dice_coefficient(a, a, 1) == 1.0


def test_dice_of_disjoint_masks_is_zero():
    a = np.array([1, 1, 0, 0])
    b = np.array([0, 0, 1, 1])
    assert dice_coefficient(a, b, 1) == 0.0


def test_dice_of_partial_overlap():
    a = np.array([1, 1, 1, 0])
    b = np.array([1, 1, 0, 0])
    assert dice_coefficient(a, b, 1) == pytest.approx(2 * 2 / 5)


def test_dice_is_one_when_label_absent_from_both():
    a = np.zeros(10, dtype=int)
    assert dice_coefficient(a, a, WHITE_MATTER) == 1.0


def test_agreement_reports_all_three_tissues(synthetic_brain):
    volume, truth = synthetic_brain
    scores = segmentation_agreement(segment_volume(volume), truth)
    assert set(scores) == {"dice_csf", "dice_grey_matter", "dice_white_matter"}


def test_agreement_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch"):
        segmentation_agreement(np.zeros((4, 4, 4)), np.zeros((5, 5, 5)))
