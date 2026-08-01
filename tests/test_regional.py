"""Unit tests for regional grid features, using synthetic segmentations
with a known spatial layout."""

import numpy as np
import pytest

from ScanAlzheimer.features.regional import (
    asymmetry_features,
    block_bounds,
    block_centre,
    block_fractions,
    block_occupancy,
    extract_regional_features,
    iter_blocks,
    parse_block_index,
    regional_tissue_fractions,
)


def _uniform_segmentation(shape=(8, 8, 8), label=2):
    return np.full(shape, label, dtype=np.int16)


# ----- grid geometry -----


def test_block_bounds_cover_the_whole_axis():
    bounds = block_bounds(100, 4)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == 100
    assert all(bounds[i][1] == bounds[i + 1][0] for i in range(3))


def test_block_bounds_are_nearly_equal_for_uneven_sizes():
    sizes = [hi - lo for lo, hi in block_bounds(10, 3)]
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == 10


def test_block_bounds_reject_impossible_split():
    with pytest.raises(ValueError, match="cannot be split"):
        block_bounds(3, 4)


def test_iter_blocks_yields_the_full_grid():
    blocks = list(iter_blocks((8, 8, 8), 2))
    assert len(blocks) == 8
    assert {idx for idx, _ in blocks} == {
        (i, j, k) for i in range(2) for j in range(2) for k in range(2)
    }


def test_iter_blocks_partition_covers_every_voxel():
    volume = np.zeros((9, 10, 11), dtype=int)
    for _, region in iter_blocks(volume.shape, 3):
        volume[region] += 1
    assert (volume == 1).all()


# ----- block fractions -----


def test_block_fractions_of_pure_tissue():
    fractions = block_fractions(_uniform_segmentation(label=2))
    assert fractions["grey_matter"] == 1.0
    assert fractions["csf"] == 0.0


def test_block_fractions_ignore_background():
    block = np.array([0, 0, 0, 0, 2, 2, 1, 1], dtype=np.int16)
    fractions = block_fractions(block)
    assert fractions["grey_matter"] == 0.5
    assert fractions["csf"] == 0.5


def test_block_fractions_of_empty_block():
    fractions = block_fractions(np.zeros(10, dtype=np.int16))
    assert all(v == 0.0 for v in fractions.values())


# ----- regional extraction -----


def test_regional_features_count_matches_grid():
    seg = _uniform_segmentation((8, 8, 8))
    features = regional_tissue_fractions(seg, n_blocks=2)
    assert len(features) == 8 * 3  # 8 blocks, 3 tissues


def test_regional_features_localise_a_lesion():
    """A change confined to one corner must show up only in that block."""
    seg = _uniform_segmentation((8, 8, 8), label=2)
    seg[0:4, 0:4, 0:4] = 1  # corner becomes CSF

    features = regional_tissue_fractions(seg, n_blocks=2)
    assert features["csf_b0_0_0"] == 1.0
    assert features["csf_b1_1_1"] == 0.0
    assert features["grey_matter_b1_1_1"] == 1.0


def test_regional_features_reject_non_3d_input():
    with pytest.raises(ValueError, match="3D segmentation"):
        regional_tissue_fractions(np.zeros((4, 4)), n_blocks=2)


# ----- occupancy -----


def test_occupancy_is_one_for_a_fully_brain_block():
    features = block_occupancy(_uniform_segmentation((4, 4, 4)), n_blocks=2)
    assert all(v == 1.0 for v in features.values())


def test_occupancy_falls_when_a_block_is_partly_background():
    seg = _uniform_segmentation((8, 8, 8), label=3)
    seg[0:4, :, :] = 0
    features = block_occupancy(seg, n_blocks=2)
    assert features["occupancy_b0_0_0"] == 0.0
    assert features["occupancy_b1_0_0"] == 1.0


# ----- asymmetry -----


def test_asymmetry_is_zero_for_a_symmetric_brain():
    seg = _uniform_segmentation((8, 8, 8), label=2)
    regional = regional_tissue_fractions(seg, n_blocks=2)
    asym = asymmetry_features(regional, n_blocks=2)
    assert asym
    assert all(v == 0.0 for v in asym.values())


def test_asymmetry_detects_one_sided_atrophy():
    """Tissue loss confined to one hemisphere must produce a nonzero
    left-right difference -- the pattern reported to grow with decline."""
    seg = _uniform_segmentation((8, 8, 8), label=2)
    seg[0:4, :, :] = 1  # one side becomes CSF

    regional = regional_tissue_fractions(seg, n_blocks=2)
    asym = asymmetry_features(regional, n_blocks=2)
    assert asym["asym_grey_matter_b0_0_0"] == pytest.approx(-1.0)


def test_asymmetry_emits_only_one_half_of_each_pair():
    seg = _uniform_segmentation((8, 8, 8))
    regional = regional_tissue_fractions(seg, n_blocks=2)
    asym = asymmetry_features(regional, n_blocks=2)
    assert all("_b0_" in name for name in asym)


# ----- full extraction -----


def test_extract_includes_all_requested_families():
    seg = _uniform_segmentation((8, 8, 8))
    features = extract_regional_features(seg, n_blocks=2)
    assert any(k.startswith("grey_matter_b") for k in features)
    assert any(k.startswith("asym_") for k in features)
    assert any(k.startswith("occupancy_") for k in features)


def test_extract_can_omit_optional_families():
    seg = _uniform_segmentation((8, 8, 8))
    features = extract_regional_features(
        seg, n_blocks=2, include_occupancy=False, include_asymmetry=False
    )
    assert len(features) == 24
    assert not any(k.startswith("asym_") for k in features)


def test_extract_does_not_mutate_input():
    seg = _uniform_segmentation((8, 8, 8))
    original = seg.copy()
    extract_regional_features(seg, n_blocks=2)
    np.testing.assert_array_equal(seg, original)


def test_feature_count_grows_cubically():
    """Dimensionality matters with ~200 subjects, so the growth rate is
    worth pinning down in a test."""
    seg = _uniform_segmentation((12, 12, 12))
    small = extract_regional_features(seg, n_blocks=2, include_asymmetry=False)
    large = extract_regional_features(seg, n_blocks=4, include_asymmetry=False)
    assert len(large) == 8 * len(small)


# ----- mapping back to anatomy -----


def test_parse_block_index_recovers_coordinates():
    assert parse_block_index("grey_matter_b1_2_3") == (1, 2, 3)
    assert parse_block_index("asym_csf_b0_1_2") == (0, 1, 2)
    assert parse_block_index("occupancy_b3_3_3") == (3, 3, 3)


def test_parse_block_index_returns_none_for_global_features():
    assert parse_block_index("grey_matter_fraction") is None
    assert parse_block_index("age") is None


def test_block_centre_lands_inside_its_block():
    shape = (176, 208, 176)
    centre = block_centre(shape, 4, (0, 0, 0))
    assert all(0 <= c < s // 4 + 1 for c, s in zip(centre, shape, strict=True))

    far = block_centre(shape, 4, (3, 3, 3))
    assert all(c > s // 2 for c, s in zip(far, shape, strict=True))
