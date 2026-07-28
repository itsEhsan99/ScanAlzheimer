"""Unit tests for volume loading, slicing, and intensity normalization.

All tests use small synthetic arrays -- no real MRI data required.
"""

import numpy as np
import pytest

from ScanAlzheimer.preprocessing.intensity import (
    clip_percentile,
    normalize_minmax,
    normalize_zscore,
    preprocess_volume,
)
from ScanAlzheimer.preprocessing.volume import (
    central_slice_indices,
    drop_singleton_dims,
    extract_central_slices,
    extract_slice,
)


@pytest.fixture
def volume():
    """A small deterministic 3D volume with distinguishable slices."""
    rng = np.random.default_rng(seed=42)
    return rng.random((8, 10, 12)).astype(np.float32)


# ----- volume -----


def test_drop_singleton_dims_removes_trailing_axis():
    arr = np.zeros((4, 5, 6, 1))
    assert drop_singleton_dims(arr).shape == (4, 5, 6)


def test_drop_singleton_dims_leaves_3d_untouched():
    arr = np.zeros((4, 5, 6))
    assert drop_singleton_dims(arr).shape == (4, 5, 6)


def test_drop_singleton_dims_preserves_real_4th_dimension():
    """A genuine 4D volume (e.g. multiple timepoints) must not be squeezed."""
    arr = np.zeros((4, 5, 6, 3))
    assert drop_singleton_dims(arr).shape == (4, 5, 6, 3)


@pytest.mark.parametrize("axis,expected", [(0, (10, 12)), (1, (8, 12)), (2, (8, 10))])
def test_extract_slice_shapes(volume, axis, expected):
    assert extract_slice(volume, axis, 0).shape == expected


def test_extract_slice_returns_correct_data(volume):
    np.testing.assert_array_equal(extract_slice(volume, 0, 3), volume[3])


def test_extract_slice_rejects_bad_axis(volume):
    with pytest.raises(ValueError, match="axis must be"):
        extract_slice(volume, 5, 0)


def test_extract_slice_rejects_out_of_range_index(volume):
    with pytest.raises(IndexError):
        extract_slice(volume, 0, 999)


def test_central_slice_indices_are_centred(volume):
    indices = central_slice_indices(volume, axis=2, n_slices=3)
    assert indices == [5, 6, 7]  # axis 2 has size 12, centre = 6


def test_central_slice_indices_single_slice_is_the_centre(volume):
    assert central_slice_indices(volume, axis=0, n_slices=1) == [4]


def test_central_slice_indices_respects_step(volume):
    assert central_slice_indices(volume, axis=2, n_slices=3, step=2) == [4, 6, 8]


def test_central_slice_indices_rejects_too_many_slices(volume):
    with pytest.raises(ValueError, match="outside axis"):
        central_slice_indices(volume, axis=0, n_slices=99)


def test_extract_central_slices_stacks_correctly(volume):
    stack = extract_central_slices(volume, axis=2, n_slices=3)
    assert stack.shape == (3, 8, 10)


# ----- intensity -----


def test_clip_percentile_bounds_extremes():
    arr = np.concatenate([np.ones(98), np.array([-1000.0, 1000.0])])
    clipped = clip_percentile(arr, 1.0, 99.0)
    assert clipped.max() < 1000.0
    assert clipped.min() > -1000.0


def test_clip_percentile_rejects_inverted_range():
    with pytest.raises(ValueError):
        clip_percentile(np.ones(10), lower=99.0, upper=1.0)


def test_normalize_minmax_maps_to_unit_range(volume):
    out = normalize_minmax(volume)
    assert np.isclose(out.min(), 0.0)
    assert np.isclose(out.max(), 1.0)


def test_normalize_minmax_handles_constant_volume():
    out = normalize_minmax(np.full((4, 4, 4), 7.0))
    assert np.all(out == 0.0)


def test_normalize_zscore_foreground_ignores_background():
    """Background zeros must not drag the mean, or contrast collapses."""
    vol = np.zeros((10, 10, 10), dtype=np.float32)
    vol[2:5, 2:5, 2:5] = np.arange(27, dtype=np.float32).reshape(3, 3, 3) + 1.0

    out = normalize_zscore(vol, foreground_only=True)
    foreground = out[vol > 0]
    assert np.isclose(foreground.mean(), 0.0, atol=1e-5)
    assert np.isclose(foreground.std(), 1.0, atol=1e-5)


def test_normalize_zscore_handles_empty_foreground():
    out = normalize_zscore(np.zeros((4, 4, 4)), foreground_only=True)
    assert np.all(out == 0.0)


def test_preprocess_volume_minmax_scheme(volume):
    out = preprocess_volume(volume, scheme="minmax")
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_preprocess_volume_rejects_unknown_scheme(volume):
    with pytest.raises(ValueError, match="Unknown normalization scheme"):
        preprocess_volume(volume, scheme="bogus")


def test_preprocessing_does_not_mutate_input(volume):
    """The critical purity guarantee: inputs are never modified in place."""
    original = volume.copy()
    preprocess_volume(volume)
    normalize_zscore(volume)
    clip_percentile(volume)
    extract_central_slices(volume, axis=2, n_slices=3)
    np.testing.assert_array_equal(volume, original)
