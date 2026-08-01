import numpy as np
import pytest

from ScanAlzheimer.preprocessing.slices import (
    build_subject_samples,
    build_triplanar_sample,
    pad_to_square,
    triplanar_offsets,
)


@pytest.fixture
def asymmetric_volume():
    """Volume with three different axis lengths, like real OASIS data."""
    return np.ones((176, 208, 176), dtype=np.float32)


def test_pad_to_square_centres_the_plane():
    plane = np.ones((4, 6), dtype=np.float32)
    padded = pad_to_square(plane, 8)
    assert padded.shape == (8, 8)
    assert padded.sum() == 24.0
    assert padded[0, 0] == 0.0


def test_pad_to_square_rejects_oversized_plane():
    with pytest.raises(ValueError, match="larger than target"):
        pad_to_square(np.ones((10, 10)), 8)


def test_triplanar_sample_stacks_three_planes(asymmetric_volume):
    """All three anatomical planes must end up the same square size."""
    sample = build_triplanar_sample(asymmetric_volume)
    assert sample.shape == (3, 208, 208)
    assert sample.dtype == np.float32


def test_triplanar_channels_are_different_views(asymmetric_volume):
    """Each channel comes from a different axis, so their footprints differ."""
    vol = asymmetric_volume.copy()
    vol[80:90, :, :] = 5.0  # a feature visible only in some views
    sample = build_triplanar_sample(vol)
    means = [float(sample[i].mean()) for i in range(3)]
    assert len({round(m, 4) for m in means}) > 1


def test_triplanar_offset_shifts_all_planes(asymmetric_volume):
    a = build_triplanar_sample(asymmetric_volume, offset=0)
    b = build_triplanar_sample(asymmetric_volume, offset=10)
    assert a.shape == b.shape


def test_triplanar_rejects_out_of_range_offset(asymmetric_volume):
    with pytest.raises(IndexError, match="out of range"):
        build_triplanar_sample(asymmetric_volume, offset=200)


def test_triplanar_offsets_are_symmetric():
    assert triplanar_offsets(3, 4) == [-4, 0, 4]
    assert triplanar_offsets(5, 2) == [-4, -2, 0, 2, 4]
    assert triplanar_offsets(1, 4) == [0]


def test_subject_samples_triplanar_mode(asymmetric_volume):
    samples, offsets = build_subject_samples(
        asymmetric_volume, n_slices=3, step=4, mode="triplanar"
    )
    assert samples.shape == (3, 3, 208, 208)
    assert offsets == [-4, 0, 4]


def test_subject_samples_rejects_unknown_mode(asymmetric_volume):
    with pytest.raises(ValueError, match="Unknown channel mode"):
        build_subject_samples(asymmetric_volume, mode="bogus")
