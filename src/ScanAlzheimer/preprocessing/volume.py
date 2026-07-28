"""Load MRI volumes and extract 2D slices.

All functions here are pure: they never modify their inputs and never touch
anything outside their own scope, which keeps them trivially testable.
"""

from pathlib import Path

import nibabel as nib
import numpy as np

# Axis names for atlas-registered OASIS volumes. Verify visually before
# relying on these -- Analyze 7.5 carries limited orientation metadata.
AXIS_NAMES = {0: "sagittal", 1: "coronal", 2: "axial"}


def drop_singleton_dims(volume: np.ndarray) -> np.ndarray:
    """Remove trailing length-1 dimensions, e.g. (176, 208, 176, 1) -> 3D.

    Analyze 7.5 files frequently carry a vestigial 4th dimension.
    """
    while volume.ndim > 3 and volume.shape[-1] == 1:
        volume = volume[..., 0]
    return volume


def load_volume(path: str | Path) -> np.ndarray:
    """Load an MRI volume from disk as a 3D float32 array.

    Raises ValueError if the file is not 3D after squeezing singleton
    dimensions, so malformed inputs fail loudly and early.
    """
    img = nib.load(str(path))
    volume = np.asarray(img.dataobj, dtype=np.float32)
    volume = drop_singleton_dims(volume)

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape} from {path}")
    return volume


def extract_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Take a single 2D slice from a 3D volume along the given axis."""
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got {volume.ndim}D")
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2; got {axis}")

    n = volume.shape[axis]
    if not 0 <= index < n:
        raise IndexError(f"Slice index {index} out of range for axis {axis} of size {n}")

    return np.take(volume, index, axis=axis)


def central_slice_indices(volume: np.ndarray, axis: int, n_slices: int, step: int = 1) -> list[int]:
    """Return `n_slices` slice indices centred on the middle of `axis`.

    Using several slices around the centre gives the model more evidence per
    subject than a single slice, while staying far from the mostly-empty
    volume edges. `step` controls the spacing between consecutive slices.
    """
    if n_slices < 1:
        raise ValueError(f"n_slices must be >= 1; got {n_slices}")
    if step < 1:
        raise ValueError(f"step must be >= 1; got {step}")

    size = volume.shape[axis]
    centre = size // 2
    offset = (n_slices - 1) * step // 2
    indices = [centre - offset + i * step for i in range(n_slices)]

    out_of_range = [i for i in indices if not 0 <= i < size]
    if out_of_range:
        raise ValueError(
            f"Requested slices fall outside axis {axis} of size {size}: {out_of_range}"
        )
    return indices


def extract_central_slices(
    volume: np.ndarray, axis: int, n_slices: int, step: int = 1
) -> np.ndarray:
    """Stack `n_slices` central 2D slices into an array of shape (n_slices, H, W)."""
    indices = central_slice_indices(volume, axis, n_slices, step)
    return np.stack([extract_slice(volume, axis, i) for i in indices])
