"""Turn a 3D volume into 3-channel 2D samples for 2D backbones.

ImageNet-pretrained models expect three input channels while an MRI slice
has one. Three strategies are provided so the choice can be tested rather
than assumed:

  replicate  -- the same slice in all three channels; the plain reference.
  adjacent   -- a slice and its two neighbours; a thin slab of 3D context,
                spatially registered the way pretrained channels expect.
  triplanar  -- sagittal, coronal and axial together; full 3D coverage at
                2D cost, but the channels are not spatially registered,
                which is unlike anything ImageNet weights were fitted on.

Which wins here is an empirical question, so all three go through the same
split and metric pipeline.
"""

import numpy as np

from ScanAlzheimer.preprocessing.volume import central_slice_indices, extract_slice

CHANNEL_MODES = ("replicate", "adjacent", "triplanar")
DEFAULT_AXIS = 1  # coronal -- where medial temporal atrophy is most visible
TRIPLANAR_AXES = (0, 1, 2)


def build_channel_indices(
    centre: int, mode: str, axis_size: int, neighbour_gap: int = 1
) -> list[int]:
    """Return the three slice indices for one single-axis sample.

    Neighbour indices are clamped to the volume bounds so samples near an
    edge degrade gracefully instead of raising. Not used by triplanar mode,
    which draws one slice from each axis instead.
    """
    if mode not in ("replicate", "adjacent"):
        raise ValueError(f"build_channel_indices does not handle mode {mode!r}")
    if not 0 <= centre < axis_size:
        raise IndexError(f"Centre index {centre} outside axis of size {axis_size}")

    if mode == "replicate":
        return [centre, centre, centre]

    below = max(0, centre - neighbour_gap)
    above = min(axis_size - 1, centre + neighbour_gap)
    return [below, centre, above]


def pad_to_square(plane: np.ndarray, size: int) -> np.ndarray:
    """Centre-pad a 2D plane with zeros to `size` x `size`.

    The three anatomical planes have different shapes, so they must be
    brought to a common size before stacking. Padding is used rather than
    resizing to avoid distorting anatomical proportions.
    """
    if plane.ndim != 2:
        raise ValueError(f"Expected a 2D plane; got shape {plane.shape}")
    height, width = plane.shape
    if height > size or width > size:
        raise ValueError(f"Plane {plane.shape} is larger than target size {size}")

    top = (size - height) // 2
    left = (size - width) // 2
    padded = np.zeros((size, size), dtype=plane.dtype)
    padded[top : top + height, left : left + width] = plane
    return padded


def build_sample(
    volume: np.ndarray,
    centre: int,
    axis: int = DEFAULT_AXIS,
    mode: str = "replicate",
    neighbour_gap: int = 1,
) -> np.ndarray:
    """Build one 3-channel sample of shape (3, H, W) from a single axis."""
    indices = build_channel_indices(centre, mode, volume.shape[axis], neighbour_gap)
    channels = [extract_slice(volume, axis, i) for i in indices]
    return np.stack(channels).astype(np.float32)


def build_triplanar_sample(volume: np.ndarray, offset: int = 0) -> np.ndarray:
    """Build one sample stacking sagittal, coronal and axial planes.

    `offset` shifts all three planes from their respective midpoints by the
    same number of voxels, which is how several samples per subject are
    obtained without leaving the informative central region.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume; got {volume.ndim}D")

    planes = []
    for axis in TRIPLANAR_AXES:
        centre = volume.shape[axis] // 2 + offset
        if not 0 <= centre < volume.shape[axis]:
            raise IndexError(
                f"Offset {offset} puts axis {axis} out of range (size {volume.shape[axis]})"
            )
        planes.append(extract_slice(volume, axis, centre))

    size = max(max(p.shape) for p in planes)
    return np.stack([pad_to_square(p, size) for p in planes]).astype(np.float32)


def triplanar_offsets(n_slices: int, step: int) -> list[int]:
    """Symmetric offsets around the midpoint, e.g. [-4, 0, 4] for n=3, step=4."""
    if n_slices < 1:
        raise ValueError(f"n_slices must be >= 1; got {n_slices}")
    if step < 1:
        raise ValueError(f"step must be >= 1; got {step}")

    half = (n_slices - 1) // 2
    return [(i - half) * step for i in range(n_slices)]


def build_subject_samples(
    volume: np.ndarray,
    axis: int = DEFAULT_AXIS,
    n_slices: int = 5,
    step: int = 4,
    mode: str = "replicate",
    neighbour_gap: int = 1,
) -> tuple[np.ndarray, list[int]]:
    """Build every sample for one subject.

    Returns (samples, positions) where samples has shape (n_slices, 3, H, W).
    For single-axis modes `positions` holds slice indices; for triplanar it
    holds offsets from the midpoint. Either way it lets a prediction be
    traced back to an anatomical location.
    """
    if mode not in CHANNEL_MODES:
        raise ValueError(f"Unknown channel mode {mode!r}. Available: {list(CHANNEL_MODES)}")

    if mode == "triplanar":
        offsets = triplanar_offsets(n_slices, step)
        samples = np.stack([build_triplanar_sample(volume, o) for o in offsets])
        return samples, offsets

    centres = central_slice_indices(volume, axis, n_slices, step)
    samples = np.stack([build_sample(volume, c, axis, mode, neighbour_gap) for c in centres])
    return samples, centres


def apply_imagenet_normalization(samples: np.ndarray) -> np.ndarray:
    """Standardize with ImageNet channel statistics.

    Pretrained weights were fitted to inputs on this scale, so matching it
    matters even though brain slices are nothing like natural images.
    Inputs are expected to already lie in [0, 1].
    """
    if samples.ndim != 4 or samples.shape[1] != 3:
        raise ValueError(f"Expected shape (n, 3, H, W); got {samples.shape}")

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    return ((samples - mean) / std).astype(np.float32)
