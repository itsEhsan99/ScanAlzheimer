"""Intensity normalization for MRI volumes.

MRI intensities have no absolute physical meaning -- the same tissue can map
to wildly different raw values across scanners and sessions. Normalizing is
therefore mandatory, not optional, and the choice of scheme is a modelling
decision we keep explicit and swappable.
"""

import numpy as np


def clip_percentile(volume: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    """Clip intensities to the given percentile range.

    Guards against a handful of extreme voxels (noise spikes, fat artefacts)
    dominating the subsequent rescaling.
    """
    if not 0.0 <= lower < upper <= 100.0:
        raise ValueError(f"Require 0 <= lower < upper <= 100; got {lower}, {upper}")

    lo, hi = np.percentile(volume, [lower, upper])
    return np.clip(volume, lo, hi)


def normalize_minmax(volume: np.ndarray) -> np.ndarray:
    """Rescale intensities to [0, 1]. Constant volumes map to all zeros."""
    lo = float(volume.min())
    hi = float(volume.max())
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume - lo) / (hi - lo)).astype(np.float32)


def normalize_zscore(volume: np.ndarray, foreground_only: bool = True) -> np.ndarray:
    """Standardize to zero mean and unit variance.

    With `foreground_only`, statistics are computed over non-zero voxels only.
    That matters for skull-stripped volumes, where the large black background
    would otherwise dominate the mean and shrink the effective contrast.
    """
    values = volume[volume > 0] if foreground_only else volume
    if values.size == 0:
        return np.zeros_like(volume, dtype=np.float32)

    mean = float(values.mean())
    std = float(values.std())
    if std == 0.0:
        return np.zeros_like(volume, dtype=np.float32)

    return ((volume - mean) / std).astype(np.float32)


def preprocess_volume(
    volume: np.ndarray,
    clip_lower: float = 1.0,
    clip_upper: float = 99.0,
    scheme: str = "minmax",
) -> np.ndarray:
    """Apply the standard intensity pipeline: percentile clipping, then rescaling.

    `scheme` selects the rescaling step and is intended to be driven from
    config, so alternatives can be compared without touching this code.
    """
    clipped = clip_percentile(volume, clip_lower, clip_upper)

    if scheme == "minmax":
        return normalize_minmax(clipped)
    if scheme == "zscore":
        return normalize_zscore(clipped)
    raise ValueError(f"Unknown normalization scheme {scheme!r}; expected 'minmax' or 'zscore'")
