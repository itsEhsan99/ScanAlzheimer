"""Unsupervised tissue segmentation of skull-stripped T1 volumes.

The tissue features this project relies on were computed by OASIS with
FSL, which means an arbitrary uploaded scan cannot be scored at all. This
module closes that gap: a three-component Gaussian mixture over foreground
voxel intensities recovers CSF, grey matter and white matter without any
labels or external tools.

Component ordering is the crux. A mixture returns components in arbitrary
order, so they are relabelled by mean intensity: on T1, CSF is darkest,
grey matter intermediate, white matter brightest. That ordering is a
physical assumption about T1 contrast and would be wrong on T2 or FLAIR.

Agreement with the FSL segmentations is measured rather than assumed --
see scripts/validate_segmentation.py.
"""

import numpy as np
from sklearn.mixture import GaussianMixture

BACKGROUND = 0
CSF = 1
GREY_MATTER = 2
WHITE_MATTER = 3

N_TISSUES = 3


def foreground_mask(volume: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Boolean mask of brain voxels.

    Skull-stripped volumes have an exact zero background, so a threshold at
    zero is enough and avoids discarding genuinely dark CSF voxels.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume; got {volume.ndim}D")
    return volume > threshold


def fit_tissue_mixture(
    intensities: np.ndarray,
    seed: int = 42,
    max_samples: int = 50_000,
) -> GaussianMixture:
    """Fit a three-component Gaussian mixture to foreground intensities.

    Fitting on a random subsample rather than every voxel keeps this fast
    enough for an interactive request; a brain holds millions of voxels and
    the mixture converges on far fewer.
    """
    intensities = np.asarray(intensities).ravel()
    if intensities.size < N_TISSUES:
        raise ValueError(f"Need at least {N_TISSUES} voxels to fit; got {intensities.size}")

    if intensities.size > max_samples:
        rng = np.random.default_rng(seed)
        intensities = rng.choice(intensities, size=max_samples, replace=False)

    model = GaussianMixture(
        n_components=N_TISSUES,
        covariance_type="full",
        random_state=seed,
        max_iter=200,
    )
    model.fit(intensities.reshape(-1, 1))
    return model


def intensity_ordered_labels(model: GaussianMixture) -> np.ndarray:
    """Map mixture component indices to tissue labels by mean intensity.

    Returns an array where position i holds the tissue label for component
    i, so that darkest -> CSF, middle -> grey matter, brightest -> white.
    """
    order = np.argsort(model.means_.ravel())
    mapping = np.empty(N_TISSUES, dtype=np.int16)
    for rank, component in enumerate(order):
        mapping[component] = [CSF, GREY_MATTER, WHITE_MATTER][rank]
    return mapping


def segment_volume(
    volume: np.ndarray,
    seed: int = 42,
    max_samples: int = 50_000,
) -> np.ndarray:
    """Segment a skull-stripped T1 volume into background/CSF/GM/WM.

    Output uses the same label convention as the OASIS FSL segmentations,
    so downstream feature extraction is identical either way.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume; got {volume.ndim}D")

    mask = foreground_mask(volume)
    segmentation = np.zeros(volume.shape, dtype=np.int16)

    if not mask.any():
        return segmentation

    model = fit_tissue_mixture(volume[mask], seed, max_samples)
    mapping = intensity_ordered_labels(model)

    components = model.predict(volume[mask].reshape(-1, 1))
    segmentation[mask] = mapping[components]
    return segmentation


def dice_coefficient(a: np.ndarray, b: np.ndarray, label: int) -> float:
    """Dice overlap between two segmentations for one tissue label.

    Returns 1.0 when the label is absent from both, since perfect agreement
    on absence should not be scored as total disagreement.
    """
    mask_a = a == label
    mask_b = b == label

    total = int(mask_a.sum() + mask_b.sum())
    if total == 0:
        return 1.0
    return float(2 * np.count_nonzero(mask_a & mask_b) / total)


def segmentation_agreement(ours: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Per-tissue Dice against a reference segmentation."""
    if ours.shape != reference.shape:
        raise ValueError(f"Shape mismatch: {ours.shape} vs {reference.shape}")

    return {
        "dice_csf": dice_coefficient(ours, reference, CSF),
        "dice_grey_matter": dice_coefficient(ours, reference, GREY_MATTER),
        "dice_white_matter": dice_coefficient(ours, reference, WHITE_MATTER),
    }
