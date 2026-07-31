"""Tissue-volume features derived from FSL segmentation maps.

Alzheimer's atrophy shows up as reduced grey matter and expanded CSF
spaces, so simple tissue proportions carry real diagnostic signal. These
features form the honest baseline that any CNN in this project has to beat.

Note: the segmentation itself is provided by OASIS, not computed here.
"""

from pathlib import Path

import numpy as np

TISSUE_LABELS: dict[int, str] = {
    0: "background",
    1: "csf",
    2: "grey_matter",
    3: "white_matter",
}

BRAIN_TISSUES = ("csf", "grey_matter", "white_matter")


def tissue_voxel_counts(segmentation: np.ndarray) -> dict[str, int]:
    """Count voxels of each tissue class in a segmentation map.

    Unexpected label values are rejected rather than silently ignored --
    a segmentation with unknown labels means our assumptions are wrong.
    """
    present = set(np.unique(segmentation).tolist())
    unknown = present - set(TISSUE_LABELS)
    if unknown:
        raise ValueError(f"Segmentation contains unknown labels: {sorted(unknown)}")

    return {
        name: int(np.count_nonzero(segmentation == value)) for value, name in TISSUE_LABELS.items()
    }


def tissue_fractions(counts: dict[str, int]) -> dict[str, float]:
    """Convert raw voxel counts into fractions of total intracranial volume.

    Normalizing by intracranial volume (all non-background voxels) removes
    head-size differences, so the features reflect atrophy rather than how
    large someone's skull happens to be.
    """
    intracranial = sum(counts[t] for t in BRAIN_TISSUES)
    if intracranial == 0:
        return {f"{t}_fraction": 0.0 for t in BRAIN_TISSUES}

    return {f"{t}_fraction": counts[t] / intracranial for t in BRAIN_TISSUES}


def derived_ratios(counts: dict[str, int]) -> dict[str, float]:
    """Ratios that sharpen the atrophy signal.

    `gm_csf_ratio` falls as grey matter is lost and CSF expands, making it
    a more direct atrophy index than either quantity alone.
    """
    gm = counts["grey_matter"]
    wm = counts["white_matter"]
    csf = counts["csf"]

    return {
        "gm_csf_ratio": gm / csf if csf else 0.0,
        "gm_wm_ratio": gm / wm if wm else 0.0,
        "brain_csf_ratio": (gm + wm) / csf if csf else 0.0,
    }


def extract_tissue_features(segmentation: np.ndarray) -> dict[str, float]:
    """Full tissue feature vector for one segmentation volume."""
    counts = tissue_voxel_counts(segmentation)
    features: dict[str, float] = {}
    features.update({f"{k}_voxels": float(v) for k, v in counts.items()})
    features.update(tissue_fractions(counts))
    features.update(derived_ratios(counts))
    features["intracranial_voxels"] = float(sum(counts[t] for t in BRAIN_TISSUES))
    return features


def feature_names() -> list[str]:
    """Stable, ordered feature names -- the contract downstream models rely on."""
    names = [f"{TISSUE_LABELS[v]}_voxels" for v in sorted(TISSUE_LABELS)]
    names += [f"{t}_fraction" for t in BRAIN_TISSUES]
    names += ["gm_csf_ratio", "gm_wm_ratio", "brain_csf_ratio", "intracranial_voxels"]
    return names


def load_segmentation(path: str | Path) -> np.ndarray:
    """Load a segmentation map as an integer array."""
    from ScanAlzheimer.preprocessing.volume import load_volume

    return load_volume(path).astype(np.int16)
