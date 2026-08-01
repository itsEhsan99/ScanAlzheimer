"""Regional tissue features from atlas-registered segmentation maps.

Whole-brain tissue fractions dilute the signal. Alzheimer's atrophy begins
in the medial temporal lobe and spreads outward, so a global average mixes
the most affected tissue with the least. Published ROC comparisons put
global grey-matter volume near AUC 0.67 for dementia versus controls, while
regional measures anchored on the hippocampus reach 0.83-0.88.

OASIS T88 volumes are registered to a shared atlas, so the same voxel
coordinates correspond to approximately the same anatomy in every subject.
A fixed spatial grid therefore acts as a coarse parcellation without any
per-subject anatomical segmentation: every block is measured in the same
place for everyone.

Which blocks carry the signal is deliberately left to the data rather than
asserted here, so the discriminative regions can be checked against known
pathology instead of assumed to match it.
"""

import numpy as np

from ScanAlzheimer.features.tissue import BRAIN_TISSUES, TISSUE_LABELS

LEFT_RIGHT_AXIS = 0  # sagittal axis in OASIS T88 volumes
LABEL_BY_NAME = {name: value for value, name in TISSUE_LABELS.items()}


def block_bounds(size: int, n_blocks: int) -> list[tuple[int, int]]:
    """Split an axis of `size` voxels into `n_blocks` contiguous ranges.

    Edges are spread as evenly as the size allows, so blocks differ by at
    most one voxel and no region is systematically over-weighted.
    """
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be >= 1; got {n_blocks}")
    if size < n_blocks:
        raise ValueError(f"Axis of size {size} cannot be split into {n_blocks} blocks")

    edges = np.linspace(0, size, n_blocks + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_blocks)]


def iter_blocks(shape: tuple[int, ...], n_blocks: int):
    """Yield ((i, j, k), (slice, slice, slice)) for every block in the grid."""
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D shape; got {shape}")

    bounds = [block_bounds(size, n_blocks) for size in shape]
    for i, (i0, i1) in enumerate(bounds[0]):
        for j, (j0, j1) in enumerate(bounds[1]):
            for k, (k0, k1) in enumerate(bounds[2]):
                yield (i, j, k), (slice(i0, i1), slice(j0, j1), slice(k0, k1))


def block_fractions(block: np.ndarray) -> dict[str, float]:
    """Tissue fractions within one block, relative to its own brain volume.

    Dividing by the block's own intracranial content rather than by its
    voxel count keeps the measure comparable between blocks that contain
    mostly brain and blocks that clip the edge of the head.
    """
    counts = {name: int(np.count_nonzero(block == LABEL_BY_NAME[name])) for name in BRAIN_TISSUES}
    total = sum(counts.values())
    if total == 0:
        return {name: 0.0 for name in BRAIN_TISSUES}
    return {name: counts[name] / total for name in BRAIN_TISSUES}


def regional_tissue_fractions(segmentation: np.ndarray, n_blocks: int = 4) -> dict[str, float]:
    """Tissue fractions for every block of an n x n x n grid."""
    if segmentation.ndim != 3:
        raise ValueError(f"Expected a 3D segmentation; got {segmentation.ndim}D")

    features: dict[str, float] = {}
    for (i, j, k), region in iter_blocks(segmentation.shape, n_blocks):
        for name, value in block_fractions(segmentation[region]).items():
            features[f"{name}_b{i}_{j}_{k}"] = value
    return features


def block_occupancy(segmentation: np.ndarray, n_blocks: int = 4) -> dict[str, float]:
    """Share of each block occupied by brain tissue of any kind.

    Complements the fractions: a block can keep a normal grey/white ratio
    while shrinking overall, and occupancy is what captures that.
    """
    features: dict[str, float] = {}
    for (i, j, k), region in iter_blocks(segmentation.shape, n_blocks):
        block = segmentation[region]
        brain = sum(int(np.count_nonzero(block == LABEL_BY_NAME[name])) for name in BRAIN_TISSUES)
        features[f"occupancy_b{i}_{j}_{k}"] = brain / block.size if block.size else 0.0
    return features


def asymmetry_features(regional: dict[str, float], n_blocks: int = 4) -> dict[str, float]:
    """Left-right differences between mirrored blocks.

    Hippocampal asymmetry is reported to grow as cognitive state declines,
    so the signed difference between mirrored blocks is a literature-backed
    feature rather than an arbitrary derived quantity. Only one half of each
    pair is emitted, since the other is its negation.
    """
    features: dict[str, float] = {}
    for i in range(n_blocks // 2):
        mirror = n_blocks - 1 - i
        for j in range(n_blocks):
            for k in range(n_blocks):
                for name in BRAIN_TISSUES:
                    left = regional.get(f"{name}_b{i}_{j}_{k}")
                    right = regional.get(f"{name}_b{mirror}_{j}_{k}")
                    if left is None or right is None:
                        continue
                    features[f"asym_{name}_b{i}_{j}_{k}"] = left - right
    return features


def extract_regional_features(
    segmentation: np.ndarray,
    n_blocks: int = 4,
    include_occupancy: bool = True,
    include_asymmetry: bool = True,
) -> dict[str, float]:
    """Full regional feature vector for one segmentation volume."""
    features = regional_tissue_fractions(segmentation, n_blocks)

    if include_asymmetry:
        features.update(asymmetry_features(features, n_blocks))
    if include_occupancy:
        features.update(block_occupancy(segmentation, n_blocks))

    return features


def parse_block_index(column: str) -> tuple[int, int, int] | None:
    """Recover the (i, j, k) grid position encoded in a feature name.

    Needed to map a model's coefficients back onto the brain, which is how
    we check whether the discriminative blocks land somewhere anatomically
    plausible rather than taking that on faith.
    """
    marker = "_b"
    if marker not in column:
        return None
    suffix = column.rsplit(marker, 1)[1]
    parts = suffix.split("_")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    return tuple(int(p) for p in parts)


def block_centre(
    shape: tuple[int, ...], n_blocks: int, index: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Voxel coordinates at the centre of a given block."""
    bounds = [block_bounds(size, n_blocks) for size in shape]
    return tuple(
        (bounds[axis][index[axis]][0] + bounds[axis][index[axis]][1]) // 2 for axis in range(3)
    )
