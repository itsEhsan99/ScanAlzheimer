"""Extract frozen-backbone embeddings for MRI slices.

With ~200 subjects, fine-tuning millions of parameters would overfit long
before it generalized. Instead the backbone is used purely as a fixed
feature extractor: one forward pass per slice, embeddings cached to disk,
and everything downstream is the same tabular pipeline the tissue baseline
already uses -- same splits, same leakage guard, same bootstrap metrics.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ScanAlzheimer.models.registry import BackboneSpec, embed_batch
from ScanAlzheimer.preprocessing.intensity import preprocess_volume
from ScanAlzheimer.preprocessing.slices import (
    apply_imagenet_normalization,
    build_subject_samples,
)
from ScanAlzheimer.preprocessing.volume import load_volume

EMBEDDING_PREFIX = "emb_"


def select_device() -> torch.device:
    """Use CUDA when it is genuinely available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def embed_volume(
    volume: np.ndarray,
    model: nn.Module,
    spec: BackboneSpec,
    axis: int = 1,
    n_slices: int = 5,
    step: int = 4,
    mode: str = "replicate",
    device: torch.device | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Embed every sample of one preprocessed volume.

    Returns (embeddings, positions) where embeddings has shape
    (n_slices, embedding_dim). Positions are slice indices for single-axis
    modes and midpoint offsets for triplanar, so any prediction can be
    traced back to an anatomical location.
    """
    device = device or select_device()

    samples, positions = build_subject_samples(
        volume, axis=axis, n_slices=n_slices, step=step, mode=mode
    )
    samples = apply_imagenet_normalization(samples)

    batch = torch.from_numpy(samples).to(device)
    embeddings = embed_batch(model, batch, spec.input_size)

    result = embeddings.detach().cpu().numpy()
    if result.shape[1] != spec.embedding_dim:
        raise RuntimeError(
            f"Backbone returned {result.shape[1]} dims, registry declares "
            f"{spec.embedding_dim} -- the registry contract is broken"
        )
    return result, positions


def embed_cohort(
    cohort: pd.DataFrame,
    model: nn.Module,
    spec: BackboneSpec,
    axis: int = 1,
    n_slices: int = 5,
    step: int = 4,
    mode: str = "replicate",
    device: torch.device | None = None,
    progress_every: int = 25,
) -> pd.DataFrame:
    """Embed every available subject into a slice-level frame.

    The result has one row per slice, carrying subject_id, label and fold
    forward unchanged. This is the point where each subject stops being a
    single row -- which is exactly what the subject-level fold assignment
    was built to survive.
    """
    device = device or select_device()
    model = model.to(device)
    model.eval()

    rows: list[dict] = []
    for counter, (_, subject) in enumerate(cohort.iterrows(), start=1):
        volume = preprocess_volume(load_volume(subject["image_path"]), scheme="minmax")
        embeddings, positions = embed_volume(
            volume, model, spec, axis, n_slices, step, mode, device
        )

        for vector, position in zip(embeddings, positions, strict=True):
            row = {
                "subject_id": subject["subject_id"],
                "label": int(subject["label"]),
                "fold": int(subject["fold"]),
                "age": subject["age"],
                "position": int(position),
            }
            row.update({f"{EMBEDDING_PREFIX}{i}": float(v) for i, v in enumerate(vector)})
            rows.append(row)

        if progress_every and counter % progress_every == 0:
            print(f"  embedded {counter} / {len(cohort)} subjects")

    return pd.DataFrame(rows)


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    """Embedding column names, in index order.

    Sorting numerically rather than lexicographically matters: plain string
    sorting would place emb_10 before emb_2 and silently scramble the
    feature order between runs.
    """
    columns = [c for c in frame.columns if c.startswith(EMBEDDING_PREFIX)]
    return sorted(columns, key=lambda c: int(c.removeprefix(EMBEDDING_PREFIX)))


def save_embeddings(frame: pd.DataFrame, path: str | Path) -> None:
    """Persist an embedding frame, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def load_embeddings(path: str | Path) -> pd.DataFrame:
    """Load a previously saved embedding frame."""
    return pd.read_parquet(path)
