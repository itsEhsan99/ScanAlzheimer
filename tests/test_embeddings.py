"""Unit tests for embedding extraction.

Backbones are built with `pretrained=False` so no network access is needed
and the tests stay fast -- what is under test is our wiring, not weights.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from ScanAlzheimer.features.embeddings import (
    embed_cohort,
    embed_volume,
    embedding_columns,
    load_embeddings,
    save_embeddings,
    select_device,
)
from ScanAlzheimer.models.registry import build_backbone, freeze, get_spec


@pytest.fixture(scope="module")
def model():
    return freeze(build_backbone("resnet18", pretrained=False))


@pytest.fixture(scope="module")
def spec():
    return get_spec("resnet18")


@pytest.fixture
def volume():
    rng = np.random.default_rng(0)
    return rng.random((60, 70, 60)).astype(np.float32)


# ----- device -----


def test_select_device_returns_a_torch_device():
    assert isinstance(select_device(), torch.device)


# ----- single volume -----


def test_embed_volume_returns_one_vector_per_slice(model, spec, volume):
    embeddings, positions = embed_volume(volume, model, spec, n_slices=3, step=2)
    assert embeddings.shape == (3, spec.embedding_dim)
    assert len(positions) == 3


def test_embed_volume_positions_are_slice_indices(model, spec, volume):
    _, positions = embed_volume(volume, model, spec, axis=1, n_slices=3, step=2)
    assert positions == [33, 35, 37]  # axis 1 size 70, centre 35


def test_embed_volume_triplanar_positions_are_offsets(model, spec, volume):
    _, positions = embed_volume(volume, model, spec, n_slices=3, step=4, mode="triplanar")
    assert positions == [-4, 0, 4]


def test_embed_volume_different_slices_give_different_embeddings(model, spec, volume):
    """A backbone collapsing every slice to the same vector would destroy
    all signal while still producing plausible-looking output."""
    embeddings, _ = embed_volume(volume, model, spec, n_slices=3, step=6)
    assert not np.allclose(embeddings[0], embeddings[2])


def test_embed_volume_is_deterministic(model, spec, volume):
    a, _ = embed_volume(volume, model, spec, n_slices=2, step=2)
    b, _ = embed_volume(volume, model, spec, n_slices=2, step=2)
    np.testing.assert_allclose(a, b, rtol=1e-5)


def test_embed_volume_does_not_mutate_input(model, spec, volume):
    original = volume.copy()
    embed_volume(volume, model, spec, n_slices=2, step=2)
    np.testing.assert_array_equal(volume, original)


# ----- column ordering -----


def test_embedding_columns_sort_numerically_not_lexically():
    """emb_10 must come after emb_2, or feature order silently scrambles."""
    frame = pd.DataFrame(
        {"subject_id": ["a"], "emb_0": [1.0], "emb_2": [2.0], "emb_10": [3.0], "emb_1": [4.0]}
    )
    assert embedding_columns(frame) == ["emb_0", "emb_1", "emb_2", "emb_10"]


def test_embedding_columns_ignores_metadata():
    frame = pd.DataFrame({"subject_id": ["a"], "label": [1], "emb_0": [1.0]})
    assert embedding_columns(frame) == ["emb_0"]


# ----- persistence -----


def test_save_and_load_roundtrip(tmp_path):
    frame = pd.DataFrame(
        {"subject_id": ["a", "b"], "label": [0, 1], "emb_0": [0.5, 1.5], "emb_1": [2.5, 3.5]}
    )
    path = tmp_path / "nested" / "embeddings.parquet"
    save_embeddings(frame, path)
    assert path.exists()
    pd.testing.assert_frame_equal(load_embeddings(path), frame)


# ----- cohort -----


def test_embed_cohort_expands_subjects_into_slices(tmp_path, model, spec, monkeypatch):
    """Each subject becomes several rows -- the moment the subject-level
    fold assignment starts doing real work."""
    rng = np.random.default_rng(1)
    fake_volume = rng.random((40, 50, 40)).astype(np.float32)
    monkeypatch.setattr("ScanAlzheimer.features.embeddings.load_volume", lambda _: fake_volume)

    cohort = pd.DataFrame(
        {
            "subject_id": ["S1", "S2"],
            "label": [0, 1],
            "fold": [0, 1],
            "age": [70, 80],
            "image_path": ["ignored", "ignored"],
        }
    )
    result = embed_cohort(cohort, model, spec, n_slices=3, step=2, progress_every=0)

    assert len(result) == 6
    assert result["subject_id"].nunique() == 2
    assert len(embedding_columns(result)) == spec.embedding_dim


def test_embed_cohort_keeps_fold_constant_within_subject(tmp_path, model, spec, monkeypatch):
    rng = np.random.default_rng(2)
    fake_volume = rng.random((40, 50, 40)).astype(np.float32)
    monkeypatch.setattr("ScanAlzheimer.features.embeddings.load_volume", lambda _: fake_volume)

    cohort = pd.DataFrame(
        {
            "subject_id": ["S1", "S2"],
            "label": [0, 1],
            "fold": [0, 1],
            "age": [70, 80],
            "image_path": ["ignored", "ignored"],
        }
    )
    result = embed_cohort(cohort, model, spec, n_slices=3, step=2, progress_every=0)
    assert (result.groupby("subject_id")["fold"].nunique() == 1).all()
