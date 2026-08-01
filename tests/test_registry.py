"""Unit tests for the backbone registry.

Backbones are built with `pretrained=False` so the tests need no network
access and stay fast -- we are testing our wiring, not the weights.
"""

import numpy as np
import pytest
import torch

from ScanAlzheimer.models.registry import (
    available_backbones,
    build_backbone,
    count_trainable_parameters,
    embed_batch,
    freeze,
    get_spec,
)


def test_registry_is_not_empty():
    assert "resnet18" in available_backbones()
    assert len(available_backbones()) >= 3


def test_get_spec_returns_declared_dimensions():
    spec = get_spec("resnet18")
    assert spec.embedding_dim == 512
    assert spec.input_size == 224


def test_get_spec_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown backbone"):
        get_spec("not_a_backbone")


def test_backbone_output_matches_declared_embedding_dim():
    """The registry's declared dim is a contract downstream code relies on."""
    spec = get_spec("resnet18")
    model = build_backbone("resnet18", pretrained=False)
    batch = torch.zeros(2, 3, spec.input_size, spec.input_size)

    with torch.no_grad():
        out = model(batch)

    assert out.shape == (2, spec.embedding_dim)


def test_freeze_removes_all_trainable_parameters():
    model = build_backbone("resnet18", pretrained=False)
    assert count_trainable_parameters(model) > 1_000_000

    freeze(model)
    assert count_trainable_parameters(model) == 0


def test_embed_batch_returns_one_vector_per_sample():
    model = freeze(build_backbone("resnet18", pretrained=False))
    batch = torch.zeros(4, 3, 224, 224)
    assert embed_batch(model, batch).shape == (4, 512)


def test_embed_batch_resizes_mismatched_input():
    """OASIS coronal slices are 176x176, not 224x224."""
    model = freeze(build_backbone("resnet18", pretrained=False))
    batch = torch.zeros(2, 3, 176, 176)
    assert embed_batch(model, batch, input_size=224).shape == (2, 512)


def test_embed_batch_rejects_wrong_channel_count():
    model = freeze(build_backbone("resnet18", pretrained=False))
    with pytest.raises(ValueError, match="Expected shape"):
        embed_batch(model, torch.zeros(2, 1, 224, 224))


def test_embeddings_differ_for_different_inputs():
    """A backbone returning constant output would silently destroy signal."""
    torch.manual_seed(0)
    model = freeze(build_backbone("resnet18", pretrained=False))
    rng = np.random.default_rng(0)
    batch = torch.tensor(rng.random((2, 3, 224, 224)), dtype=torch.float32)

    out = embed_batch(model, batch)
    assert not torch.allclose(out[0], out[1])
