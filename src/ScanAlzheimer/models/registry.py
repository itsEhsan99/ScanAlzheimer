"""Config-driven registry of pretrained 2D backbones used as frozen feature
extractors.

Backbones are selected by name so that swapping one for another is a config
change, not a code change, and so that every candidate is evaluated through
exactly the same split and metric pipeline.

Domain-specific pretraining is not assumed to be better: ImageNet weights
are included as a first-class candidate, and which one wins on this cohort
is treated as an empirical question.
"""

from dataclasses import dataclass

import timm
import torch
from torch import nn


@dataclass(frozen=True)
class BackboneSpec:
    """Everything needed to build one backbone and interpret its output."""

    timm_name: str
    embedding_dim: int
    input_size: int
    description: str


BACKBONES: dict[str, BackboneSpec] = {
    "resnet18": BackboneSpec(
        timm_name="resnet18.a1_in1k",
        embedding_dim=512,
        input_size=224,
        description="ImageNet ResNet-18 -- small, fast, the reference point",
    ),
    "resnet50": BackboneSpec(
        timm_name="resnet50.a1_in1k",
        embedding_dim=2048,
        input_size=224,
        description="ImageNet ResNet-50 -- deeper, richer embedding",
    ),
    "efficientnet_b0": BackboneSpec(
        timm_name="efficientnet_b0.ra_in1k",
        embedding_dim=1280,
        input_size=224,
        description="ImageNet EfficientNet-B0 -- strong accuracy per FLOP",
    ),
    "convnext_tiny": BackboneSpec(
        timm_name="convnext_tiny.in12k_ft_in1k",
        embedding_dim=768,
        input_size=224,
        description="Modern conv architecture, ImageNet-12k pretraining",
    ),
    "densenet121": BackboneSpec(
        timm_name="densenet121.ra_in1k",
        embedding_dim=1024,
        input_size=224,
        description="Common baseline in medical imaging literature",
    ),
}

DEFAULT_BACKBONE = "resnet18"


def available_backbones() -> list[str]:
    """Names that can be passed to `build_backbone`."""
    return sorted(BACKBONES)


def get_spec(name: str) -> BackboneSpec:
    """Look up a backbone specification by name."""
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone {name!r}. Available: {available_backbones()}")
    return BACKBONES[name]


def build_backbone(name: str = DEFAULT_BACKBONE, pretrained: bool = True) -> nn.Module:
    """Build a backbone that outputs a pooled embedding vector.

    `num_classes=0` tells timm to drop the classification head, so the
    forward pass returns features rather than logits.
    """
    spec = get_spec(name)
    model = timm.create_model(spec.timm_name, pretrained=pretrained, num_classes=0)
    model.eval()
    return model


def freeze(model: nn.Module) -> nn.Module:
    """Disable gradients for every parameter.

    Used for linear probing: with only ~200 subjects, fine-tuning millions
    of parameters would overfit long before it generalized.
    """
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def count_trainable_parameters(model: nn.Module) -> int:
    """Number of parameters that would be updated by an optimizer."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def embed_batch(
    model: nn.Module, batch: torch.Tensor, input_size: int | None = None
) -> torch.Tensor:
    """Run a frozen backbone over one batch and return embeddings.

    Inputs are resized to the size the backbone was pretrained at. Returns
    a tensor of shape (n, embedding_dim).
    """
    if batch.ndim != 4 or batch.shape[1] != 3:
        raise ValueError(f"Expected shape (n, 3, H, W); got {tuple(batch.shape)}")

    if input_size is not None and batch.shape[-2:] != (input_size, input_size):
        batch = nn.functional.interpolate(
            batch, size=(input_size, input_size), mode="bilinear", align_corners=False
        )

    return model(batch)
