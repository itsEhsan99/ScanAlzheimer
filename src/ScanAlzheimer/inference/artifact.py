"""Save and load a trained model together with everything needed to use it.

A pickled estimator on its own is not deployable: whoever loads it must
also know which features it expects, in which order, how slices are
aggregated, and what performance was actually measured. Bundling that
metadata with the weights keeps the API layer from having to re-derive
any of it, and makes the reported numbers travel with the model rather
than living only in a README.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

ARTIFACT_VERSION = "1.0"
MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"


@dataclass
class ModelMetadata:
    """The contract between training and inference."""

    name: str
    feature_columns: list[str]
    feature_source: str
    task: str
    positive_class: str
    n_train_subjects: int
    metrics: dict[str, float]
    aggregation: str = "mean"
    threshold: float = 0.5
    backbone: str | None = None
    slice_config: dict[str, Any] = field(default_factory=dict)
    artifact_version: str = ARTIFACT_VERSION
    created_at: str = ""
    limitations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.feature_columns:
            raise ValueError("feature_columns must not be empty")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1); got {self.threshold}")
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_artifact(model: Any, metadata: ModelMetadata, directory: str | Path) -> Path:
    """Write model and metadata to a directory, creating it if needed."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, directory / MODEL_FILENAME)
    (directory / METADATA_FILENAME).write_text(
        json.dumps(asdict(metadata), indent=2), encoding="utf-8"
    )
    return directory


def load_artifact(directory: str | Path) -> tuple[Any, ModelMetadata]:
    """Load a model and its metadata, failing loudly if either is missing."""
    directory = Path(directory)

    model_path = directory / MODEL_FILENAME
    metadata_path = directory / METADATA_FILENAME
    for path in (model_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"Artifact is incomplete: {path} not found")

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    version = payload.get("artifact_version")
    if version != ARTIFACT_VERSION:
        raise ValueError(
            f"Artifact version {version!r} does not match expected {ARTIFACT_VERSION!r}"
        )

    return joblib.load(model_path), ModelMetadata(**payload)


def validate_features(metadata: ModelMetadata, provided: dict[str, float]) -> list[float]:
    """Order a feature dictionary to match what the model was trained on.

    Silently accepting a differently-ordered dictionary would produce
    confident, meaningless predictions, so missing keys raise instead.
    """
    missing = [c for c in metadata.feature_columns if c not in provided]
    if missing:
        raise ValueError(f"Missing required features: {missing[:10]}")
    return [float(provided[c]) for c in metadata.feature_columns]
