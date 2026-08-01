"""Unit tests for model artifact persistence."""

import json

import pytest
from sklearn.linear_model import LogisticRegression

from ScanAlzheimer.inference.artifact import (
    METADATA_FILENAME,
    MODEL_FILENAME,
    ModelMetadata,
    load_artifact,
    save_artifact,
    validate_features,
)


def _metadata(**overrides):
    defaults = {
        "name": "tissue_plus_age",
        "feature_columns": ["grey_matter_fraction", "csf_fraction", "age"],
        "feature_source": "fsl_seg",
        "task": "CDR 0 vs CDR >= 0.5",
        "positive_class": "demented",
        "n_train_subjects": 198,
        "metrics": {"balanced_accuracy": 0.687, "auc": 0.748},
    }
    return ModelMetadata(**{**defaults, **overrides})


def _model():
    model = LogisticRegression()
    model.fit([[0.4, 0.2, 70], [0.3, 0.3, 80]], [0, 1])
    return model


def test_metadata_sets_creation_time():
    assert _metadata().created_at


def test_metadata_rejects_empty_features():
    with pytest.raises(ValueError, match="must not be empty"):
        _metadata(feature_columns=[])


def test_metadata_rejects_impossible_threshold():
    with pytest.raises(ValueError, match="threshold"):
        _metadata(threshold=1.5)


def test_save_creates_both_files(tmp_path):
    directory = save_artifact(_model(), _metadata(), tmp_path / "artifact")
    assert (directory / MODEL_FILENAME).exists()
    assert (directory / METADATA_FILENAME).exists()


def test_roundtrip_preserves_metadata(tmp_path):
    original = _metadata()
    save_artifact(_model(), original, tmp_path / "a")
    _, loaded = load_artifact(tmp_path / "a")

    assert loaded.feature_columns == original.feature_columns
    assert loaded.metrics == original.metrics
    assert loaded.task == original.task


def test_loaded_model_predicts(tmp_path):
    save_artifact(_model(), _metadata(), tmp_path / "a")
    model, _ = load_artifact(tmp_path / "a")
    assert model.predict_proba([[0.4, 0.2, 70]]).shape == (1, 2)


def test_load_rejects_incomplete_artifact(tmp_path):
    directory = tmp_path / "a"
    save_artifact(_model(), _metadata(), directory)
    (directory / METADATA_FILENAME).unlink()

    with pytest.raises(FileNotFoundError, match="incomplete"):
        load_artifact(directory)


def test_load_rejects_version_mismatch(tmp_path):
    """A future format change must fail loudly, not silently misbehave."""
    directory = tmp_path / "a"
    save_artifact(_model(), _metadata(), directory)

    path = directory / METADATA_FILENAME
    payload = json.loads(path.read_text())
    payload["artifact_version"] = "0.1"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="does not match expected"):
        load_artifact(directory)


def test_validate_features_returns_training_order():
    """Feature order is part of the contract: a shuffled dict must come
    back in the order the model expects, not the order it arrived in."""
    metadata = _metadata()
    values = validate_features(
        metadata, {"age": 75.0, "csf_fraction": 0.25, "grey_matter_fraction": 0.42}
    )
    assert values == [0.42, 0.25, 75.0]


def test_validate_features_rejects_missing_keys():
    with pytest.raises(ValueError, match="Missing required features"):
        validate_features(_metadata(), {"age": 75.0})


def test_validate_features_ignores_extra_keys():
    values = validate_features(
        _metadata(),
        {"grey_matter_fraction": 0.4, "csf_fraction": 0.2, "age": 70, "unused": 99},
    )
    assert len(values) == 3
