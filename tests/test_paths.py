"""Unit tests for OASIS path resolution. Builds a fake OASIS directory tree
in a temporary folder -- never touches the real dataset.
"""

import pandas as pd
import pytest
from ScanAlzheimer.data.paths import (
    attach_image_paths,
    build_image_path,
    discover_sessions,
)


def _make_fake_session(data_root, disc, stem, with_image=True):
    """Create a minimal fake OASIS session directory."""
    session_dir = data_root / disc / stem
    image_dir = session_dir / "PROCESSED" / "MPRAGE" / "T88_111"
    image_dir.mkdir(parents=True)
    if with_image:
        for ext in (".hdr", ".img"):
            (image_dir / f"{stem}_mpr_n4_anon_111_t88_masked_gfc{ext}").write_bytes(b"")
    return session_dir


def test_build_image_path_masked_variant(tmp_path):
    path = build_image_path(tmp_path, "OAS1_0001_MR1", "t88_masked_gfc")
    assert path.name == "OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc.hdr"
    assert "T88_111" in path.parts


def test_build_image_path_unmasked_variant_differs(tmp_path):
    masked = build_image_path(tmp_path, "OAS1_0001_MR1", "t88_masked_gfc")
    unmasked = build_image_path(tmp_path, "OAS1_0001_MR1", "t88_gfc")
    assert masked != unmasked
    assert "masked" not in unmasked.name


def test_build_image_path_rejects_unknown_variant(tmp_path):
    with pytest.raises(ValueError, match="Unknown image variant"):
        build_image_path(tmp_path, "OAS1_0001_MR1", "nonexistent")


def test_discover_sessions_across_multiple_discs(tmp_path):
    _make_fake_session(tmp_path, "disc1", "OAS1_0001_MR1")
    _make_fake_session(tmp_path, "disc1", "OAS1_0002_MR1")
    _make_fake_session(tmp_path, "disc2", "OAS1_0050_MR1")

    sessions = discover_sessions(tmp_path)
    assert set(sessions) == {"OAS1_0001_MR1", "OAS1_0002_MR1", "OAS1_0050_MR1"}


def test_discover_sessions_ignores_non_oasis_folders(tmp_path):
    _make_fake_session(tmp_path, "disc1", "OAS1_0001_MR1")
    (tmp_path / "disc1" / "README_junk").mkdir()

    sessions = discover_sessions(tmp_path)
    assert set(sessions) == {"OAS1_0001_MR1"}


def test_discover_sessions_on_empty_root(tmp_path):
    assert discover_sessions(tmp_path) == {}


def test_attach_image_paths_marks_missing_sessions(tmp_path):
    """A subject whose disc has not been downloaded must be flagged, not crash."""
    _make_fake_session(tmp_path, "disc1", "OAS1_0001_MR1")

    manifest = pd.DataFrame(
        {
            "subject_id": ["OAS1_0001", "OAS1_0999"],
            "raw_id": ["OAS1_0001_MR1", "OAS1_0999_MR1"],
        }
    )
    result = attach_image_paths(manifest, tmp_path)

    availability = dict(zip(result["raw_id"], result["image_available"], strict=True))
    assert availability["OAS1_0001_MR1"]
    assert not availability["OAS1_0999_MR1"]
    assert result.loc[result["raw_id"] == "OAS1_0999_MR1", "image_path"].item() == ""


def test_attach_image_paths_marks_session_with_missing_file(tmp_path):
    """A session folder that exists but lacks the image file is not available."""
    _make_fake_session(tmp_path, "disc1", "OAS1_0002_MR1", with_image=False)

    manifest = pd.DataFrame({"subject_id": ["OAS1_0002"], "raw_id": ["OAS1_0002_MR1"]})
    result = attach_image_paths(manifest, tmp_path)
    assert not result["image_available"].item()


def test_attach_image_paths_does_not_mutate_input(tmp_path):
    manifest = pd.DataFrame({"subject_id": ["OAS1_0001"], "raw_id": ["OAS1_0001_MR1"]})
    attach_image_paths(manifest, tmp_path)
    assert "image_path" not in manifest.columns
