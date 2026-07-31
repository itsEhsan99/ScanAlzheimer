"""Unit tests for OASIS path resolution. Builds a fake OASIS directory tree
in a temporary folder -- never touches the real dataset.
"""

import pandas as pd
import pytest

from ScanAlzheimer.data.paths import (
    attach_image_paths,
    discover_sessions,
    find_image_path,
    image_pattern,
)


def _make_fake_session(data_root, disc, stem, with_image=True, n_acq=4):
    """Create a minimal fake OASIS session directory.

    `n_acq` mimics the varying acquisition count in real filenames.
    """
    session_dir = data_root / disc / stem
    image_dir = session_dir / "PROCESSED" / "MPRAGE" / "T88_111"
    image_dir.mkdir(parents=True)
    if with_image:
        base = f"{stem}_mpr_n{n_acq}_anon_111_t88_masked_gfc"
        for ext in (".hdr", ".img"):
            (image_dir / f"{base}{ext}").write_bytes(b"")
    return session_dir


def test_image_pattern_includes_wildcard_for_acquisition_count():
    pattern = image_pattern("OAS1_0001_MR1", "t88_masked_gfc")
    assert "mpr_n*" in pattern
    assert pattern.endswith("t88_masked_gfc.hdr")


def test_image_pattern_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Unknown image variant"):
        image_pattern("OAS1_0001_MR1", "nonexistent")


def test_masked_and_unmasked_patterns_differ():
    masked = image_pattern("OAS1_0001_MR1", "t88_masked_gfc")
    unmasked = image_pattern("OAS1_0001_MR1", "t88_gfc")
    assert masked != unmasked
    assert "masked" not in unmasked


@pytest.mark.parametrize("n_acq", [1, 3, 4])
def test_find_image_path_matches_any_acquisition_count(tmp_path, n_acq):
    """The bug this replaced: hard-coding n4 dropped every n3 session."""
    session_dir = _make_fake_session(tmp_path, "disc1", "OAS1_0001_MR1", n_acq=n_acq)
    found = find_image_path(session_dir, "OAS1_0001_MR1")
    assert found is not None
    assert f"mpr_n{n_acq}_" in found.name


def test_find_image_path_returns_none_when_absent(tmp_path):
    session_dir = _make_fake_session(tmp_path, "disc1", "OAS1_0002_MR1", with_image=False)
    assert find_image_path(session_dir, "OAS1_0002_MR1") is None


def test_find_image_path_raises_on_ambiguous_match(tmp_path):
    """Two candidates means our layout assumptions are wrong -- fail loudly."""
    session_dir = tmp_path / "disc1" / "OAS1_0003_MR1"
    image_dir = session_dir / "PROCESSED" / "MPRAGE" / "T88_111"
    image_dir.mkdir(parents=True)
    for n in (3, 4):
        (image_dir / f"OAS1_0003_MR1_mpr_n{n}_anon_111_t88_masked_gfc.hdr").write_bytes(b"")

    with pytest.raises(ValueError, match="Ambiguous"):
        find_image_path(session_dir, "OAS1_0003_MR1")


def test_masked_pattern_does_not_match_unmasked_file(tmp_path):
    """The two variants must stay distinguishable despite the wildcard."""
    session_dir = tmp_path / "disc1" / "OAS1_0004_MR1"
    image_dir = session_dir / "PROCESSED" / "MPRAGE" / "T88_111"
    image_dir.mkdir(parents=True)
    (image_dir / "OAS1_0004_MR1_mpr_n4_anon_111_t88_masked_gfc.hdr").write_bytes(b"")

    assert find_image_path(session_dir, "OAS1_0004_MR1", "t88_gfc") is None
    assert find_image_path(session_dir, "OAS1_0004_MR1", "t88_masked_gfc") is not None


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


def test_attach_image_paths_handles_mixed_acquisition_counts(tmp_path):
    """A cohort mixing n3 and n4 sessions must come back fully available."""
    _make_fake_session(tmp_path, "disc1", "OAS1_0001_MR1", n_acq=4)
    _make_fake_session(tmp_path, "disc1", "OAS1_0002_MR1", n_acq=3)

    manifest = pd.DataFrame(
        {
            "subject_id": ["OAS1_0001", "OAS1_0002"],
            "raw_id": ["OAS1_0001_MR1", "OAS1_0002_MR1"],
        }
    )
    result = attach_image_paths(manifest, tmp_path)
    assert result["image_available"].all()


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
