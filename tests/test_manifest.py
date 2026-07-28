"""Unit tests for the manifest-building logic. Uses small synthetic
DataFrames, never the real OASIS files -- these tests must run without
any data present.
"""

import pandas as pd
import pytest

from ScanAlzheimer.data.manifest import (
    add_control_eligibility,
    build_manifest,
    derive_label,
    parse_oasis_id,
)


def test_parse_oasis_id_standard_case():
    subject_id, session_id = parse_oasis_id("OAS1_0001_MR1")
    assert subject_id == "OAS1_0001"
    assert session_id == "MR1"


def test_parse_oasis_id_rescan_session():
    subject_id, session_id = parse_oasis_id("OAS1_0001_MR2")
    assert subject_id == "OAS1_0001"
    assert session_id == "MR2"


def test_parse_oasis_id_invalid_format_raises():
    with pytest.raises(ValueError):
        parse_oasis_id("not-a-valid-id")


@pytest.mark.parametrize(
    "cdr,expected",
    [
        (0.0, 0),
        (0.5, 1),
        (1.0, 1),
        (2.0, 1),
        (None, None),
        (float("nan"), None),
    ],
)
def test_derive_label(cdr, expected):
    assert derive_label(cdr) == expected


def _make_raw_row(id_="OAS1_0001_MR1", sex="F", age=74, cdr=0.0):
    return {
        "ID": id_,
        "M/F": sex,
        "Hand": "R",
        "Age": age,
        "Educ": 2.0,
        "SES": 3.0,
        "MMSE": 29.0,
        "CDR": cdr,
        "eTIV": 1344,
        "nWBV": 0.743,
        "ASF": 1.306,
        "Delay": float("nan"),
    }


def test_build_manifest_drops_rows_without_cdr():
    raw_df = pd.DataFrame(
        [
            _make_raw_row(id_="OAS1_0001_MR1", cdr=0.0),
            _make_raw_row(id_="OAS1_0002_MR1", cdr=float("nan")),
        ]
    )
    manifest = build_manifest(raw_df)
    assert len(manifest) == 1
    assert manifest.iloc[0]["subject_id"] == "OAS1_0001"


def test_build_manifest_assigns_correct_labels():
    raw_df = pd.DataFrame(
        [
            _make_raw_row(id_="OAS1_0001_MR1", cdr=0.0),
            _make_raw_row(id_="OAS1_0002_MR1", cdr=0.5),
            _make_raw_row(id_="OAS1_0003_MR1", cdr=1.0),
        ]
    )
    manifest = build_manifest(raw_df)
    labels = dict(zip(manifest["subject_id"], manifest["label"], strict=True))
    assert labels["OAS1_0001"] == 0
    assert labels["OAS1_0002"] == 1
    assert labels["OAS1_0003"] == 1


def test_add_control_eligibility_excludes_young_controls():
    raw_df = pd.DataFrame(
        [
            _make_raw_row(id_="OAS1_0001_MR1", cdr=0.0, age=25),
            _make_raw_row(id_="OAS1_0002_MR1", cdr=0.0, age=70),
            _make_raw_row(id_="OAS1_0003_MR1", cdr=1.0, age=45),
        ]
    )
    manifest = build_manifest(raw_df)
    manifest = add_control_eligibility(manifest, min_control_age=60)

    eligibility = dict(zip(manifest["subject_id"], manifest["age_matched_cohort"], strict=True))
    assert not eligibility["OAS1_0001"]
    assert eligibility["OAS1_0002"]
    assert eligibility["OAS1_0003"]


def test_subject_id_stable_across_rescan_sessions():
    """Sanity check: a subject with multiple sessions (e.g. reliability
    rescans) must map to a single subject_id, so grouping stays correct."""
    raw_df = pd.DataFrame(
        [
            _make_raw_row(id_="OAS1_0001_MR1", cdr=0.0),
            _make_raw_row(id_="OAS1_0001_MR2", cdr=0.0),
        ]
    )
    manifest = build_manifest(raw_df)
    assert manifest["subject_id"].nunique() == 1
    assert len(manifest) == 2
