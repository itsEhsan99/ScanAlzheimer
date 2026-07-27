"""Build and validate the OASIS-1 subject manifest -- the single source of
truth for subject/session metadata and labels used by every other module
in this project. No other module should read the raw OASIS spreadsheet
directly.
"""

from pathlib import Path

import pandas as pd
from pydantic import BaseModel

REQUIRED_COLUMNS = ["ID", "M/F", "Age", "Educ", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]


class SubjectRecord(BaseModel):
    """A single validated subject/session record with a derived label."""

    subject_id: str
    session_id: str
    raw_id: str
    sex: str
    age: int
    education: float | None = None
    ses: float | None = None
    mmse: float | None = None
    cdr: float
    label: int
    etiv: int
    nwbv: float
    asf: float


def parse_oasis_id(raw_id: str) -> tuple[str, str]:
    """Split an OASIS-1 ID like 'OAS1_0001_MR1' into (subject_id, session_id).

    subject_id is the grouping key used for subject-level train/test splits.
    """
    parts = raw_id.strip().split("_")
    if len(parts) != 3:
        raise ValueError(f"Unexpected OASIS ID format: {raw_id!r}")
    subject_id = f"{parts[0]}_{parts[1]}"
    session_id = parts[2]
    return subject_id, session_id


def derive_label(cdr: float | None) -> int | None:
    """Map a CDR score to a binary label.

    Returns 0 for CDR == 0 (cognitively normal), 1 for CDR >= 0.5 (very
    mild to moderate dementia), or None if no CDR was recorded (subject
    was not clinically assessed and cannot be used for classification).
    """
    if cdr is None or pd.isna(cdr):
        return None
    if cdr == 0.0:
        return 0
    if cdr >= 0.5:
        return 1
    return None


def load_raw_metadata(xlsx_path: str | Path) -> pd.DataFrame:
    """Load the raw OASIS-1 demographic/clinical spreadsheet, unmodified."""
    df = pd.read_excel(xlsx_path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in metadata file: {missing}")
    return df


def build_manifest(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw OASIS-1 metadata into a clean, validated, labeled manifest.

    Rows without a CDR score are dropped -- they have no clinical dementia
    assessment and cannot be used for supervised classification.
    """
    records: list[dict] = []
    for _, row in raw_df.iterrows():
        label = derive_label(row["CDR"])
        if label is None:
            continue

        subject_id, session_id = parse_oasis_id(row["ID"])

        record = SubjectRecord(
            subject_id=subject_id,
            session_id=session_id,
            raw_id=row["ID"],
            sex=row["M/F"],
            age=row["Age"],
            education=row["Educ"] if pd.notna(row["Educ"]) else None,
            ses=row["SES"] if pd.notna(row["SES"]) else None,
            mmse=row["MMSE"] if pd.notna(row["MMSE"]) else None,
            cdr=row["CDR"],
            label=label,
            etiv=row["eTIV"],
            nwbv=row["nWBV"],
            asf=row["ASF"],
        )
        records.append(record.model_dump())

    return pd.DataFrame.from_records(records)


def add_control_eligibility(manifest: pd.DataFrame, min_control_age: int = 60) -> pd.DataFrame:
    """Flag which rows belong to an age-matched modeling cohort.

    Demented subjects (label=1) are always eligible. Controls (label=0)
    are only eligible if age >= min_control_age -- otherwise the model
    could learn brain age as a shortcut for dementia status, since OASIS-1
    controls range from 18-96 while all demented subjects are 60+.
    """
    manifest = manifest.copy()
    manifest["age_matched_cohort"] = (manifest["label"] == 1) | (
        (manifest["label"] == 0) & (manifest["age"] >= min_control_age)
    )
    return manifest
