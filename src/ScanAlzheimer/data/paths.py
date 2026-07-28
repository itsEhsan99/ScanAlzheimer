"""Resolve OASIS-1 on-disk image paths for a given session.

This module is the only place that knows about OASIS's directory layout.
Everything downstream consumes the manifest's `image_path` column and stays
agnostic to how the data happens to be organised on disk.
"""

from pathlib import Path

import pandas as pd

IMAGE_VARIANTS: dict[str, str] = {
    "t88_masked_gfc": "PROCESSED/MPRAGE/T88_111/{stem}_mpr_n4_anon_111_t88_masked_gfc.hdr",
    "t88_gfc": "PROCESSED/MPRAGE/T88_111/{stem}_mpr_n4_anon_111_t88_gfc.hdr",
    "subject_native": "PROCESSED/MPRAGE/SUBJ_111/{stem}_mpr_n4_anon_sbj_111.hdr",
    "fsl_seg": "FSL_SEG/{stem}_mpr_n4_anon_111_t88_masked_gfc_fseg.hdr",
}

DEFAULT_VARIANT = "t88_masked_gfc"


def build_image_path(session_dir: Path, stem: str, variant: str = DEFAULT_VARIANT) -> Path:
    """Return the expected image path for one session and image variant.

    Does not check whether the file exists -- that is the caller's job, so
    that missing files can be reported rather than raising mid-pipeline.
    """
    if variant not in IMAGE_VARIANTS:
        raise ValueError(f"Unknown image variant {variant!r}. Available: {sorted(IMAGE_VARIANTS)}")
    return session_dir / IMAGE_VARIANTS[variant].format(stem=stem)


def discover_sessions(data_root: Path) -> dict[str, Path]:
    """Scan all `disc*` directories under `data_root` for session folders.

    Returns a mapping of session ID (e.g. "OAS1_0001_MR1") to its directory.
    Discs that have not been downloaded yet are simply absent from the result.
    """
    sessions: dict[str, Path] = {}
    for disc_dir in sorted(Path(data_root).glob("disc*")):
        if not disc_dir.is_dir():
            continue
        for session_dir in sorted(disc_dir.iterdir()):
            if session_dir.is_dir() and session_dir.name.startswith("OAS1_"):
                sessions[session_dir.name] = session_dir
    return sessions


def attach_image_paths(
    manifest: pd.DataFrame,
    data_root: Path,
    variant: str = DEFAULT_VARIANT,
) -> pd.DataFrame:
    """Add `image_path` and `image_available` columns to a manifest.

    Rows whose session directory or image file is missing get an empty path
    and `image_available=False`, so that partial downloads are visible in the
    manifest instead of causing failures later.
    """
    sessions = discover_sessions(data_root)

    manifest = manifest.copy()
    paths: list[str] = []
    available: list[bool] = []

    for raw_id in manifest["raw_id"]:
        session_dir = sessions.get(raw_id)
        if session_dir is None:
            paths.append("")
            available.append(False)
            continue

        image_path = build_image_path(session_dir, raw_id, variant)
        paths.append(str(image_path))
        available.append(image_path.exists())

    manifest["image_path"] = paths
    manifest["image_available"] = available
    return manifest
