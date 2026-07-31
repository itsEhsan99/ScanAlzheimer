"""Resolve OASIS-1 on-disk image paths for a given session.

This module is the only place that knows about OASIS's directory layout.
Everything downstream consumes the manifest's `image_path` column and stays
agnostic to how the data happens to be organised on disk.

Filenames encode how many MPRAGE acquisitions were averaged (`mpr_n3`,
`mpr_n4`, ...), which varies per session, so paths are matched by pattern
rather than hard-coded. Hard-coding one count silently drops the sessions
that used another -- and since acquisitions are usually discarded for
motion, which correlates with dementia, that loss would not be random.
"""

from pathlib import Path

import pandas as pd

# Glob patterns relative to a session directory. `stem` is the session ID,
# e.g. "OAS1_0001_MR1"; `n*` absorbs the acquisition count.
IMAGE_VARIANTS: dict[str, str] = {
    # Atlas-registered, gain-field corrected, skull-stripped.
    "t88_masked_gfc": "PROCESSED/MPRAGE/T88_111/{stem}_mpr_n*_anon_111_t88_masked_gfc.hdr",
    # Atlas-registered, gain-field corrected, skull intact.
    "t88_gfc": "PROCESSED/MPRAGE/T88_111/{stem}_mpr_n*_anon_111_t88_gfc.hdr",
    # Motion-corrected average in the subject's own native space.
    "subject_native": "PROCESSED/MPRAGE/SUBJ_111/{stem}_mpr_n*_anon_sbj_111.hdr",
    # FSL tissue segmentation (grey/white/CSF).
    "fsl_seg": "FSL_SEG/{stem}_mpr_n*_anon_111_t88_masked_gfc_fseg.hdr",
}

DEFAULT_VARIANT = "t88_masked_gfc"


def image_pattern(stem: str, variant: str = DEFAULT_VARIANT) -> str:
    """Return the glob pattern for one session and image variant."""
    if variant not in IMAGE_VARIANTS:
        raise ValueError(f"Unknown image variant {variant!r}. Available: {sorted(IMAGE_VARIANTS)}")
    return IMAGE_VARIANTS[variant].format(stem=stem)


def find_image_path(session_dir: Path, stem: str, variant: str = DEFAULT_VARIANT) -> Path | None:
    """Locate the image file for one session, or None if it is absent.

    Raises if more than one file matches: an ambiguous match means our
    assumptions about the layout are wrong, and silently picking one would
    hide that.
    """
    pattern = image_pattern(stem, variant)
    matches = sorted(Path(session_dir).glob(pattern))

    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Ambiguous match for {stem} / {variant}: {[m.name for m in matches]}")
    return matches[0]


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
        image_path = None if session_dir is None else find_image_path(session_dir, raw_id, variant)

        paths.append("" if image_path is None else str(image_path))
        available.append(image_path is not None)

    manifest["image_path"] = paths
    manifest["image_available"] = available
    return manifest
