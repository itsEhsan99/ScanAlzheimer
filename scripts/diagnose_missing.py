"""Investigate which cohort subjects lack FSL segmentation and whether the
missingness is related to disease severity -- differentially missing hard
cases would silently inflate every downstream metric.
"""

from pathlib import Path

import pandas as pd

from ScanAlzheimer.data.paths import attach_image_paths, discover_sessions

FOLDS_PATH = Path("data/metadata/manifest_folds.csv")
DATA_ROOT = Path("data/raw")


def main() -> None:
    cohort = pd.read_csv(FOLDS_PATH)
    cohort = attach_image_paths(cohort, DATA_ROOT, variant="fsl_seg")
    missing = cohort[~cohort["image_available"]]

    print(f"Missing fsl_seg: {len(missing)} of {len(cohort)}\n")
    if missing.empty:
        return

    print("By label:")
    print(missing["label"].value_counts().rename({0: "CN", 1: "Demented"}).to_string())

    print("\nCDR distribution of missing subjects:")
    print(missing["cdr"].value_counts().sort_index().to_string())

    print("\nCDR distribution of present subjects:")
    present = cohort[cohort["image_available"]]
    print(present["cdr"].value_counts().sort_index().to_string())

    print("\nMean age -- missing vs present:")
    print(f"  missing: {missing['age'].mean():.1f}")
    print(f"  present: {present['age'].mean():.1f}")

    sessions = discover_sessions(DATA_ROOT)
    no_folder = [r for r in missing["raw_id"] if r not in sessions]
    print(f"\nSession folder absent entirely: {len(no_folder)}")
    print(f"Folder present but no fsl_seg:   {len(missing) - len(no_folder)}")

    for raw_id in missing["raw_id"].head(3):
        session_dir = sessions.get(raw_id)
        print(f"\n--- {raw_id} ---")
        if session_dir is None:
            print("  session folder not found on disk")
            continue
        seg_dir = session_dir / "FSL_SEG"
        if not seg_dir.exists():
            print("  no FSL_SEG directory")
            print(f"  subdirs present: {[p.name for p in session_dir.iterdir() if p.is_dir()]}")
        else:
            print(f"  FSL_SEG contents: {[p.name for p in seg_dir.iterdir()]}")


if __name__ == "__main__":
    main()
