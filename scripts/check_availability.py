"""Report how much of the age-matched cohort actually has images on disk.

Downloads arrive disc by disc, so this makes partial coverage visible
instead of letting it silently shrink the training set.
"""

from pathlib import Path

import pandas as pd

from ScanAlzheimer.data.paths import attach_image_paths

FOLDS_PATH = Path("data/metadata/manifest_folds.csv")
DATA_ROOT = Path("data/raw")
VARIANT = "fsl_seg"


def main() -> None:
    cohort = pd.read_csv(FOLDS_PATH)
    cohort = attach_image_paths(cohort, DATA_ROOT, variant=VARIANT)

    n_total = len(cohort)
    available = cohort[cohort["image_available"]]

    print(f"Cohort subjects:        {n_total}")
    print(f"With {VARIANT} on disk: {len(available)}  ({100 * len(available) / n_total:.0f}%)\n")

    if available.empty:
        print("No images found. Check that discs are extracted under data/raw/")
        return

    by_fold = (
        available.groupby("fold")
        .agg(n=("subject_id", "count"), n_demented=("label", "sum"))
        .reindex(range(cohort["fold"].nunique()), fill_value=0)
    )
    by_fold["n_cn"] = by_fold["n"] - by_fold["n_demented"]
    print(by_fold.to_string())

    smallest_class = min(available["label"].sum(), (available["label"] == 0).sum())
    n_folds = cohort["fold"].nunique()
    print(f"\nSmallest class: {int(smallest_class)} subjects")
    if smallest_class < n_folds:
        print(f"NOT ENOUGH for {n_folds}-fold CV yet -- download the remaining discs.")
    else:
        print(f"Sufficient for {n_folds}-fold CV.")


if __name__ == "__main__":
    main()
