"""Assign subject-level cross-validation folds to the age-matched cohort,
verify no leakage, and report per-fold composition.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ScanAlzheimer.evaluation.splits import (
    assert_no_subject_leakage,
    assign_subject_folds,
    attach_folds,
    summarize_folds,
)

MANIFEST_PATH = Path("data/metadata/manifest.csv")
OUTPUT_PATH = Path("data/metadata/manifest_folds.csv")
FIGURE_PATH = Path("docs/figures/fold_composition.png")

N_SPLITS = 5
SEED = 42


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    cohort = manifest[manifest["age_matched_cohort"]].reset_index(drop=True)

    print(f"Full labeled manifest:  {len(manifest)} subjects")
    print(f"Age-matched cohort:     {len(cohort)} subjects")
    print(f"Folds:                  {N_SPLITS} (seed={SEED})\n")

    subject_folds = assign_subject_folds(cohort, n_splits=N_SPLITS, seed=SEED)
    cohort = attach_folds(cohort, subject_folds)

    assert_no_subject_leakage(cohort)
    print("Leakage guard: PASSED -- no subject appears in more than one fold.\n")

    summary = summarize_folds(cohort)
    print(summary.to_string(index=False))

    cohort.to_csv(OUTPUT_PATH, index=False)
    _plot_fold_composition(summary, FIGURE_PATH)

    print(f"\nSaved folds  -> {OUTPUT_PATH}")
    print(f"Saved figure -> {FIGURE_PATH}")


def _plot_fold_composition(summary: pd.DataFrame, out_path: Path) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4))

    ax_left.bar(summary["fold"], summary["n_cn"], label="CN", color="#4C9AFF")
    ax_left.bar(
        summary["fold"],
        summary["n_demented"],
        bottom=summary["n_cn"],
        label="Demented",
        color="#FF6B6B",
    )
    ax_left.set_xlabel("Fold")
    ax_left.set_ylabel("Subjects")
    ax_left.set_title("Class composition per fold")
    ax_left.set_xticks(summary["fold"])
    ax_left.legend()

    ax_right.bar(summary["fold"], summary["mean_age"], color="#9AA5B1")
    ax_right.set_xlabel("Fold")
    ax_right.set_ylabel("Mean age")
    ax_right.set_title("Mean age per fold")
    ax_right.set_xticks(summary["fold"])
    ax_right.set_ylim(bottom=min(summary["mean_age"]) - 5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
