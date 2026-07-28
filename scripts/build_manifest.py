"""Build the OASIS-1 manifest from raw metadata, save it, and generate
diagnostic plots -- most importantly the age-distribution plot that shows
the age confound between controls and demented subjects.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ScanAlzheimer.data.manifest import (
    add_control_eligibility,
    build_manifest,
    load_raw_metadata,
)

METADATA_PATH = Path("data/metadata/oasis_cross-sectional-5708aa0a98d82080.xlsx")
OUTPUT_CSV = Path("data/metadata/manifest.csv")
OUTPUT_DIR = Path("docs/figures")


def main() -> None:
    raw_df = load_raw_metadata(METADATA_PATH)
    manifest = build_manifest(raw_df)
    manifest = add_control_eligibility(manifest, min_control_age=60)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUT_CSV, index=False)

    n_cn = (manifest["label"] == 0).sum()
    n_dem = (manifest["label"] == 1).sum()
    n_cohort = manifest["age_matched_cohort"].sum()
    n_cohort_cn = ((manifest["label"] == 0) & manifest["age_matched_cohort"]).sum()
    n_cohort_dem = ((manifest["label"] == 1) & manifest["age_matched_cohort"]).sum()

    print(f"Raw metadata rows:              {len(raw_df)}")
    print(f"Rows with a valid CDR label:    {len(manifest)}")
    print(f"  CN (label=0):                 {n_cn}")
    print(f"  Demented (label=1):           {n_dem}")
    print(f"Unique subjects:                {manifest['subject_id'].nunique()}")
    print()
    print(f"Age-matched cohort (controls >= 60): {n_cohort}")
    print(f"  CN in cohort:                       {n_cohort_cn}")
    print(f"  Demented in cohort:                 {n_cohort_dem}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_age_distribution(manifest, OUTPUT_DIR / "age_distribution.png")
    _plot_class_balance(manifest, OUTPUT_DIR / "class_balance.png")

    print(f"\nSaved manifest -> {OUTPUT_CSV}")
    print(f"Saved plots -> {OUTPUT_DIR}")


def _plot_age_distribution(manifest: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, name, color in [(0, "CN", "#4C9AFF"), (1, "Demented", "#FF6B6B")]:
        subset = manifest[manifest["label"] == label]
        ax.hist(subset["age"], bins=15, alpha=0.6, label=name, color=color)
    ax.axvline(60, color="black", linestyle="--", linewidth=1, label="age = 60 cutoff")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.set_title("Age distribution by label (before age filtering)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_class_balance(manifest: pd.DataFrame, out_path: Path) -> None:
    counts_all = manifest["label"].value_counts().reindex([0, 1], fill_value=0)
    cohort = manifest[manifest["age_matched_cohort"]]
    counts_cohort = cohort["label"].value_counts().reindex([0, 1], fill_value=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.35
    positions = [0, 1]
    ax.bar([p - width / 2 for p in positions], counts_all, width, label="All labeled")
    ax.bar([p + width / 2 for p in positions], counts_cohort, width, label="Age-matched cohort")
    ax.set_xticks(positions)
    ax.set_xticklabels(["CN", "Demented"])
    ax.set_ylabel("Count")
    ax.set_title("Class balance: all labeled vs. age-matched cohort")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
