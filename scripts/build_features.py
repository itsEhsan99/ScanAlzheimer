"""Extract tissue features for every available cohort subject and plot the
distributions by label -- the first look at whether the biological signal
is actually present in this data.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ScanAlzheimer.data.paths import attach_image_paths
from ScanAlzheimer.features.tissue import extract_tissue_features, load_segmentation

FOLDS_PATH = Path("data/metadata/manifest_folds.csv")
DATA_ROOT = Path("data/raw")
OUTPUT_PATH = Path("data/metadata/features_tissue.csv")
FIGURE_PATH = Path("docs/figures/tissue_features.png")

PLOT_FEATURES = ["grey_matter_fraction", "csf_fraction", "gm_csf_ratio", "brain_csf_ratio"]


def main() -> None:
    cohort = pd.read_csv(FOLDS_PATH)
    cohort = attach_image_paths(cohort, DATA_ROOT, variant="fsl_seg")
    available = cohort[cohort["image_available"]].reset_index(drop=True)

    print(f"Extracting features for {len(available)} / {len(cohort)} cohort subjects...")

    rows = []
    for i, row in available.iterrows():
        segmentation = load_segmentation(row["image_path"])
        features = extract_tissue_features(segmentation)
        rows.append(
            {
                "subject_id": row["subject_id"],
                "label": row["label"],
                "fold": row["fold"],
                "age": row["age"],
                **features,
            }
        )
        if (i + 1) % 25 == 0:
            print(f"  {i + 1} done")

    features_df = pd.DataFrame(rows)
    features_df.to_csv(OUTPUT_PATH, index=False)

    print("\nMean values by group:")
    summary = features_df.groupby("label")[PLOT_FEATURES].mean().round(4)
    summary.index = ["CN", "Demented"]
    print(summary.to_string())

    _plot_distributions(features_df, FIGURE_PATH)
    print(f"\nSaved features -> {OUTPUT_PATH}")
    print(f"Saved figure   -> {FIGURE_PATH}")


def _plot_distributions(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(PLOT_FEATURES), figsize=(4 * len(PLOT_FEATURES), 4))
    for ax, feature in zip(axes, PLOT_FEATURES, strict=True):
        data = [df.loc[df["label"] == 0, feature], df.loc[df["label"] == 1, feature]]
        parts = ax.boxplot(data, tick_labels=["CN", "Demented"], patch_artist=True)
        for patch, colour in zip(parts["boxes"], ["#4C9AFF", "#FF6B6B"], strict=True):
            patch.set_facecolor(colour)
        ax.set_title(feature, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Tissue features by dementia status")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
