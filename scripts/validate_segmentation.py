"""Measure how well our GMM segmentation reproduces the OASIS FSL maps,
and whether features derived from it perform equivalently.

Agreement on the segmentation itself is necessary but not sufficient: what
matters for the application is whether a model trained on FSL features
still works when fed features we computed ourselves.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ScanAlzheimer.data.paths import attach_image_paths, find_image_path
from ScanAlzheimer.features.tissue import extract_tissue_features, load_segmentation
from ScanAlzheimer.preprocessing.intensity import preprocess_volume
from ScanAlzheimer.preprocessing.segmentation import (
    segment_volume,
    segmentation_agreement,
)
from ScanAlzheimer.preprocessing.volume import load_volume

FOLDS_PATH = Path("data/metadata/manifest_folds.csv")
DATA_ROOT = Path("data/raw")
OUTPUT_PATH = Path("data/metadata/features_tissue_gmm.csv")
FIGURE_PATH = Path("docs/figures/segmentation_agreement.png")

COMPARE_FEATURES = ["grey_matter_fraction", "csf_fraction", "gm_csf_ratio"]


def main() -> None:
    cohort = pd.read_csv(FOLDS_PATH)
    cohort = attach_image_paths(cohort, DATA_ROOT, variant="t88_masked_gfc")
    cohort = cohort[cohort["image_available"]].reset_index(drop=True)

    from ScanAlzheimer.data.paths import discover_sessions

    sessions = discover_sessions(DATA_ROOT)

    print(f"Segmenting {len(cohort)} subjects with our own GMM...")
    rows = []
    for counter, (_, subject) in enumerate(cohort.iterrows(), start=1):
        volume = preprocess_volume(load_volume(subject["image_path"]), scheme="minmax")
        ours = segment_volume(volume)

        reference_path = find_image_path(sessions[subject["raw_id"]], subject["raw_id"], "fsl_seg")
        reference = load_segmentation(reference_path)

        row = {
            "subject_id": subject["subject_id"],
            "label": int(subject["label"]),
            "fold": int(subject["fold"]),
            "age": subject["age"],
            **segmentation_agreement(ours, reference),
            **extract_tissue_features(ours),
        }
        rows.append(row)

        if counter % 25 == 0:
            print(f"  {counter} / {len(cohort)}")

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_PATH, index=False)

    print("\nDice agreement with FSL:")
    dice_columns = ["dice_csf", "dice_grey_matter", "dice_white_matter"]
    print(frame[dice_columns].describe().loc[["mean", "std", "min"]].round(3).to_string())

    _compare_features(frame)
    _plot(frame, FIGURE_PATH)

    print(f"\nSaved features -> {OUTPUT_PATH}")
    print(f"Saved figure   -> {FIGURE_PATH}")


def _compare_features(ours: pd.DataFrame) -> None:
    """Feature-level comparison, which is what actually matters downstream."""
    fsl = pd.read_csv("data/metadata/features_tissue.csv")
    merged = ours.merge(fsl, on="subject_id", suffixes=("_gmm", "_fsl"))

    print("\nFeature agreement (our GMM vs FSL):")
    for feature in COMPARE_FEATURES:
        correlation = merged[f"{feature}_gmm"].corr(merged[f"{feature}_fsl"])
        mean_gmm = merged[f"{feature}_gmm"].mean()
        mean_fsl = merged[f"{feature}_fsl"].mean()
        print(f"  {feature:<24} r={correlation:.3f}   mean {mean_gmm:.4f} vs {mean_fsl:.4f}")

    print("\nGroup separation preserved?")
    for feature in COMPARE_FEATURES:
        for source in ("gmm", "fsl"):
            column = f"{feature}_{source}"
            cn = merged.loc[merged["label_gmm"] == 0, column].mean()
            dem = merged.loc[merged["label_gmm"] == 1, column].mean()
            print(
                f"  {feature:<24} [{source}]  CN {cn:.4f}  Demented {dem:.4f}  "
                f"delta {dem - cn:+.4f}"
            )


def _plot(frame: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [
        frame["dice_csf"],
        frame["dice_grey_matter"],
        frame["dice_white_matter"],
    ]
    parts = ax.boxplot(data, tick_labels=["CSF", "Grey", "White"], patch_artist=True)
    for patch, colour in zip(parts["boxes"], ["#4C9AFF", "#9AA5B1", "#FFD166"], strict=True):
        patch.set_facecolor(colour)

    ax.axhline(0.7, color="red", linestyle="--", linewidth=1, label="Dice = 0.7")
    ax.set_ylabel("Dice coefficient")
    ax.set_title("Our GMM segmentation vs OASIS FSL reference")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
