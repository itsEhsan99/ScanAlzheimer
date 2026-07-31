"""Break down baseline performance by CDR severity and test whether adding
age gives a real improvement, using a paired bootstrap of the difference.
"""

from pathlib import Path

import pandas as pd

from ScanAlzheimer.evaluation.crossval import run_cross_validation
from ScanAlzheimer.evaluation.metrics import (
    auc,
    balanced_accuracy,
    bootstrap_difference,
    format_ci,
)
from ScanAlzheimer.models.baseline import build_baseline_model, feature_columns

FEATURES_PATH = Path("data/metadata/features_tissue.csv")
FOLDS_PATH = Path("data/metadata/manifest_folds.csv")


def main() -> None:
    features = pd.read_csv(FEATURES_PATH)
    cdr = pd.read_csv(FOLDS_PATH)[["subject_id", "cdr"]]
    features = features.merge(cdr, on="subject_id", how="left", validate="one_to_one")

    tissue = run_cross_validation(features, feature_columns("tissue_only"), build_baseline_model)
    combined = run_cross_validation(
        features, feature_columns("tissue_plus_age"), build_baseline_model
    )

    print("=" * 58)
    print("Does adding age genuinely help? (paired bootstrap)")
    print("=" * 58)

    y_true = tissue["label"].to_numpy()
    d, lo, hi = bootstrap_difference(
        y_true, tissue["y_pred"].to_numpy(), combined["y_pred"].to_numpy(), balanced_accuracy
    )
    print(f"  Delta balanced accuracy : {format_ci(d, lo, hi)}")
    print(f"  Excludes zero           : {'YES' if lo > 0 else 'NO'}")

    d, lo, hi = bootstrap_difference(
        y_true, tissue["y_score"].to_numpy(), combined["y_score"].to_numpy(), auc
    )
    print(f"  Delta AUC               : {format_ci(d, lo, hi)}")
    print(f"  Excludes zero           : {'YES' if lo > 0 else 'NO'}")

    print("\n" + "=" * 58)
    print("Sensitivity by dementia severity (tissue_plus_age)")
    print("=" * 58)

    positives = combined[combined["label"] == 1]
    rows = []
    for cdr_value, group in positives.groupby("cdr"):
        rows.append(
            {
                "cdr": cdr_value,
                "severity": {0.5: "very mild", 1.0: "mild", 2.0: "moderate"}.get(cdr_value, "?"),
                "n": len(group),
                "detected": int(group["y_pred"].sum()),
                "sensitivity": round(group["y_pred"].mean(), 3),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    n_mild = (positives["cdr"] == 0.5).sum()
    print(f"\nCDR 0.5 is {100 * n_mild / len(positives):.0f}% of the positive class.")
    print("Sensitivity rising with severity is the expected clinical pattern:")
    print("earlier disease means subtler atrophy, and is genuinely harder.")


if __name__ == "__main__":
    main()
