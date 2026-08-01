"""Extract regional grid features and evaluate them against the global
tissue baseline, using nested cross-validation for an unbiased estimate.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ScanAlzheimer.data.paths import attach_image_paths
from ScanAlzheimer.evaluation.metrics import (
    auc,
    balanced_accuracy,
    beats_chance,
    bootstrap_ci,
    classification_report,
    format_ci,
)
from ScanAlzheimer.evaluation.nested import (
    run_nested_cross_validation,
    selection_stability,
)
from ScanAlzheimer.features.regional import extract_regional_features
from ScanAlzheimer.features.tissue import load_segmentation
from ScanAlzheimer.models.baseline import feature_columns

FOLDS_PATH = Path("data/metadata/manifest_folds.csv")
DATA_ROOT = Path("data/raw")
GLOBAL_FEATURES_PATH = Path("data/metadata/features_tissue.csv")

PARAM_GRID = [{"C": c, "k": k} for c in (0.01, 0.1, 1.0) for k in (20, 40, 80)]


def main() -> None:
    args = _parse_args()
    cache = Path(f"data/metadata/features_regional_n{args.n_blocks}.csv")

    if cache.exists() and not args.rebuild:
        print(f"Loading cached regional features from {cache}")
        regional = pd.read_csv(cache)
    else:
        regional = _extract(args.n_blocks, cache)

    globals_df = pd.read_csv(GLOBAL_FEATURES_PATH)
    global_cols = feature_columns("tissue_plus_age")
    merged = regional.merge(
        globals_df[["subject_id", *[c for c in global_cols if c != "age"]]],
        on="subject_id",
        how="inner",
        validate="one_to_one",
    )

    regional_cols = _informative_columns(merged, prefix_of_interest=True)
    print(f"\nSubjects: {len(merged)}   Grid: {args.n_blocks}^3")
    print(f"Regional features: {len(regional_cols)} (after dropping constants)")
    print(f"Global features:   {len(global_cols)}")

    results = []
    results.append(_evaluate("global tissue + age", merged, global_cols))
    results.append(_evaluate("regional grid", merged, regional_cols))
    results.append(_evaluate("regional + global", merged, regional_cols + global_cols))

    print("\n" + "=" * 72)
    print("NESTED CV RESULTS -- hyperparameters chosen inside training folds only")
    print("=" * 72)
    summary = pd.DataFrame(results).sort_values("balanced_accuracy", ascending=False)
    print(summary.to_string(index=False))

    out = Path(f"data/metadata/regional_results_n{args.n_blocks}.csv")
    summary.to_csv(out, index=False)
    print(f"\nSaved -> {out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def _extract(n_blocks: int, cache: Path) -> pd.DataFrame:
    cohort = pd.read_csv(FOLDS_PATH)
    cohort = attach_image_paths(cohort, DATA_ROOT, variant="fsl_seg")
    cohort = cohort[cohort["image_available"]].reset_index(drop=True)

    print(f"Extracting {n_blocks}^3 regional features for {len(cohort)} subjects...")
    rows = []
    for counter, (_, subject) in enumerate(cohort.iterrows(), start=1):
        segmentation = load_segmentation(subject["image_path"])
        features = extract_regional_features(segmentation, n_blocks=n_blocks)
        rows.append(
            {
                "subject_id": subject["subject_id"],
                "label": int(subject["label"]),
                "fold": int(subject["fold"]),
                "age": subject["age"],
                **features,
            }
        )
        if counter % 25 == 0:
            print(f"  {counter} / {len(cohort)}")

    frame = pd.DataFrame(rows)
    frame.to_csv(cache, index=False)
    print(f"Saved regional features -> {cache}")
    return frame


def _informative_columns(frame: pd.DataFrame, prefix_of_interest: bool) -> list[str]:
    """Regional columns with nonzero variance.

    Corner blocks are pure background for every subject and carry no
    information. Dropping them uses no label information, so it cannot leak.
    """
    metadata = {"subject_id", "label", "fold", "age"}
    candidates = [
        c for c in frame.columns if c not in metadata and ("_b" in c) and prefix_of_interest
    ]
    return [c for c in candidates if frame[c].std() > 1e-9]


def _build_model(params: dict):
    """Univariate selection then logistic regression, both inside the
    pipeline so selection only ever sees training-fold labels."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=params["k"])),
            (
                "clf",
                LogisticRegression(C=params["C"], max_iter=2000, class_weight="balanced"),
            ),
        ]
    )


def _evaluate(name: str, frame: pd.DataFrame, columns: list[str]) -> dict:
    grid = [p for p in PARAM_GRID if p["k"] <= len(columns)] or [{"C": 0.1, "k": len(columns)}]

    subjects, selections = run_nested_cross_validation(frame, columns, _build_model, grid)

    y_true = subjects["label"].to_numpy()
    ba, ba_lo, ba_hi = bootstrap_ci(y_true, subjects["y_pred"].to_numpy(), balanced_accuracy)
    auc_point, auc_lo, auc_hi = bootstrap_ci(y_true, subjects["y_score"].to_numpy(), auc)
    report = classification_report(y_true, subjects["y_pred"].to_numpy())

    print(f"\n--- {name}  ({len(columns)} features) ---")
    print(f"  Balanced accuracy : {format_ci(ba, ba_lo, ba_hi)}")
    print(f"  AUC               : {format_ci(auc_point, auc_lo, auc_hi)}")
    print(f"  Sensitivity       : {report['sensitivity']:.3f}")
    print(f"  Specificity       : {report['specificity']:.3f}")
    print(f"  Beats chance      : {'YES' if beats_chance(ba_lo) else 'NO'}")
    print("  Selected per fold :")
    print(selection_stability(selections).to_string(index=False))

    return {
        "model": name,
        "n_features": len(columns),
        "balanced_accuracy": round(ba, 4),
        "ba_lower": round(ba_lo, 4),
        "ba_upper": round(ba_hi, 4),
        "auc": round(auc_point, 4),
        "auc_lower": round(auc_lo, 4),
        "auc_upper": round(auc_hi, 4),
    }


if __name__ == "__main__":
    main()


_ = np  # numpy is used indirectly by the pipeline components
