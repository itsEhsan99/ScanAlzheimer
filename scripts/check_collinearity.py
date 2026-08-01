"""Test whether a smaller, non-collinear feature set performs as well.

Six tissue features derived from three fractions that sum to one are
heavily collinear, which lets logistic regression split weight between
them in ways that flip coefficient signs. That does not hurt accuracy,
but it makes per-feature explanations misleading -- and this model is
meant to be explained to a user.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ScanAlzheimer.evaluation.metrics import (
    auc,
    balanced_accuracy,
    bootstrap_ci,
    format_ci,
)
from ScanAlzheimer.evaluation.nested import run_nested_cross_validation

FEATURES_PATH = Path("data/metadata/features_tissue_gmm.csv")
PARAM_GRID = [{"C": c} for c in (0.01, 0.1, 1.0)]

CANDIDATE_SETS = {
    "full (6 tissue + age)": [
        "grey_matter_fraction",
        "csf_fraction",
        "white_matter_fraction",
        "gm_csf_ratio",
        "gm_wm_ratio",
        "brain_csf_ratio",
        "age",
    ],
    "fractions + age": ["grey_matter_fraction", "csf_fraction", "age"],
    "atrophy ratio + age": ["brain_csf_ratio", "age"],
    "gm/csf + age": ["gm_csf_ratio", "age"],
    "minimal (3 orthogonal)": ["brain_csf_ratio", "gm_wm_ratio", "age"],
}


def main() -> None:
    frame = pd.read_csv(FEATURES_PATH)

    print("Correlation among the full feature set:")
    tissue = CANDIDATE_SETS["full (6 tissue + age)"]
    correlations = frame[tissue].corr().abs().to_numpy(copy=True)
    np.fill_diagonal(correlations, 0.0)
    correlations = pd.DataFrame(correlations, index=tissue, columns=tissue)
    print(correlations.round(2).to_string())
    print(f"\nHighest off-diagonal correlation: {correlations.values.max():.2f}")

    results = []
    for name, columns in CANDIDATE_SETS.items():
        results.append(_evaluate(name, frame, columns))

    summary = pd.DataFrame(results).sort_values("auc", ascending=False)
    print("\n" + "=" * 70)
    print("FEATURE SET COMPARISON (nested CV)")
    print("=" * 70)
    print(summary.to_string(index=False))
    print("\nIf a small set matches the full one, prefer it: fewer features")
    print("means coefficients that can honestly be shown to a user.")


def _build(params: dict) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(C=params["C"], max_iter=2000, class_weight="balanced"),
            ),
        ]
    )


def _evaluate(name: str, frame: pd.DataFrame, columns: list[str]) -> dict:
    predictions, _ = run_nested_cross_validation(frame, columns, _build, PARAM_GRID)

    y_true = predictions["label"].to_numpy()
    ba, ba_lo, ba_hi = bootstrap_ci(y_true, predictions["y_pred"].to_numpy(), balanced_accuracy)
    auc_point, auc_lo, auc_hi = bootstrap_ci(y_true, predictions["y_score"].to_numpy(), auc)

    model = _build({"C": 1.0})
    model.fit(frame[columns].to_numpy(), frame["label"].to_numpy())
    coefficients = model.named_steps["clf"].coef_.ravel()

    print(f"\n--- {name}  ({len(columns)} features) ---")
    print(f"  Balanced accuracy : {format_ci(ba, ba_lo, ba_hi)}")
    print(f"  AUC               : {format_ci(auc_point, auc_lo, auc_hi)}")
    print("  Coefficients (positive -> demented):")
    for column, value in zip(columns, coefficients, strict=True):
        print(f"    {column:<24} {value:+.3f}")

    return {
        "features": name,
        "n": len(columns),
        "balanced_accuracy": round(ba, 4),
        "ba_lower": round(ba_lo, 4),
        "ba_upper": round(ba_hi, 4),
        "auc": round(auc_point, 4),
        "auc_lower": round(auc_lo, 4),
        "auc_upper": round(auc_hi, 4),
    }


if __name__ == "__main__":
    main()
