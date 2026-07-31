"""Evaluate the tabular baselines with subject-level cross-validation and
bootstrap confidence intervals, plus a shuffled-label negative control.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

from ScanAlzheimer.evaluation.crossval import run_cross_validation, shuffle_labels
from ScanAlzheimer.evaluation.metrics import (
    auc,
    balanced_accuracy,
    beats_chance,
    bootstrap_ci,
    classification_report,
    confusion,
    format_ci,
)
from ScanAlzheimer.models.baseline import FEATURE_SETS, build_baseline_model, feature_columns

FEATURES_PATH = Path("data/metadata/features_tissue.csv")
FIGURE_PATH = Path("docs/figures/baseline_roc.png")
RESULTS_PATH = Path("data/metadata/baseline_results.csv")


def main() -> None:
    features = pd.read_csv(FEATURES_PATH)
    print(
        f"Subjects: {len(features)}  "
        f"(CN {(features['label'] == 0).sum()}, "
        f"Demented {(features['label'] == 1).sum()})"
    )
    print(f"Folds:    {features['fold'].nunique()}  |  subject-level\n")

    results = {}
    for name in FEATURE_SETS:
        cols = feature_columns(name)
        predictions = run_cross_validation(features, cols, build_baseline_model)
        results[name] = predictions
        _report(name, predictions, n_features=len(cols))

    print("\n" + "=" * 62)
    print("NEGATIVE CONTROL -- labels shuffled, signal destroyed")
    print("=" * 62)
    shuffled = shuffle_labels(features, seed=0)
    control = run_cross_validation(shuffled, feature_columns("tissue_only"), build_baseline_model)
    _report("shuffled_labels", control, n_features=len(feature_columns("tissue_only")))
    print("\nA pipeline that scores well here would be broken. Near 0.5 is correct.")

    _plot_roc(results, FIGURE_PATH)
    _save_results(results, RESULTS_PATH)
    print(f"\nSaved ROC     -> {FIGURE_PATH}")
    print(f"Saved results -> {RESULTS_PATH}")


def _report(name: str, predictions: pd.DataFrame, n_features: int) -> None:
    y_true = predictions["label"].to_numpy()
    y_pred = predictions["y_pred"].to_numpy()
    y_score = predictions["y_score"].to_numpy()

    report = classification_report(y_true, y_pred, y_score)
    ba, ba_lo, ba_hi = bootstrap_ci(y_true, y_pred, balanced_accuracy)
    auc_point, auc_lo, auc_hi = bootstrap_ci(y_true, y_score, auc)

    print(f"\n--- {name}  ({n_features} features) ---")
    print(f"  Balanced accuracy : {format_ci(ba, ba_lo, ba_hi)}")
    print(f"  AUC               : {format_ci(auc_point, auc_lo, auc_hi)}")
    print(f"  Sensitivity       : {report['sensitivity']:.3f}")
    print(f"  Specificity       : {report['specificity']:.3f}")
    print(f"  Beats chance      : {'YES' if beats_chance(ba_lo) else 'NO'}")

    tn, fp, fn, tp = confusion(y_true, y_pred).ravel()
    print(f"  Confusion         : TN={tn} FP={fp} FN={fn} TP={tp}")


def _plot_roc(results: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, predictions in results.items():
        fpr, tpr, _ = roc_curve(predictions["label"], predictions["y_score"])
        score = auc(predictions["label"].to_numpy(), predictions["y_score"].to_numpy())
        ax.plot(fpr, tpr, label=f"{name} (AUC {score:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Out-of-fold ROC, subject-level CV")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_results(results: dict[str, pd.DataFrame], out_path: Path) -> None:
    rows = []
    for name, predictions in results.items():
        y_true = predictions["label"].to_numpy()
        y_pred = predictions["y_pred"].to_numpy()
        y_score = predictions["y_score"].to_numpy()

        ba, ba_lo, ba_hi = bootstrap_ci(y_true, y_pred, balanced_accuracy)
        auc_point, auc_lo, auc_hi = bootstrap_ci(y_true, y_score, auc)
        report = classification_report(y_true, y_pred, y_score)

        rows.append(
            {
                "model": name,
                "balanced_accuracy": round(ba, 4),
                "ba_lower": round(ba_lo, 4),
                "ba_upper": round(ba_hi, 4),
                "auc": round(auc_point, 4),
                "auc_lower": round(auc_lo, 4),
                "auc_upper": round(auc_hi, 4),
                "sensitivity": round(report["sensitivity"], 4),
                "specificity": round(report["specificity"], 4),
            }
        )

    pd.DataFrame(rows).to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
