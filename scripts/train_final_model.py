"""Train the deployable model on features from our own segmentation.

The tissue baseline was developed against OASIS-provided FSL maps, which
an uploaded scan does not come with. Training on features from our own
GMM segmentation instead means the pipeline that produced the training
features is the same one that runs at inference -- no domain shift between
development and deployment, and no dependency on precomputed data.
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ScanAlzheimer.evaluation.crossval import shuffle_labels
from ScanAlzheimer.evaluation.metrics import (
    auc,
    balanced_accuracy,
    beats_chance,
    bootstrap_ci,
    classification_report,
    confusion,
    format_ci,
)
from ScanAlzheimer.evaluation.nested import (
    run_nested_cross_validation,
    selection_stability,
)
from ScanAlzheimer.inference.artifact import ModelMetadata, save_artifact
from ScanAlzheimer.models.baseline import feature_columns

GMM_FEATURES_PATH = Path("data/metadata/features_tissue_gmm.csv")
FSL_FEATURES_PATH = Path("data/metadata/features_tissue.csv")
ARTIFACT_DIR = Path("artifacts/model")
RESULTS_PATH = Path("data/metadata/final_model_results.csv")

FEATURE_COLUMNS = feature_columns("atrophy_plus_age")
PARAM_GRID = [{"C": c} for c in (0.001, 0.01, 0.1, 1.0)]

LIMITATIONS = [
    "Input must be a skull-stripped, bias-corrected T1 volume; the "
    "segmentation assumes T1 contrast ordering and would be wrong on T2.",
    "Segmentation is intensity-only, with no spatial prior. Mean Dice "
    "against the OASIS FSL reference is 0.90-0.92.",
    "70% of the positive class is CDR 0.5, the earliest stage, which makes "
    "this task harder than the AD-vs-healthy comparisons behind many "
    "published accuracy figures.",
    "Controls are age-restricted to 60+; performance on younger controls "
    "is untested and would likely be optimistic.",
    "No external validation on an independent cohort yet.",
    "Research prototype, not a medical device.",
    "The deployed model uses two features. Larger tissue feature sets were "
    "evaluated and performed no better; their coefficients were also "
    "uninterpretable due to collinearity up to r=0.97.",
]


def main() -> None:
    gmm = pd.read_csv(GMM_FEATURES_PATH)
    print(
        f"Subjects: {len(gmm)}  "
        f"(CN {(gmm['label'] == 0).sum()}, Demented {(gmm['label'] == 1).sum()})"
    )
    print(f"Features: {FEATURE_COLUMNS}\n")

    results = [_evaluate("our GMM segmentation", gmm)]

    fsl = pd.read_csv(FSL_FEATURES_PATH)
    results.append(_evaluate("OASIS FSL segmentation (reference)", fsl))

    print("\n" + "=" * 66)
    print("NEGATIVE CONTROL -- labels shuffled")
    print("=" * 66)
    _evaluate("shuffled labels", shuffle_labels(gmm, seed=0), quiet_selection=True)
    print("\nNear 0.5 is correct.")

    summary = pd.DataFrame(results)
    summary.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved results -> {RESULTS_PATH}")

    _fit_and_save(results[0], gmm)


def _build_model(params: dict) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(C=params["C"], max_iter=2000, class_weight="balanced"),
            ),
        ]
    )


def _evaluate(name: str, frame: pd.DataFrame, quiet_selection: bool = False) -> dict:
    predictions, selections = run_nested_cross_validation(
        frame, FEATURE_COLUMNS, _build_model, PARAM_GRID
    )

    y_true = predictions["label"].to_numpy()
    y_pred = predictions["y_pred"].to_numpy()
    y_score = predictions["y_score"].to_numpy()

    ba, ba_lo, ba_hi = bootstrap_ci(y_true, y_pred, balanced_accuracy)
    auc_point, auc_lo, auc_hi = bootstrap_ci(y_true, y_score, auc)
    report = classification_report(y_true, y_pred, y_score)

    print(f"\n--- {name} ---")
    print(f"  Balanced accuracy : {format_ci(ba, ba_lo, ba_hi)}")
    print(f"  AUC               : {format_ci(auc_point, auc_lo, auc_hi)}")
    print(f"  Sensitivity       : {report['sensitivity']:.3f}")
    print(f"  Specificity       : {report['specificity']:.3f}")
    print(f"  Beats chance      : {'YES' if beats_chance(ba_lo) else 'NO'}")

    tn, fp, fn, tp = confusion(y_true, y_pred).ravel()
    print(f"  Confusion         : TN={tn} FP={fp} FN={fn} TP={tp}")

    if not quiet_selection:
        print("  Selected per fold :")
        print(selection_stability(selections).to_string(index=False))

    return {
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


def _fit_and_save(metrics: dict, frame: pd.DataFrame) -> None:
    """Refit on the full cohort using the most frequently selected setting.

    Stored metrics are the nested-CV estimates: scoring this fit on its own
    training data would be meaningless.
    """
    _, selections = run_nested_cross_validation(frame, FEATURE_COLUMNS, _build_model, PARAM_GRID)
    chosen = float(selection_stability(selections).iloc[0]["C"])
    print(f"\nRefitting on all {len(frame)} subjects with C={chosen}")

    model = _build_model({"C": chosen})
    model.fit(frame[FEATURE_COLUMNS].to_numpy(), frame["label"].to_numpy())

    metadata = ModelMetadata(
        name="tissue_gmm_plus_age",
        feature_columns=list(FEATURE_COLUMNS),
        feature_source="in-house GMM tissue segmentation",
        task="CDR 0 (cognitively normal) vs CDR >= 0.5 (very mild to moderate dementia)",
        positive_class="demented",
        n_train_subjects=int(len(frame)),
        metrics={
            "balanced_accuracy": metrics["balanced_accuracy"],
            "ba_lower": metrics["ba_lower"],
            "ba_upper": metrics["ba_upper"],
            "auc": metrics["auc"],
            "auc_lower": metrics["auc_lower"],
            "auc_upper": metrics["auc_upper"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
        },
        limitations=LIMITATIONS,
    )

    directory = save_artifact(model, metadata, ARTIFACT_DIR)
    print(f"Saved artifact -> {directory}")
    _print_coefficients(model)


def _print_coefficients(model: Pipeline) -> None:
    """Standardized coefficients -- the exact basis of every prediction,
    and what the dashboard's explanation panel will display."""
    coefficients = model.named_steps["clf"].coef_.ravel()
    ranked = sorted(
        zip(FEATURE_COLUMNS, coefficients, strict=True),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )
    print("\nStandardized coefficients (positive pushes toward 'demented'):")
    for name, value in ranked:
        print(f"  {name:<24} {value:+.3f}")


if __name__ == "__main__":
    main()
