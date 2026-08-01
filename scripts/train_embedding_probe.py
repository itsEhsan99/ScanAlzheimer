"""Linear probe over frozen backbone embeddings, evaluated with the same
subject-level cross-validation and bootstrap metrics as the tabular
baseline, so the two are directly comparable.
"""

import argparse
from pathlib import Path

import pandas as pd

from ScanAlzheimer.data.paths import attach_image_paths
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
from ScanAlzheimer.evaluation.splits import assert_no_subject_leakage
from ScanAlzheimer.features.embeddings import (
    embed_cohort,
    embedding_columns,
    load_embeddings,
    save_embeddings,
    select_device,
)
from ScanAlzheimer.inference.aggregate import aggregate_predictions
from ScanAlzheimer.models.baseline import build_baseline_model
from ScanAlzheimer.models.registry import build_backbone, freeze, get_spec

FOLDS_PATH = Path("data/metadata/manifest_folds.csv")
DATA_ROOT = Path("data/raw")
ARTIFACT_DIR = Path("artifacts/embeddings")

BASELINE_BA = 0.682
BASELINE_AUC = 0.756


def main() -> None:
    args = _parse_args()
    spec = get_spec(args.backbone)
    cache = ARTIFACT_DIR / f"{args.backbone}_{args.mode}_axis{args.axis}_n{args.n_slices}.parquet"

    if cache.exists() and not args.rebuild:
        print(f"Loading cached embeddings from {cache}")
        slices = load_embeddings(cache)
    else:
        slices = _build_embeddings(args, spec, cache)

    assert_no_subject_leakage(slices)
    features = embedding_columns(slices)

    print(f"\nBackbone : {args.backbone}  ({spec.description})")
    print(f"Mode     : {args.mode}, axis {args.axis}, {args.n_slices} slices/subject")
    print(f"Slices   : {len(slices)} rows from {slices['subject_id'].nunique()} subjects")
    print(f"Embedding: {len(features)} dimensions")
    print("Leakage guard: PASSED")

    predictions = run_cross_validation(slices, features, _probe_factory)
    subjects = aggregate_predictions(predictions, method=args.aggregation)
    _report(f"{args.backbone} / {args.mode}", subjects)

    print("\n" + "=" * 60)
    print("NEGATIVE CONTROL -- labels shuffled")
    print("=" * 60)
    control = run_cross_validation(_shuffle_by_subject(slices), features, _probe_factory)
    _report("shuffled", aggregate_predictions(control, method=args.aggregation))
    print("\nNear 0.5 is correct. Anything higher means something leaks.")

    _compare_to_baseline(subjects)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument(
        "--mode", default="replicate", choices=["replicate", "adjacent", "triplanar"]
    )
    parser.add_argument("--axis", type=int, default=1, help="0=sagittal, 1=coronal, 2=axial")
    parser.add_argument("--n-slices", type=int, default=5)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--aggregation", default="mean", choices=["mean", "median", "max"])
    parser.add_argument("--rebuild", action="store_true", help="ignore cached embeddings")
    return parser.parse_args()


def _build_embeddings(args, spec, cache: Path) -> pd.DataFrame:
    cohort = pd.read_csv(FOLDS_PATH)
    cohort = attach_image_paths(cohort, DATA_ROOT, variant="t88_masked_gfc")
    cohort = cohort[cohort["image_available"]].reset_index(drop=True)

    device = select_device()
    print(f"Device: {device}")
    print(f"Embedding {len(cohort)} subjects with {args.backbone}...")

    model = freeze(build_backbone(args.backbone, pretrained=True))
    slices = embed_cohort(
        cohort,
        model,
        spec,
        axis=args.axis,
        n_slices=args.n_slices,
        step=args.step,
        mode=args.mode,
        device=device,
    )
    save_embeddings(slices, cache)
    print(f"Saved embeddings -> {cache}")
    return slices


def _probe_factory():
    """A linear probe is the same model as the tabular baseline, so any
    difference in score comes from the features, not the classifier."""
    return build_baseline_model(c=0.01)


def _shuffle_by_subject(slices: pd.DataFrame) -> pd.DataFrame:
    """Shuffle labels at subject level, keeping each subject's slices consistent."""
    subjects = slices[["subject_id", "label"]].drop_duplicates()
    shuffled = shuffle_labels(subjects, seed=0)
    mapping = dict(zip(shuffled["subject_id"], shuffled["label"], strict=True))

    result = slices.copy()
    result["label"] = result["subject_id"].map(mapping)
    return result


def _report(name: str, subjects: pd.DataFrame) -> None:
    y_true = subjects["label"].to_numpy()
    y_pred = subjects["y_pred"].to_numpy()
    y_score = subjects["y_score"].to_numpy()

    report = classification_report(y_true, y_pred, y_score)
    ba, ba_lo, ba_hi = bootstrap_ci(y_true, y_pred, balanced_accuracy)
    auc_point, auc_lo, auc_hi = bootstrap_ci(y_true, y_score, auc)

    print(f"\n--- {name}  (subject level, n={len(subjects)}) ---")
    print(f"  Balanced accuracy : {format_ci(ba, ba_lo, ba_hi)}")
    print(f"  AUC               : {format_ci(auc_point, auc_lo, auc_hi)}")
    print(f"  Sensitivity       : {report['sensitivity']:.3f}")
    print(f"  Specificity       : {report['specificity']:.3f}")
    print(f"  Beats chance      : {'YES' if beats_chance(ba_lo) else 'NO'}")

    tn, fp, fn, tp = confusion(y_true, y_pred).ravel()
    print(f"  Confusion         : TN={tn} FP={fp} FN={fn} TP={tp}")


def _compare_to_baseline(subjects: pd.DataFrame) -> None:
    y_true = subjects["label"].to_numpy()
    ba, _, _ = bootstrap_ci(y_true, subjects["y_pred"].to_numpy(), balanced_accuracy)
    auc_point, _, _ = bootstrap_ci(y_true, subjects["y_score"].to_numpy(), auc)

    print("\n" + "=" * 60)
    print("VERSUS TABULAR BASELINE (tissue + age)")
    print("=" * 60)
    print(f"  Balanced accuracy : {ba:.3f}  vs  {BASELINE_BA:.3f}  ({ba - BASELINE_BA:+.3f})")
    print(
        f"  AUC               : {auc_point:.3f}  vs  {BASELINE_AUC:.3f}  "
        f"({auc_point - BASELINE_AUC:+.3f})"
    )
    print("\nSix tissue numbers are a serious baseline. If the embeddings do not")
    print("beat them, the added complexity is not yet earning its place.")


if __name__ == "__main__":
    main()
