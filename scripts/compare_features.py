"""Compare feature sets on the same subject-level splits.

The headline question is not whether embeddings beat tissue features on
their own, but whether they carry information the tissue features lack.
Only a combined model can answer that.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ScanAlzheimer.evaluation.crossval import run_cross_validation
from ScanAlzheimer.evaluation.metrics import (
    auc,
    balanced_accuracy,
    bootstrap_ci,
    format_ci,
)
from ScanAlzheimer.features.embeddings import embedding_columns, load_embeddings
from ScanAlzheimer.inference.aggregate import aggregate_predictions
from ScanAlzheimer.models.baseline import feature_columns

FEATURES_PATH = Path("data/metadata/features_tissue.csv")
TISSUE_COLUMNS = feature_columns("tissue_plus_age")


def main() -> None:
    args = _parse_args()
    slices = load_embeddings(Path(args.embeddings))
    emb_cols = embedding_columns(slices)

    tissue = pd.read_csv(FEATURES_PATH)
    merged = slices.merge(
        tissue[["subject_id", *[c for c in TISSUE_COLUMNS if c != "age"]]],
        on="subject_id",
        how="left",
        validate="many_to_one",
    )

    print(f"Slices: {len(merged)} from {merged['subject_id'].nunique()} subjects")
    print(f"Embedding dims: {len(emb_cols)}   Tissue features: {len(TISSUE_COLUMNS)}\n")

    rows = []
    for c in args.c_values:
        rows.append(_evaluate(f"tissue+age (C={c})", merged, TISSUE_COLUMNS, c))
        rows.append(_evaluate(f"embeddings (C={c})", merged, emb_cols, c))
        rows.append(_evaluate(f"emb PCA-{args.pca} (C={c})", merged, emb_cols, c, pca=args.pca))
        rows.append(
            _evaluate(
                f"tissue + emb PCA-{args.pca} (C={c})",
                merged,
                TISSUE_COLUMNS + emb_cols,
                c,
                pca=args.pca,
                passthrough=len(TISSUE_COLUMNS),
            )
        )

    summary = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    print("\n" + "=" * 74)
    print("RANKED RESULTS (subject level, out-of-fold)")
    print("=" * 74)
    print(summary.to_string(index=False))

    out = Path("data/metadata/feature_comparison.csv")
    summary.to_csv(out, index=False)
    print(f"\nSaved -> {out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings",
        default="artifacts/embeddings/resnet18_adjacent_axis1_n5.parquet",
    )
    parser.add_argument("--pca", type=int, default=32)
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0])
    return parser.parse_args()


def _make_factory(c: float, pca: int | None, passthrough: int):
    """Build a probe, optionally compressing the embedding block with PCA.

    PCA is fitted inside the pipeline, so it only ever sees training-fold
    data -- fitting it beforehand would leak test-fold structure.
    """

    def factory():
        steps = [("scaler", StandardScaler())]
        if pca is not None:
            steps.append(("pca", _BlockPCA(pca, passthrough)))
        steps.append(("clf", LogisticRegression(C=c, max_iter=2000, class_weight="balanced")))
        return Pipeline(steps)

    return factory


class _BlockPCA:
    """Apply PCA to the trailing embedding columns, leaving the leading
    tissue columns untouched, so a handful of interpretable features are
    not drowned out by hundreds of compressed ones."""

    def __init__(self, n_components: int, passthrough: int = 0):
        self.n_components = n_components
        self.passthrough = passthrough
        self.pca = PCA(n_components=n_components, random_state=42)

    def fit(self, x, y=None):
        self.pca.fit(x[:, self.passthrough :])
        return self

    def transform(self, x):
        import numpy as np

        compressed = self.pca.transform(x[:, self.passthrough :])
        if self.passthrough == 0:
            return compressed
        return np.hstack([x[:, : self.passthrough], compressed])

    def fit_transform(self, x, y=None):
        return self.fit(x, y).transform(x)

    def get_params(self, deep=True):
        return {"n_components": self.n_components, "passthrough": self.passthrough}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def _evaluate(
    name: str,
    frame: pd.DataFrame,
    columns: list[str],
    c: float,
    pca: int | None = None,
    passthrough: int = 0,
) -> dict:
    predictions = run_cross_validation(frame, columns, _make_factory(c, pca, passthrough))
    subjects = aggregate_predictions(predictions, method="mean")

    y_true = subjects["label"].to_numpy()
    ba, ba_lo, ba_hi = bootstrap_ci(y_true, subjects["y_pred"].to_numpy(), balanced_accuracy)
    auc_point, auc_lo, auc_hi = bootstrap_ci(y_true, subjects["y_score"].to_numpy(), auc)

    print(f"  {name:<34} BA {format_ci(ba, ba_lo, ba_hi)}   AUC {auc_point:.3f}")

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
