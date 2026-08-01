"""Logistic-regression baseline over tabular features.

Wrapped in a scikit-learn Pipeline so that feature scaling is fitted on
the training fold only. Scaling before cross-validation would leak test-set
statistics into training -- a subtler cousin of subject-level leakage, and
just as capable of inflating results.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TISSUE_FEATURES = [
    "grey_matter_fraction",
    "csf_fraction",
    "white_matter_fraction",
    "gm_csf_ratio",
    "gm_wm_ratio",
    "brain_csf_ratio",
]

AGE_FEATURES = ["age"]

FEATURE_SETS: dict[str, list[str]] = {
    "age_only": AGE_FEATURES,
    "tissue_only": TISSUE_FEATURES,
    "tissue_plus_age": TISSUE_FEATURES + AGE_FEATURES,
    # Deployed model: brain/CSF ratio is a direct atrophy index, and the
    # remaining tissue features are collinear with it (r up to 0.97), which
    # both hurt accuracy and flipped coefficient signs.
    "atrophy_plus_age": ["brain_csf_ratio"] + AGE_FEATURES,
}


def build_baseline_model(c: float = 1.0, seed: int = 42) -> Pipeline:
    """Standardize features, then fit a regularized logistic regression.

    `class_weight="balanced"` keeps the model from favouring the majority
    class in folds where the split is slightly uneven.
    """
    if c <= 0:
        raise ValueError(f"Regularization strength c must be positive; got {c}")

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=c,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def feature_columns(name: str) -> list[str]:
    """Look up a named feature set."""
    if name not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {name!r}. Available: {sorted(FEATURE_SETS)}")
    return list(FEATURE_SETS[name])
