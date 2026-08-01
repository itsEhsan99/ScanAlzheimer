# ScanAlzheimer

Subject-level dementia screening from structural brain MRI (OASIS-1),
with an emphasis on honest evaluation and clean, swappable architecture.

> ⚠️ Research/portfolio prototype — **not a medical device.**
> Not validated for any clinical use.

## Status
🚧 In development — Stage 4 (tabular baseline) complete.

## Task
Binary classification of **CDR = 0 (cognitively normal)** vs
**CDR ≥ 0.5 (very mild to moderate dementia)** at the **subject level**.

Labels are clinical dementia ratings, not biomarker-confirmed Alzheimer's
pathology. 70% of the positive class is CDR 0.5 — the earliest, subtlest
stage — which makes this task harder than the AD-vs-healthy comparisons
that many published accuracy figures are based on.

## Cohort
| | |
|---|---|
| Source | OASIS-1, 436 sessions |
| With CDR labels | 235 subjects |
| Age-matched cohort | **198 subjects** (98 CN / 100 demented) |
| Validation | 5-fold `StratifiedGroupKFold`, grouped by subject |

Controls are restricted to age ≥ 60. OASIS-1 controls span ages 18–96 while
all demented subjects are 60+, so an unfiltered comparison would let a model
score highly by detecting brain age rather than atrophy.

## Baseline results
Tabular baseline: logistic regression over FSL tissue-volume fractions,
evaluated with out-of-fold subject-level cross-validation. Intervals are
95% bootstrap percentiles over subjects.

| Model | Balanced accuracy | AUC |
|---|---|---|
| Age only | 0.480 [0.412, 0.552] | 0.466 [0.383, 0.547] |
| Tissue only | 0.656 [0.590, 0.722] | 0.700 [0.625, 0.769] |
| Tissue + age | **0.682 [0.614, 0.747]** | **0.756 [0.685, 0.823]** |
| *Shuffled labels (control)* | *0.496 [0.428, 0.565]* | *0.454 [0.373, 0.538]* |

**Age alone does not beat chance** (interval includes 0.5), confirming the
age-matching worked: whatever signal the tissue model finds is anatomical,
not demographic.

Adding age to tissue features improves ranking (ΔAUC 0.055 [0.014, 0.099],
paired bootstrap, excludes zero) but not thresholded decisions
(Δbalanced accuracy 0.025 [-0.034, 0.082], includes zero).

Sensitivity by severity: 0.67 at CDR 0.5 (n=70), 0.79 at CDR 1.0 (n=28).
Earlier disease is genuinely harder to detect.

## Evaluation policy
Evaluation integrity is treated as a first-class concern:

- **Subject-level splits** (`StratifiedGroupKFold`, `groups=subject_id`),
  enforced by a runtime guard that raises if any subject appears in more
  than one fold. Slice-level splitting on brain MRI has been shown to
  inflate accuracy by ~30 percentage points and to reach ~96% accuracy on
  randomly shuffled labels.
- **Negative control**: the full pipeline is re-run on shuffled labels and
  must collapse to chance. It does (0.496).
- **Age-matched controls**, to prevent brain age acting as a shortcut.
- **Bootstrap confidence intervals** on every headline number. A point
  estimate from ~200 subjects on its own is not informative.
- **Scaling inside the CV pipeline**, so no test-fold statistics leak into
  training.
- Modest, honest numbers are preferred over inflated ones.

## Known limitations
- Baseline features use OASIS-provided FSL segmentations, not a segmentation
  pipeline implemented here.
- Images are the atlas-registered, skull-stripped variant. Skull-stripping
  has been reported to induce shortcut learning in MRI-based Alzheimer's
  classification; comparing against the skull-intact variant is planned.
- No external validation yet. A held-out cohort (Mendeley `kcjt4v658x`,
  26 subjects, different scanner and country) is reserved for this.
- Mean age varies across folds by up to 4 years; folds are stratified by
  label only, not by age.
- The prediction is a research prototype and is not clinically validated.

## Setup
## Setup
```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash

# CPU-only PyTorch (the default wheel pulls ~2.5 GB of CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[dev]"
pytest
```
