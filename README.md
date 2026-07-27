# ScanAlzheimer

Subject-level dementia screening from structural brain MRI (OASIS-1),
with an emphasis on honest evaluation and clean, swappable architecture.

> ⚠️ Research/portfolio prototype — **not a medical device.**
> Not validated for any clinical use.

## Status
��� In development — Stage 0 (project scaffolding) complete.

## Task
Binary classification of **CDR = 0 (cognitively normal)** vs
**CDR ≥ 0.5 (very mild to moderate dementia)** at the **subject level**.

Note: labels are clinical dementia ratings, not biomarker-confirmed
Alzheimer's pathology.

## Evaluation policy
This project treats evaluation integrity as a first-class concern:

- Splits are **subject-level** (`StratifiedGroupKFold`, `groups=subject_id`).
  Slice-level splitting on brain MRI has been shown to inflate accuracy by
  ~30 percentage points and can yield ~96% accuracy on randomly shuffled labels.
- Control subjects are **age-restricted to ≥ 60** to avoid the model learning
  brain age instead of atrophy — OASIS-1 controls span ages 18–96 while all
  demented subjects are 60+.
- Metrics are reported with bootstrap confidence intervals.
- Modest, honest numbers are preferred over inflated ones.

## Data
OASIS-1: https://sites.wustl.edu/oasisbrains/

Data were provided by OASIS: Marcus, DS, Wang, TH, Parker, J, Csernansky, JG,
Morris, JC, Buckner, RL. "Open Access Series of Imaging Studies (OASIS):
Cross-Sectional MRI Data in Young, Middle Aged, Nondemented, and Demented
Older Adults." Journal of Cognitive Neuroscience, 19, 1498-1507, 2007.

Data and trained models are **not** committed to this repository.

## Setup
```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -e ".[dev]"
pytest
```

## Known limitations
- (to be documented as the project develops)
cat > README.md << 'EOF'
# ScanAlzheimer

Subject-level dementia screening from structural brain MRI (OASIS-1),
with an emphasis on honest evaluation and clean, swappable architecture.

> ⚠️ Research/portfolio prototype — **not a medical device.**
> Not validated for any clinical use.

## Status
��� In development — Stage 0 (project scaffolding) complete.

## Task
Binary classification of **CDR = 0 (cognitively normal)** vs
**CDR ≥ 0.5 (very mild to moderate dementia)** at the **subject level**.

Note: labels are clinical dementia ratings, not biomarker-confirmed
Alzheimer's pathology.

## Evaluation policy
This project treats evaluation integrity as a first-class concern:

- Splits are **subject-level** (`StratifiedGroupKFold`, `groups=subject_id`).
  Slice-level splitting on brain MRI has been shown to inflate accuracy by
  ~30 percentage points and can yield ~96% accuracy on randomly shuffled labels.
- Control subjects are **age-restricted to ≥ 60** to avoid the model learning
  brain age instead of atrophy — OASIS-1 controls span ages 18–96 while all
  demented subjects are 60+.
- Metrics are reported with bootstrap confidence intervals.
- Modest, honest numbers are preferred over inflated ones.

## Data
OASIS-1: https://sites.wustl.edu/oasisbrains/

Data were provided by OASIS: Marcus, DS, Wang, TH, Parker, J, Csernansky, JG,
Morris, JC, Buckner, RL. "Open Access Series of Imaging Studies (OASIS):
Cross-Sectional MRI Data in Young, Middle Aged, Nondemented, and Demented
Older Adults." Journal of Cognitive Neuroscience, 19, 1498-1507, 2007.

Data and trained models are **not** committed to this repository.

## Setup
```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -e ".[dev]"
pytest
```

## Known limitations
- (to be documented as the project develops)
