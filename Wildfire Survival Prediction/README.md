# Wildfire Survival Prediction
**WiDS Global Datathon 2026** | Data Science Project Team 22

## Overview
This project predicts the cumulative probability of a wildfire reaching an evacuation zone within 12h, 24h, 48h, and 72h, using only the first 5 hours of wildfire observations.

Rather than framing this as a binary classification task, we approached it as a **survival analysis (time-to-event)** problem — modeling *when* a fire is likely to arrive, not just *whether* it will.

## Dataset
- 221 wildfire incidents (WiDS Datathon 2026)
- 34 numerical input features based on initial 5-hour observations
- Event rate: 31.2% (69 events) / Right-censored: 68.8% (152 cases)
- Data available at: [WiDS Datathon 2026 on Kaggle](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26/data)

## Approach

### Feature Engineering & Preprocessing
- Applied log1p transformation to the distance-to-evacuation-zone variable to reduce skewness
- Protected the transformed distance variable from correlation-based removal
- Model-specific preprocessing: StandardScaler for Lasso-Cox, no scaling for RSF (tree-based)
- All preprocessing fitted on training folds only to prevent data leakage

### Validation Strategy
- Stratified 5-Fold Out-of-Fold (OOF) cross-validation
- Stratified by event rate (31%) to maintain consistent class distribution across folds

### Probability Extraction
- Extracted time-horizon probabilities from survival function: P(T≤H) = 1 − S(H)
- Applied monotonicity correction to ensure P(T≤12h) ≤ P(T≤24h) ≤ P(T≤48h) ≤ P(T≤72h)

### Models
| Model | Description |
|---|---|
| **Lasso-Cox** | Baseline survival model with L1 regularization; interpretable coefficients |
| **Random Survival Forest** | Nonlinear ensemble model; captures variable interactions |
| **Ensemble** | Weighted blend: 0.8 × RSF + 0.2 × Cox |

## Results

### Evaluation Metric
```
Hybrid Score = 0.3 × C-index + 0.7 × (1 − Weighted Brier)
Weighted Brier = 0.3 × @24h + 0.4 × @48h + 0.3 × @72h
```

### Performance Comparison
| Model | OOF C-index | Weighted Brier | Hybrid Score | Public Score |
|---|---|---|---|---|
| Lasso-Cox | 0.9251 | 0.0379 | 0.9510 | 0.9358 |
| RSF | 0.9423 | 0.0143 | 0.9727 | 0.9623 |
| **Ensemble** | **0.9435** | **0.0156** | **0.9722** | **0.9632** |



## Repository Structure
| File | Description |
|---|---|
| `eda/wildfire-eda.ipynb` | EDA: Kaplan–Meier curves, log-rank tests, log-log plots for the proportional hazards check |
| `lasso_cox_baseline.ipynb` | Lasso-Cox baseline: fold-wise preprocessing, penalizer search, competition metric implementation |
| `random_survival_forest.ipynb` | Random Survival Forest: grid search and survival-function-based probability extraction |
| `ensemble.ipynb` | Final submission: 0.8 × RSF + 0.2 × Cox blend with monotonicity correction |
| `ensemble_oof_score.py` | OOF evaluation of the ensemble (uses the `*_oof_*.npy` files) |
| `cox_oof_*.npy`, `rsf_oof_*.npy` | Out-of-fold predicted probabilities and risk scores |
| `*_submission.csv` | Kaggle submission files for each model |
| `22조_Final.pdf`, `Wildfire Survival Prediction presentation.pdf` | Final report and presentation slides |

## How to Run
The competition data is not included in this repository. Download `train.csv`, `test.csv`, `sample_submission.csv` and `metaData.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26/data) and place them in this folder before running the notebooks. The EDA notebook was run on Kaggle and reads the data from `/kaggle/input/`.

## Team
| Name | Student ID |
|---|---|
| 홍태선 | 2021320111 |
| 배지언 | 2022390707 |
| 손충기 | 2024100120 |

2026 Spring Semester — Data Science Project, Team 22
