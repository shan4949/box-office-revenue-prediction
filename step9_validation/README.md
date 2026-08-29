# Step 9 — Validation Report

## What it does
Assembles the final deliverable: a complete validation report and a JSON file of
resume-ready metrics.

### Three things it produces

1. **Per-fold CV table** — pinball P50 + coverage for LightGBM vs both baselines,
   across all 20 rolling-origin folds. This is more honest than a single holdout number
   because it shows whether the model is consistently better or just lucky on 2016.

2. **SHAP feature importance** — computed on the 2016 test set using the P50 model.
   - Confirms that `log_budget_real` dominates (expected).
   - Acts as a **leakage check**: if any post-release feature (vote_count, popularity, etc.)
     appears in the top 10, the pipeline failed Step 3. The script raises `RuntimeError`
     if this happens.

3. **Resume-ready summary** — the three numbers you need for a portfolio or CV.

## Key metrics (what to put on a resume)

```
Pinball P50 (2016 holdout)  — primary loss metric; lower is better
  LightGBM model      :  see resume_metrics.json
  log-budget-only LR  :  baseline to beat
  genre-mean          :  second baseline to beat

P10–P90 empirical coverage  — fraction of actual revenues inside the predicted interval
  Target              :  ~80%  (well-calibrated 80% prediction intervals)

N = 3,991 films | train 1995–2015 | test 2016
Model: LightGBM Quantile Regression, ~40 pre-release features
```

## Inputs
```
step6_split/output/modeling_df.parquet
step6_split/output/splits.json
step7_baselines/output/cv_baseline_metrics.csv
step7_baselines/output/test_baseline_metrics.json
step8_model/output/models/p50.txt
step8_model/output/models/feature_cols.json
step8_model/output/cv_fold_metrics.csv
step8_model/output/test_metrics.json
```

## Outputs
```
step9_validation/output/validation_report.md    Full Markdown report
step9_validation/output/resume_metrics.json     Machine-readable summary of key numbers
```

## Verification checks
| Check | Expected |
|-------|----------|
| LightGBM P50 < LR baseline on 2016 holdout | Required |
| LightGBM P50 < genre baseline on 2016 holdout | Required |
| P10–P90 coverage | 0.75–0.85 |
| Top SHAP feature | `log_budget_real` |
| No post-release feature in top 10 SHAP | ✓ (script raises error if violated) |

## How to run
```bash
../.venv/bin/python run.py
```
Runtime: ~30 seconds (SHAP computation on 2016 test set).
