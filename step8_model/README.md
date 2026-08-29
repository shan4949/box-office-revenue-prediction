# Step 8 — LightGBM Quantile Model

## What it does
Trains three LightGBM regressors simultaneously — P10, P50, P90 — using the
**quantile objective** (`objective='quantile'`, `alpha=τ`).

Instead of a single point estimate, the model produces a **prediction interval** for
each film: the P10–P90 interval captures the uncertainty range; P50 is the primary
point forecast.

### Why quantile regression?
Box-office revenue has a heavy right tail (a few blockbusters dominate the distribution).
Quantile regression is robust to this and directly optimises the pinball loss that the
project reports as its headline metric.

### Feature set (input to model)

| Group | Features |
|-------|----------|
| Budget | `log_budget_real` |
| Runtime | `runtime` |
| Genres | `genre_action`, `genre_drama`, … (17 cols) |
| Language | `lang_english`, `lang_fr`, … (11 cols) |
| Calendar | `release_month`, `release_dow`, `holiday_flag` |
| Franchise | `franchise_flag` |
| Studio | `studio_disney`, `studio_warnerbros`, … (14 cols) |
| Talent | `log_star_power`, `is_debut` |

NaN in `log_star_power` (debut talent) is handled natively by LightGBM's split criterion —
no imputation needed.

### Hyperparameters
```python
n_estimators=600, learning_rate=0.03, num_leaves=63,
min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
reg_alpha=0.1, reg_lambda=1.0
```
No early stopping — fixed for reproducibility across 20 × 3 = 60 model fits.

### CV protocol
Same 20 expanding-window folds from Step 6.
Models **not saved** for CV folds (only final-holdout models are persisted).

### Model serialisation
Models are saved in **LightGBM's native text format** (`.txt`) via
`booster_.save_model()` — not pickle. This format is safe, portable, and
version-stable across LightGBM releases.

## Inputs
```
step6_split/output/modeling_df.parquet
step6_split/output/splits.json
```

## Outputs
```
step8_model/output/models/p10.txt              Final P10 model
step8_model/output/models/p50.txt              Final P50 model
step8_model/output/models/p90.txt              Final P90 model
step8_model/output/models/feature_cols.json    Ordered feature list
step8_model/output/cv_fold_metrics.csv         pinball + coverage per CV fold
step8_model/output/test_metrics.json           2016 holdout pinball + coverage
step8_model/output/test_predictions.parquet    Per-film P10/P50/P90 for 2016
```

## Verification checks
| Check | Expected |
|-------|----------|
| pinball_p50 (2016) < LR baseline pinball | Required |
| P10–P90 coverage (2016) | ~0.75–0.85 (targeting 0.80) |
| SHAP top feature | `log_budget_real` (confirmed in Step 9) |

## How to run
```bash
../.venv/bin/python run.py
```
Runtime: ~5–10 minutes (20 folds × 3 quantiles = 60 model fits).
