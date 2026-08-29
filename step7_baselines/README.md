# Step 7 — Baselines

## What it does
Establishes two simple baselines that the full LightGBM model must beat on both
rolling-CV pinball loss and the 2016 holdout.

| Baseline | Description |
|----------|-------------|
| **LR (log-budget)** | OLS linear regression using `log_budget_real` as the sole feature |
| **Genre-mean** | For each film, predict the training-set mean `log_revenue_real` for its genres |

### Why these two?
- **LR on budget** is the natural starting point: budget is the single strongest known
  predictor of box-office revenue. Any model that can't beat this is useless.
- **Genre-mean** captures domain knowledge (Horror films earn differently from Animation)
  without using any numeric features. It's a tougher baseline than the global mean.

### Primary metric: Pinball loss at P50
Pinball (quantile) loss at α = 0.5 equals `0.5 × MAE`. It is the correct
loss for a P50 (median) forecast and maps directly to the LightGBM objective used in
Step 8. Lower is better.

```
pinball_P50 = mean(max(0.5 × (y_true − ŷ), 0.5 × (ŷ − y_true)))
```

### Rolling-CV protocol
Both baselines are re-fitted on each expanding training window.
The genre means are computed **within each fold's training set** to avoid leakage.

## Inputs
```
step6_split/output/modeling_df.parquet
step6_split/output/splits.json
```

## Outputs
```
step7_baselines/output/cv_baseline_metrics.csv     per-fold R² and pinball for both baselines
step7_baselines/output/test_baseline_metrics.json  2016 holdout metrics
```

## Verification checks
| Check | Expected |
|-------|----------|
| LR CV mean R² | 0.3–0.5 (budget explains 30–50% of log-revenue variance) |
| Genre CV mean R² | 0.1–0.3 |
| LightGBM (Step 8) pinball < LR pinball | Required |

## How to run
```bash
../.venv/bin/python run.py
```
Runtime: ~30 seconds (20 folds × 2 baselines, small dataset).
