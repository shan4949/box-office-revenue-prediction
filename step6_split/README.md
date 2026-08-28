# Step 6 — Temporal Split

## What it does
Creates the **expanding-window rolling-origin** cross-validation folds and the
final held-out test set.

### Why expanding windows, not sliding windows?
Box-office revenue depends on market conditions (inflation, platform competition, genre
trends) that evolve over time. An expanding window mimics the real-world scenario where
a forecaster has **all available history** up to the prediction date, not just a fixed
look-back period.

### Fold structure
```
Fold  1: train 1995        → val 1996
Fold  2: train 1995–1996   → val 1997
…
Fold 20: train 1995–2014   → val 2015
Final  : train 1995–2015   → test 2016  ← held-out, never used during CV
```

2017 is excluded from the test set because the Kaggle snapshot only contains
71 rows for that partial year — too few for a stable metric.

### Key invariant
For every fold: `min(val release_year) > max(train release_year)`.
The script asserts this and raises an error if violated.

## Inputs
```
step5_starpower/output/with_starpower.parquet
```

## Outputs
```
step6_split/output/splits.pkl           Python dict with 'folds' (list) and 'test' (dict)
step6_split/output/modeling_df.parquet  Full feature matrix with integer row index
```

### `splits.pkl` schema
```python
{
  'folds': [
    {'val_year': 1996, 'train_idx': [...], 'val_idx': [...], 'n_train': int, 'n_val': int},
    …
  ],
  'test': {'train_idx': [...], 'test_idx': [...], 'n_train': int, 'n_test': int}
}
```

## Verification checks
| Check | Expected |
|-------|----------|
| Number of CV folds | 20 |
| min(val_year) > max(train_year) for every fold | ✓ |
| Test year (2016) > max train year (2015) | ✓ |

## How to run
```bash
../.venv/bin/python run.py
```
Runtime: < 5 seconds.
