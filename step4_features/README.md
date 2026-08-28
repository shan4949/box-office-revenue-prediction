# Step 4 — Feature Engineering

## What it does
Builds the tabular feature matrix from columns that are **knowable before a film is released**.
Every feature has a clear pre-release justification (shown below).

| Feature group | Columns | Pre-release justification |
|---------------|---------|--------------------------|
| Budget | `log_budget_real` | Production budget is set during green-lighting |
| Technical | `runtime` | Locked before theatrical cut |
| Genre | `genre_action`, `genre_drama`, … (17 cols) | Genre tags are part of the pitch |
| Language | `lang_english`, `lang_fr`, … (11 cols) | Set at production |
| Release calendar | `release_month`, `release_dow` | Scheduled in advance |
| Holiday window | `holiday_flag` | Known from the distribution calendar |
| Franchise | `franchise_flag` | Sequel/spinoff status is known pre-release |
| Studio | `studio_disney`, `studio_warnerbros`, … (14 cols) | Production company is known |

### Key decisions
- **`log_budget_real`** uses `log1p` to handle edge cases and compress the right tail.
- **`runtime`** missing values filled with the median (~107 min) — a neutral imputation.
- **Genre multi-hot**: top 17 genres from the TMDB taxonomy; a film can belong to multiple.
- **Holiday flag**: marks the ~2-week windows around Memorial Day, July 4th, Labor Day, Thanksgiving, and Christmas — the five peak Hollywood release weekends.
- **Studio**: maps the 400+ production companies down to 13 major studios + "Other".

## Inputs
```
step3_leakage/output/clean.parquet
```

## Outputs
```
step4_features/output/features.parquet   (same rows + engineered feature columns)
```

## Verification checks
| Check | Expected |
|-------|----------|
| No nulls in engineered features | ✓ (runtime filled, others derived) |
| No post-release feature included | ✓ (all sources listed above are pre-release) |
| Spot-check 5 rows | Inspect manually against release_date |

## How to run
```bash
../.venv/bin/python run.py
```
Runtime: ~15 seconds (genre/studio extraction per row).
