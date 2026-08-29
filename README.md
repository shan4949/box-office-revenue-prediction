# Box Office Revenue Prediction

End-to-end quantile regression pipeline that predicts theatrical revenue for films using
only information available **before release**.

## Quick start

```bash
# 1. Create virtual environment and install dependencies
python3 -m venv .venv
.venv/bin/pip install pandas numpy lightgbm scikit-learn shap requests

# 2. Run the full pipeline (~10–15 minutes)
bash run_pipeline.sh

# 3. View results
cat step9_validation/output/validation_report.md
cat step9_validation/output/resume_metrics.json
```

## Pipeline overview

```
Data1/
  movies_metadata.csv
  credits.csv
  keywords.csv
         │
         ▼
step0_ingest/      Merge + parse 3 raw CSVs → merged.parquet (45K rows)
         │
         ▼
step1_universe/    Filter budget>0, revenue>0, 1995–2017 → ~3,991 rows
         │
         ▼
step2_cpi/         CPI-deflate to 2017 USD (FRED CPIAUCSL)
         │
         ▼
step3_leakage/     Drop post-release columns (votes, popularity, etc.)
         │
         ▼
step4_features/    Build pre-release feature matrix (budget, genres, studio, calendar…)
         │
         ▼
step5_starpower/   Compute career-to-date revenue for lead cast + director
         │
         ▼
step6_split/       Expanding-window rolling-origin CV (20 folds) + 2016 holdout
         │
         ▼
step7_baselines/   LR + genre-mean baselines (must be beaten by step 8)
         │
         ▼
step8_model/       LightGBM quantile (P10/P50/P90) — CV + final holdout
         │
         ▼
step9_validation/  Validation report + SHAP leakage check + resume metrics
```

Each step folder contains:
- `run.py` — the executable script
- `README.md` — what it does, inputs/outputs, verification checks
- `output/` — generated artifacts (created on first run)

## Deliverable metrics (from `step9_validation/output/resume_metrics.json`)

| Metric | Description |
|--------|-------------|
| `pinball_p50_test` | Primary loss on 2016 holdout; compare against baselines |
| `lr_baseline_pinball_p50` | Log-budget-only linear regression baseline |
| `genre_baseline_pinball_p50` | Genre-mean baseline |
| `coverage_p10_p90_test` | Fraction of actuals inside P10–P90 interval (target ~80%) |
| `cv_mean_pinball_p50` | Mean pinball across 20 rolling-origin folds |
| `top10_shap_features` | Confirms budget dominates and no leakage sneaked through |

## Data

The pipeline expects the Kaggle TMDB 5000 / MovieLens dataset in `Data1/`:
- `movies_metadata.csv`
- `credits.csv`
- `keywords.csv`

## Step-by-step run (if running individually)

```bash
cd step0_ingest   && ../.venv/bin/python run.py && cd ..
cd step1_universe && ../.venv/bin/python run.py && cd ..
cd step2_cpi      && ../.venv/bin/python run.py && cd ..
cd step3_leakage  && ../.venv/bin/python run.py && cd ..
cd step4_features && ../.venv/bin/python run.py && cd ..
cd step5_starpower && ../.venv/bin/python run.py && cd ..
cd step6_split    && ../.venv/bin/python run.py && cd ..
cd step7_baselines && ../.venv/bin/python run.py && cd ..
cd step8_model    && ../.venv/bin/python run.py && cd ..
cd step9_validation && ../.venv/bin/python run.py && cd ..
```
