# Step 0 — Ingest + Clean

## What it does
Loads the three raw Kaggle TMDB CSV files and joins them into one clean dataframe.

| File | Rows (raw) | Role |
|------|-----------|------|
| `movies_metadata.csv` | ~45,466 | Title, budget, revenue, genres, dates |
| `credits.csv` | ~45,476 | Cast & crew JSON per movie |
| `keywords.csv` | ~46,419 | Plot keywords per movie |

Key operations:
1. **Coerce `id` to integer** — `pd.to_numeric(..., errors='coerce')` then drop NaN.  
   This silently removes 3 rows whose `id` field contains a date string or float text instead of a valid TMDB integer.
2. **Inner-join** all three files on `id`.
3. **Parse literal-eval columns** (`genres`, `production_companies`, `cast`, `crew`, `keywords`, `belongs_to_collection`) from raw string representation into Python objects.
4. **Re-serialise as JSON strings** before writing to parquet.  
   Parquet + pyarrow can store lists of structs, but JSON strings are simpler and more portable across readers.

## Inputs
```
Data1/movies_metadata.csv
Data1/credits.csv
Data1/keywords.csv
```

## Outputs
```
step0_ingest/output/merged.parquet   (~45,463 rows)
```

## Verification checks
| Check | Expected |
|-------|----------|
| Rows after merge | ~45,463 |
| Malformed rows dropped | 3 |

## How to run
```bash
../.venv/bin/python run.py
```
Runtime: ~30 seconds (most time is parsing 45K literal-eval strings).
