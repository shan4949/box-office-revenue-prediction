# Step 1 — Define Modeling Universe

## What it does
Applies three hard filters to the 45 K merged dataset, leaving only films where both
budget and revenue were actually reported and the release year falls in the study window.

| Filter | Rationale |
|--------|-----------|
| `budget > 0` | Zero/missing budget = unmodellable |
| `revenue > 0` | Zero/missing revenue = unmodellable target |
| `release_year ∈ [1995, 2017]` | Pre-1995 sparse; post-2017 partial year |

## Inputs
```
step0_ingest/output/merged.parquet
```

## Outputs
```
step1_universe/output/universe.parquet   (~3,991 rows)
```

## Verification checks
| Check | Expected |
|-------|----------|
| Total rows | ~3,991 |
| Min rows in any single year | ≥ 50 (ensures every CV fold is non-trivial) |
| Year range | 1995–2017 |

## How to run
```bash
../.venv/bin/python run.py
```
Runtime: < 5 seconds.
