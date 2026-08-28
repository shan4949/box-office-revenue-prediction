# Step 2 — CPI Deflation

## What it does
Converts nominal budget and revenue figures into **constant 2017 US dollars** using the
BLS Consumer Price Index for All Urban Consumers (CPI-U, series CPIAUCSL) fetched live
from the Federal Reserve Economic Data (FRED) API.

### Formula
```
real_value = nominal_value × (CPI_2017 / CPI_release_year)
```

Annual CPI is the simple average of the 12 monthly readings for that year.

## Why this matters
A $100 M budget in 1995 is not the same as $100 M in 2017.
Without deflation the model would confuse era-level dollar inflation with true scale.
Deflating to a common base year lets budget be compared across the full 1995–2017 window.

## Data source
- **Live**: `https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL` (public, no API key)
- **Fallback**: hard-coded annual averages in the script (used if network is unavailable)

## Inputs
```
step1_universe/output/universe.parquet
```

## Outputs
```
step2_cpi/output/deflated.parquet    (adds budget_real, revenue_real, cpi_deflator)
step2_cpi/output/cpi_table.csv       (year, cpi, deflator_to_2017 — reference table)
```

## Verification checks
| Check | Expected |
|-------|----------|
| 1995 deflator | ~1.5–1.6 |
| 2015 deflator | ~1.01–1.04 |
| 2017 deflator | exactly 1.000 |
| Missing deflators | 0 rows |

## How to run
```bash
../.venv/bin/python run.py
```
Runtime: < 5 seconds (plus a short network request to FRED).
