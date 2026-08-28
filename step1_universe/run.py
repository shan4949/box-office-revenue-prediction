"""
Step 1: Define Modeling Universe
Filters merged dataset to movies with budget > 0, revenue > 0, release year 1995-2017.
"""
import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STEP0_OUT  = os.path.join(SCRIPT_DIR, '..', 'step0_ingest', 'output', 'merged.parquet')
OUT_DIR    = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("=== Step 1: Define Modeling Universe ===")

    df = pd.read_parquet(STEP0_OUT)
    print(f"  Input: {len(df):,} rows")

    # Numeric coercion
    df['budget']  = pd.to_numeric(df['budget'],  errors='coerce').fillna(0)
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')

    # Release date + year
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year']  = df['release_date'].dt.year

    # Filter
    mask = (
        (df['budget']  > 0) &
        (df['revenue'] > 0) &
        (df['release_year'] >= 1995) &
        (df['release_year'] <= 2017)
    )
    universe = df[mask].copy().reset_index(drop=True)
    print(f"  After filters: {len(universe):,} rows (expected ~3,991)")

    # Yearly distribution
    yearly = universe.groupby('release_year').size()
    print(f"\n[VERIFY] Yearly distribution:\n{yearly.to_string()}")

    min_count = int(yearly.min())
    print(f"\n[VERIFY] Min yearly count = {min_count} (must be ≥ 50)")
    assert min_count >= 50, f"Year {yearly.idxmin()} has only {min_count} rows — too small for a fold"

    print(f"\n[VERIFY] N = {len(universe):,}  |  years {int(yearly.index.min())}–{int(yearly.index.max())}")

    out_path = os.path.join(OUT_DIR, 'universe.parquet')
    universe.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
