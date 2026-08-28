"""
Step 2: CPI Deflation
Fetches CPI-U (CPIAUCSL) from FRED and deflates budget + revenue to constant 2017 dollars.
Falls back to hard-coded annual CPI values if the network request fails.
"""
import os
import io
import json
import requests
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STEP1_OUT  = os.path.join(SCRIPT_DIR, '..', 'step1_universe', 'output', 'universe.parquet')
OUT_DIR    = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

FRED_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL'

# Hard-coded fallback: CPIAUCSL annual averages (BLS series, not seasonally adjusted)
FALLBACK_CPI = {
    1993: 144.5, 1994: 148.2, 1995: 152.4, 1996: 156.9, 1997: 160.5,
    1998: 163.0, 1999: 166.6, 2000: 172.2, 2001: 177.1, 2002: 179.9,
    2003: 184.0, 2004: 188.9, 2005: 195.3, 2006: 201.6, 2007: 207.3,
    2008: 215.3, 2009: 214.5, 2010: 218.1, 2011: 224.9, 2012: 229.6,
    2013: 233.0, 2014: 236.7, 2015: 237.0, 2016: 240.0, 2017: 245.1,
    2018: 251.1, 2019: 255.7, 2020: 258.8,
}


def fetch_cpi_fred() -> dict:
    """Return {year: annual_avg_cpi} from FRED CPIAUCSL."""
    resp = requests.get(FRED_URL, timeout=30)
    resp.raise_for_status()
    cpi_df = pd.read_csv(io.StringIO(resp.text))
    cpi_df.columns = ['date', 'cpi']
    cpi_df['date'] = pd.to_datetime(cpi_df['date'])
    cpi_df['year'] = cpi_df['date'].dt.year
    return cpi_df.groupby('year')['cpi'].mean().to_dict()


def get_cpi() -> dict:
    try:
        cpi = fetch_cpi_fred()
        print("  CPI fetched from FRED.")
        return cpi
    except Exception as e:
        print(f"  FRED fetch failed ({e}); using hard-coded fallback CPI values.")
        return FALLBACK_CPI


def main():
    print("=== Step 2: CPI Deflation ===")

    df = pd.read_parquet(STEP1_OUT)
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year

    cpi = get_cpi()
    base_cpi  = cpi[2017]
    cpi_1995  = cpi.get(1995, FALLBACK_CPI[1995])
    cpi_2015  = cpi.get(2015, FALLBACK_CPI[2015])

    print(f"  Base CPI (2017): {base_cpi:.2f}")
    print(f"  CPI 1995: {cpi_1995:.2f}  →  deflator = {base_cpi/cpi_1995:.3f}")
    print(f"  CPI 2015: {cpi_2015:.2f}  →  deflator = {base_cpi/cpi_2015:.3f}")

    # Save CPI table
    cpi_df = (pd.DataFrame({'year': list(cpi.keys()), 'cpi': list(cpi.values())})
              .sort_values('year')
              .reset_index(drop=True))
    cpi_df['deflator_to_2017'] = base_cpi / cpi_df['cpi']
    cpi_df.to_csv(os.path.join(OUT_DIR, 'cpi_table.csv'), index=False)

    # Map deflator onto each movie
    deflator_map = {yr: base_cpi / v for yr, v in cpi.items()}
    df['cpi_deflator'] = df['release_year'].map(deflator_map)

    # Deflate
    df['budget_real']  = df['budget']  * df['cpi_deflator']
    df['revenue_real'] = df['revenue'] * df['cpi_deflator']

    # Verify
    d1995 = base_cpi / cpi_1995
    d2015 = base_cpi / cpi_2015
    print(f"\n[VERIFY] 1995 deflator = {d1995:.3f} (expected ~1.5–1.6)")
    print(f"[VERIFY] 2015 deflator = {d2015:.3f} (expected ~1.01–1.04)")
    print(f"[VERIFY] 2017 deflator = 1.000 (base year)")

    sample = (df[df['release_year'] == 1995]
              .head(3)[['title', 'release_year', 'budget', 'budget_real']]
              .assign(ratio=lambda x: x['budget_real'] / x['budget']))
    print(f"\n[VERIFY] 1995 spot-check (ratio should be ~{d1995:.2f}):\n{sample.to_string()}")

    null_deflators = df['cpi_deflator'].isna().sum()
    print(f"\n[VERIFY] Rows with missing CPI deflator: {null_deflators} (should be 0)")

    out_path = os.path.join(OUT_DIR, 'deflated.parquet')
    df.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print(f"Saved → {OUT_DIR}/cpi_table.csv")


if __name__ == '__main__':
    main()
