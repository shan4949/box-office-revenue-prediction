"""
Step 0: Ingest + Clean
Merges movies_metadata + credits + keywords on id.
Drops 3 malformed rows where id is not a valid integer.
Serialises complex columns (cast, crew, genres, etc.) as JSON strings for parquet storage.
"""
import os
import ast
import json
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, '..', 'Data1')
OUT_DIR    = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

LIST_COLS = [
    'genres', 'production_companies', 'production_countries',
    'spoken_languages', 'cast', 'crew', 'keywords',
]


def safe_eval(val):
    """Parse a literal Python list/dict string; return [] on failure."""
    if pd.isna(val) or str(val).strip() in ('', 'nan', 'NaN'):
        return []
    try:
        result = ast.literal_eval(str(val))
        return result if isinstance(result, (list, dict)) else []
    except Exception:
        return []


def safe_eval_collection(val):
    """Parse belongs_to_collection (may be a dict or NaN)."""
    if pd.isna(val) or str(val).strip() in ('', 'nan', 'NaN'):
        return None
    try:
        result = ast.literal_eval(str(val))
        return result if isinstance(result, dict) and result else None
    except Exception:
        return None


def main():
    print("=== Step 0: Ingest + Clean ===")

    print("Loading movies_metadata.csv …")
    movies = pd.read_csv(os.path.join(DATA_DIR, 'movies_metadata.csv'), low_memory=False)
    print(f"  Raw rows: {len(movies):,}")

    print("Loading credits.csv …")
    credits = pd.read_csv(os.path.join(DATA_DIR, 'credits.csv'))

    print("Loading keywords.csv …")
    keywords = pd.read_csv(os.path.join(DATA_DIR, 'keywords.csv'))

    # --- Cast id to int, drop malformed rows --------------------------------
    before = len(movies)
    movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
    movies = movies.dropna(subset=['id']).copy()
    movies['id'] = movies['id'].astype(int)
    dropped = before - len(movies)
    print(f"  Dropped {dropped} malformed rows from movies_metadata (expected 3)")

    for df_name, df in [('credits', credits), ('keywords', keywords)]:
        df['id'] = pd.to_numeric(df['id'], errors='coerce')

    credits  = credits.dropna(subset=['id']).drop_duplicates(subset=['id']).copy()
    keywords = keywords.dropna(subset=['id']).drop_duplicates(subset=['id']).copy()
    credits['id']  = credits['id'].astype(int)
    keywords['id'] = keywords['id'].astype(int)

    # --- Merge --------------------------------------------------------------
    merged = movies.merge(credits,  on='id', how='inner')
    merged = merged.merge(keywords, on='id', how='inner')
    print(f"  After inner merge: {len(merged):,} rows (expected ~45,463)")

    # --- Parse literal-eval columns -----------------------------------------
    print("Parsing literal-eval columns …")
    for col in LIST_COLS:
        if col in merged.columns:
            merged[col] = merged[col].apply(safe_eval)

    if 'belongs_to_collection' in merged.columns:
        merged['belongs_to_collection'] = merged['belongs_to_collection'].apply(safe_eval_collection)

    # --- Serialise complex columns as JSON strings for parquet --------------
    for col in LIST_COLS:
        if col in merged.columns:
            merged[col] = merged[col].apply(json.dumps)

    merged['belongs_to_collection'] = merged['belongs_to_collection'].apply(
        lambda x: json.dumps(x) if x is not None else None
    )

    # --- Verify -------------------------------------------------------------
    print(f"\n[VERIFY] Merged frame: {len(merged):,} rows (expected ~45,463)")
    print(f"[VERIFY] Malformed rows dropped: {dropped} (expected 3)")
    print(f"[VERIFY] Sample columns: {list(merged.columns[:10])}")

    # --- Save ---------------------------------------------------------------
    out_path = os.path.join(OUT_DIR, 'merged.parquet')
    merged.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
