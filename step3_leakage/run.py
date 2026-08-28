"""
Step 3: Leakage Audit
Drops every post-release column from the explicit exclude list before any feature
engineering or model training can see it.
"""
import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STEP2_OUT  = os.path.join(SCRIPT_DIR, '..', 'step2_cpi', 'output', 'deflated.parquet')
OUT_DIR    = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

# Columns that are only known AFTER a film is released — must never reach the model
POST_RELEASE = [
    'vote_average',    # aggregated after release
    'vote_count',      # aggregated after release
    'popularity',      # computed from post-release engagement
    'homepage',        # sometimes updated post-release
    'overview',        # synopsis (NLP leakage risk if used as raw text proxy)
    'tagline',         # marketing copy; can encode commercial expectations leakily
    'status',          # "Released" is tautological for any row in the universe
]


def main():
    print("=== Step 3: Leakage Audit ===")

    df = pd.read_parquet(STEP2_OUT)
    print(f"  Input: {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Columns present: {sorted(df.columns.tolist())}")

    cols_to_drop = [c for c in POST_RELEASE if c in df.columns]
    cols_missing = [c for c in POST_RELEASE if c not in df.columns]

    print(f"\n  Dropping {len(cols_to_drop)} post-release columns: {cols_to_drop}")
    if cols_missing:
        print(f"  (Already absent — nothing to drop): {cols_missing}")

    df = df.drop(columns=cols_to_drop)

    # --- Verify: none of the exclude list remains ---------------------------
    leaked = [c for c in POST_RELEASE if c in df.columns]
    if leaked:
        raise RuntimeError(f"[FAIL] Post-release columns still present: {leaked}")
    print(f"\n[VERIFY] No post-release columns remain ✓")
    print(f"[VERIFY] Remaining columns ({len(df.columns)}):")
    for c in sorted(df.columns):
        print(f"  {c}")

    out_path = os.path.join(OUT_DIR, 'clean.parquet')
    df.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
