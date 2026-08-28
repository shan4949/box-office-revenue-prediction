"""
Step 6: Temporal Split
Generates expanding-window rolling-origin CV folds (val years 1996–2015)
plus a final held-out test set (2016). Saves splits as JSON (safe, no arbitrary
code execution risk) and the model-ready dataframe as parquet.
"""
import os
import json
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STEP5_OUT  = os.path.join(SCRIPT_DIR, '..', 'step5_starpower', 'output', 'with_starpower.parquet')
OUT_DIR    = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("=== Step 6: Temporal Split ===")

    df = pd.read_parquet(STEP5_OUT)
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year
    df = df.reset_index(drop=True)

    print(f"  Input: {len(df):,} rows, years {int(df['release_year'].min())}–{int(df['release_year'].max())}")

    # Expanding-window rolling-origin CV
    # Fold k: train on release_year < val_year, validate on release_year == val_year
    # val years: 1996 … 2015  (20 folds)
    folds = []
    for val_year in range(1996, 2016):
        train_idx = df.index[df['release_year'] <  val_year].tolist()
        val_idx   = df.index[df['release_year'] == val_year].tolist()
        folds.append({
            'val_year' : val_year,
            'train_idx': train_idx,
            'val_idx'  : val_idx,
            'n_train'  : len(train_idx),
            'n_val'    : len(val_idx),
        })

    # Final holdout: train 1995–2015, test 2016
    # (2017 excluded — too few rows for a reliable metric)
    train_mask = df['release_year'] <= 2015
    test_mask  = df['release_year'] == 2016
    test_split = {
        'train_idx': df.index[train_mask].tolist(),
        'test_idx' : df.index[test_mask].tolist(),
        'n_train'  : int(train_mask.sum()),
        'n_test'   : int(test_mask.sum()),
    }

    print(f"\n  CV folds : {len(folds)} (val years 1996–2015)")
    print(f"  {'val_year':>8}  {'n_train':>8}  {'n_val':>6}")
    for fold in folds:
        print(f"  {fold['val_year']:>8}  {fold['n_train']:>8}  {fold['n_val']:>6}")

    print(f"\n  Final holdout : n_train={test_split['n_train']:,}  n_test={test_split['n_test']:,}")

    # --- Verify: no val/test row earlier than any train row -----------------
    print("\n[VERIFY] Temporal ordering check …")
    for fold in folds:
        if not fold['val_idx'] or not fold['train_idx']:
            continue
        max_train = int(df.loc[fold['train_idx'], 'release_year'].max())
        min_val   = int(df.loc[fold['val_idx'],   'release_year'].min())
        assert min_val > max_train, (
            f"Fold {fold['val_year']}: min_val_year={min_val} ≤ max_train_year={max_train}"
        )
    max_train_final = int(df.loc[test_split['train_idx'], 'release_year'].max())
    min_test_final  = int(df.loc[test_split['test_idx'],  'release_year'].min())
    assert min_test_final > max_train_final, "Test set overlaps training set"
    print("  All folds: val_year > max train_year ✓")
    print(f"  Final test year ({min_test_final}) > max train year ({max_train_final}) ✓")

    # --- CQR final test split: train ≤ 2014, calib = 2015, test = 2016 ------
    # (loses one year of training vs the plain test split — price of calibration)
    cqr_train = df['release_year'] <= 2014
    cqr_calib = df['release_year'] == 2015
    cqr_test  = df['release_year'] == 2016
    test_cqr = {
        'train_idx': df.index[cqr_train].tolist(),
        'calib_idx': df.index[cqr_calib].tolist(),
        'test_idx' : df.index[cqr_test ].tolist(),
        'n_train'  : int(cqr_train.sum()),
        'n_calib'  : int(cqr_calib.sum()),
        'n_test'   : int(cqr_test.sum()),
    }
    print(f"\n  CQR final split : train ≤ 2014 ({test_cqr['n_train']:,})  "
          f"calib 2015 ({test_cqr['n_calib']})  test 2016 ({test_cqr['n_test']})")

    # Verify calib dates: all ≥ 2015-01-01 and < 2016-01-01
    calib_years = df.loc[test_cqr['calib_idx'], 'release_year'].unique()
    assert list(calib_years) == [2015], f"Calib fold years: {calib_years}"
    max_train_cqr = int(df.loc[test_cqr['train_idx'], 'release_year'].max())
    min_test_cqr  = int(df.loc[test_cqr['test_idx'],  'release_year'].min())
    assert max_train_cqr < 2015, "CQR train bleeds into calib year"
    assert min_test_cqr  > 2015, "CQR test bleeds into calib year"
    print("  CQR calib isolated to 2015, no overlap with train or test ✓")

    # --- Save ---------------------------------------------------------------
    splits = {'folds': folds, 'test': test_split, 'test_cqr': test_cqr}
    with open(os.path.join(OUT_DIR, 'splits.json'), 'w') as f:
        json.dump(splits, f)
    df.to_parquet(os.path.join(OUT_DIR, 'modeling_df.parquet'), index=False)

    print(f"\nSaved → {OUT_DIR}/splits.json")
    print(f"Saved → {OUT_DIR}/modeling_df.parquet")


if __name__ == '__main__':
    main()
