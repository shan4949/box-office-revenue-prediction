"""
Step 7: Baselines
Two simple baselines the LightGBM model must beat:
  1. Linear regression on log(budget_real) alone
  2. Genre-mean: predict the training-set mean log-revenue for each film's genres

Metrics: R² and pinball loss at P50 (equivalent to scaled MAE).
Evaluated on all 20 CV folds AND the 2016 holdout test set.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STEP6_DIR  = os.path.join(SCRIPT_DIR, '..', 'step6_split', 'output')
OUT_DIR    = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = 'log_revenue_real'


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.5) -> float:
    residual = y_true - y_pred
    return float(np.mean(np.where(residual >= 0, alpha * residual, (alpha - 1) * residual)))


def train_genre_means(df: pd.DataFrame, train_idx: list, genre_cols: list) -> dict:
    global_mean = float(df.loc[train_idx, TARGET].mean())
    means = {}
    for gc in genre_cols:
        mask = df.loc[train_idx, gc] == 1
        means[gc] = float(df.loc[train_idx][mask][TARGET].mean()) if mask.sum() > 0 else global_mean
    return means, global_mean


def predict_genre_mean(df: pd.DataFrame, val_idx: list, genre_cols: list,
                       genre_means: dict, global_mean: float) -> np.ndarray:
    preds = []
    for idx in val_idx:
        row_preds = [genre_means[gc] for gc in genre_cols if df.at[idx, gc] == 1]
        preds.append(float(np.mean(row_preds)) if row_preds else global_mean)
    return np.array(preds)


def evaluate_fold(df, train_idx, eval_idx, genre_cols):
    y_train = df.loc[train_idx, TARGET].values
    y_eval  = df.loc[eval_idx,  TARGET].values

    # Baseline 1: linear regression on log_budget_real
    X_tr = df.loc[train_idx, 'log_budget_real'].values.reshape(-1, 1)
    X_ev = df.loc[eval_idx,  'log_budget_real'].values.reshape(-1, 1)
    lr   = LinearRegression().fit(X_tr, y_train)
    yp_lr = lr.predict(X_ev)

    # Baseline 2: genre-mean
    gmeans, global_mean = train_genre_means(df, train_idx, genre_cols)
    yp_genre = predict_genre_mean(df, eval_idx, genre_cols, gmeans, global_mean)

    return {
        'lr_r2'       : round(float(r2_score(y_eval, yp_lr)),    4),
        'lr_pinball'  : round(pinball_loss(y_eval, yp_lr,   0.5), 4),
        'genre_r2'    : round(float(r2_score(y_eval, yp_genre)),  4),
        'genre_pinball': round(pinball_loss(y_eval, yp_genre, 0.5), 4),
    }


def main():
    print("=== Step 7: Baselines ===")

    df = pd.read_parquet(os.path.join(STEP6_DIR, 'modeling_df.parquet'))
    df[TARGET] = np.log(df['revenue_real'])

    with open(os.path.join(STEP6_DIR, 'splits.json')) as f:
        splits = json.load(f)

    genre_cols = [c for c in df.columns if c.startswith('genre_')]
    print(f"  Genre columns: {len(genre_cols)}")

    # --- CV folds -----------------------------------------------------------
    fold_rows = []
    for fold in splits['folds']:
        m = evaluate_fold(df, fold['train_idx'], fold['val_idx'], genre_cols)
        m['val_year'] = fold['val_year']
        m['n_val']    = fold['n_val']
        fold_rows.append(m)
        print(f"  val_year={fold['val_year']:4d}  "
              f"lr_pinball={m['lr_pinball']:.4f}  lr_R²={m['lr_r2']:.3f}  |  "
              f"genre_pinball={m['genre_pinball']:.4f}  genre_R²={m['genre_r2']:.3f}")

    fold_df = pd.DataFrame(fold_rows)
    print(f"\n[VERIFY] CV mean across {len(fold_rows)} folds:")
    print(f"  LR baseline   : R²={fold_df['lr_r2'].mean():.3f}  pinball={fold_df['lr_pinball'].mean():.4f}")
    print(f"  Genre baseline: R²={fold_df['genre_r2'].mean():.3f}  pinball={fold_df['genre_pinball'].mean():.4f}")

    # --- 2016 holdout -------------------------------------------------------
    tst  = splits['test']
    test_m = evaluate_fold(df, tst['train_idx'], tst['test_idx'], genre_cols)
    test_m['test_year'] = 2016
    test_m['n_test']    = tst['n_test']

    print(f"\n[VERIFY] 2016 holdout baseline metrics (N={tst['n_test']:,}):")
    print(f"  LR    : R²={test_m['lr_r2']:.3f}  pinball_P50={test_m['lr_pinball']:.4f}")
    print(f"  Genre : R²={test_m['genre_r2']:.3f}  pinball_P50={test_m['genre_pinball']:.4f}")

    # --- Save ---------------------------------------------------------------
    fold_df.to_csv(os.path.join(OUT_DIR, 'cv_baseline_metrics.csv'), index=False)
    with open(os.path.join(OUT_DIR, 'test_baseline_metrics.json'), 'w') as f:
        json.dump(test_m, f, indent=2)

    print(f"\nSaved → {OUT_DIR}/cv_baseline_metrics.csv")
    print(f"Saved → {OUT_DIR}/test_baseline_metrics.json")


if __name__ == '__main__':
    main()
