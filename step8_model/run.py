"""
Step 8: LightGBM Quantile Model + CQR Calibration

Walk-forward CV uses a 3-way split for each test year Y:
  - Train : years ≤ Y-2
  - Calib : year  == Y-1  (CQR conformity scores)
  - Test  : year  == Y

CQR shift Q is computed from calibration nonconformity scores:
  E_i = max(q10_pred_i - y_true_i, y_true_i - q90_pred_i)
  Q   = quantile(E, min(1, ceil((1-α)(n+1)/n)), method='higher')   α = 0.20

Adjusted interval: [q10 - Q, q90 + Q]

Reports raw and CQR-adjusted coverage + width side-by-side.
Models saved in LightGBM native text format (no pickle).
"""
import os
import json
import math
import numpy as np
import pandas as pd
import lightgbm as lgb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STEP6_DIR  = os.path.join(SCRIPT_DIR, '..', 'step6_split', 'output')
OUT_DIR    = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = 'log_revenue_real'
ALPHA  = 0.20   # target miscoverage rate → 1 - ALPHA = 80% coverage

LGB_BASE = dict(
    n_estimators     = 600,
    learning_rate    = 0.03,
    num_leaves       = 63,
    min_child_samples= 20,
    subsample        = 0.8,
    subsample_freq   = 1,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    verbose          = -1,
    n_jobs           = -1,
)

QUANTILES = {'p10': 0.10, 'p50': 0.50, 'p90': 0.90}

FEATURE_PREFIXES = (
    'log_budget_real', 'runtime',
    'genre_', 'lang_',
    'release_month', 'release_dow', 'holiday_flag',
    'franchise_flag', 'studio_',
    'log_star_power', 'is_debut',
)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        for prefix in FEATURE_PREFIXES:
            if col == prefix or col.startswith(prefix):
                cols.append(col)
                break
    return cols


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    r = y_true - y_pred
    return float(np.mean(np.where(r >= 0, alpha * r, (alpha - 1) * r)))


def empirical_coverage(y_true: np.ndarray, y_low: np.ndarray, y_high: np.ndarray) -> float:
    return float(np.mean((y_true >= y_low) & (y_true <= y_high)))


def fit_models(X_tr: np.ndarray, y_tr: np.ndarray) -> dict:
    """Fit P10/P50/P90 LGBMRegressors; return {name: model}."""
    models = {}
    for name, alpha in QUANTILES.items():
        m = lgb.LGBMRegressor(**{**LGB_BASE, 'objective': 'quantile', 'alpha': alpha})
        m.fit(X_tr, y_tr)
        models[name] = m
    return models


def cqr_quantile(n: int) -> float:
    """Finite-sample quantile level for (1-ALPHA) CQR coverage."""
    return min(1.0, math.ceil((1 - ALPHA) * (n + 1)) / n)


def compute_cqr_Q(models: dict, X_ca: np.ndarray, y_ca: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Compute CQR conformity scores and the shift Q.
    E_i = max(q10_pred_i - y_true_i,  y_true_i - q90_pred_i)
    Returns (Q, E_array).
    """
    p10_ca = models['p10'].predict(X_ca)
    p90_ca = models['p90'].predict(X_ca)
    E = np.maximum(p10_ca - y_ca, y_ca - p90_ca)
    n = len(y_ca)
    q_level = cqr_quantile(n)
    Q = float(np.quantile(E, q_level, method='higher'))
    return Q, E


def main():
    print("=== Step 8: LightGBM Quantile Model + CQR Calibration ===")

    df = pd.read_parquet(os.path.join(STEP6_DIR, 'modeling_df.parquet'))
    df[TARGET] = np.log(df['revenue_real'])

    with open(os.path.join(STEP6_DIR, 'splits.json')) as f:
        splits = json.load(f)

    feature_cols = get_feature_cols(df)
    print(f"  Features ({len(feature_cols)}): {feature_cols}\n")

    # -----------------------------------------------------------------------
    # Phase 1: uncalibrated walk-forward CV (20 folds, val years 1996–2015)
    # Used for comparison with baselines in step9.
    # -----------------------------------------------------------------------
    print("--- Phase 1: Uncalibrated walk-forward CV (20 folds) ---")
    uncal_rows = []
    for fold in splits['folds']:
        tr = fold['train_idx']
        vl = fold['val_idx']
        if not vl:
            continue
        X_tr = df.loc[tr, feature_cols].values
        y_tr = df.loc[tr, TARGET].values
        X_vl = df.loc[vl, feature_cols].values
        y_vl = df.loc[vl, TARGET].values

        mdls = fit_models(X_tr, y_tr)
        p10v = mdls['p10'].predict(X_vl)
        p50v = mdls['p50'].predict(X_vl)
        p90v = mdls['p90'].predict(X_vl)

        uncal_rows.append({
            'val_year'      : fold['val_year'],
            'n_val'         : fold['n_val'],
            'pinball_p50'   : round(pinball_loss(y_vl, p50v, 0.5), 4),
            'coverage_p10_p90': round(empirical_coverage(y_vl, p10v, p90v), 4),
        })
        print(f"  val={fold['val_year']:4d}  pinball_p50={uncal_rows[-1]['pinball_p50']:.4f}  "
              f"coverage={uncal_rows[-1]['coverage_p10_p90']:.3f}")

    uncal_df = pd.DataFrame(uncal_rows)
    uncal_df.to_csv(os.path.join(OUT_DIR, 'cv_fold_metrics.csv'), index=False)
    print(f"\n  Uncal CV mean  pinball={uncal_df['pinball_p50'].mean():.4f}  "
          f"coverage={uncal_df['coverage_p10_p90'].mean():.3f}")

    # -----------------------------------------------------------------------
    # Phase 2: CQR walk-forward CV  (test years 1997–2015, 19 folds)
    # For each test year Y: train ≤ Y-2, calib = Y-1, test = Y
    # -----------------------------------------------------------------------
    print("\n--- Phase 2: CQR walk-forward CV (19 folds, test years 1997–2015) ---")
    cqr_rows = []
    for val_year in range(1997, 2016):
        tr_idx = df.index[df['release_year'] <= val_year - 2].tolist()
        ca_idx = df.index[df['release_year'] == val_year - 1].tolist()
        te_idx = df.index[df['release_year'] == val_year    ].tolist()

        if not tr_idx or not ca_idx or not te_idx:
            continue

        # --- Verify fold isolation (step 1 requirement) --------------------
        ca_years = df.loc[ca_idx, 'release_year'].unique()
        te_years = df.loc[te_idx, 'release_year'].unique()
        max_tr   = int(df.loc[tr_idx, 'release_year'].max())
        assert all(y == val_year - 1 for y in ca_years), \
            f"Calib fold year mismatch: {ca_years}"
        assert all(y == val_year for y in te_years), \
            f"Test fold year mismatch: {te_years}"
        assert max_tr < val_year - 1, \
            f"Train bleeds into calib: max_train_year={max_tr}"

        X_tr = df.loc[tr_idx, feature_cols].values
        y_tr = df.loc[tr_idx, TARGET].values
        X_ca = df.loc[ca_idx, feature_cols].values
        y_ca = df.loc[ca_idx, TARGET].values
        X_te = df.loc[te_idx, feature_cols].values
        y_te = df.loc[te_idx, TARGET].values

        # Fit models on train only
        mdls = fit_models(X_tr, y_tr)

        # --- Verify E (step 2 requirement) ---------------------------------
        Q, E = compute_cqr_Q(mdls, X_ca, y_ca)
        assert len(E) == len(ca_idx), \
            f"E length {len(E)} ≠ calib size {len(ca_idx)}"
        assert not np.any(np.isnan(E)), "E contains NaNs"

        # --- Verify Q sanity (step 3 requirement) --------------------------
        n_ca    = len(ca_idx)
        q_level = cqr_quantile(n_ca)
        rmse_ca = float(np.std(y_ca - mdls['p50'].predict(X_ca)))
        assert Q < 20 * rmse_ca, \
            f"Q={Q:.4f} is suspiciously large (>20x calib RMSE {rmse_ca:.4f})"

        # Raw test predictions
        p10_raw = mdls['p10'].predict(X_te)
        p50_te  = mdls['p50'].predict(X_te)
        p90_raw = mdls['p90'].predict(X_te)

        # CQR-adjusted test predictions (step 4)
        p10_cqr = p10_raw - Q
        p90_cqr = p90_raw + Q

        cov_raw   = empirical_coverage(y_te, p10_raw, p90_raw)
        cov_cqr   = empirical_coverage(y_te, p10_cqr, p90_cqr)
        width_raw = float(np.mean(p90_raw - p10_raw))
        width_cqr = float(np.mean(p90_cqr - p10_cqr))

        # --- Verify width increased (step 5 requirement) -------------------
        assert width_cqr > width_raw, \
            f"CQR width must increase: raw={width_raw:.4f} cqr={width_cqr:.4f}"

        row = {
            'test_year'   : val_year,
            'n_calib'     : n_ca,
            'n_test'      : len(te_idx),
            'Q'           : round(Q, 4),
            'q_level'     : round(q_level, 4),
            'pinball_p50' : round(pinball_loss(y_te, p50_te, 0.5), 4),
            'cov_raw'     : round(cov_raw,   3),
            'cov_cqr'     : round(cov_cqr,   3),
            'width_raw'   : round(width_raw, 4),
            'width_cqr'   : round(width_cqr, 4),
        }
        cqr_rows.append(row)
        print(f"  test={val_year:4d}  n_ca={n_ca:3d}  Q={Q:6.4f}  "
              f"cov {cov_raw:.3f}→{cov_cqr:.3f}  "
              f"width {width_raw:.4f}→{width_cqr:.4f}")

    cqr_df = pd.DataFrame(cqr_rows)
    cqr_df.to_csv(os.path.join(OUT_DIR, 'cv_cqr_metrics.csv'), index=False)
    print(f"\n  CQR CV mean  cov_raw={cqr_df['cov_raw'].mean():.3f}  "
          f"cov_cqr={cqr_df['cov_cqr'].mean():.3f}  "
          f"Q={cqr_df['Q'].mean():.4f}")

    # -----------------------------------------------------------------------
    # Phase 3: Final holdout with CQR
    # test_cqr: train ≤ 2014, calib = 2015, test = 2016
    # -----------------------------------------------------------------------
    print("\n--- Phase 3: Final holdout (train ≤ 2014 | calib 2015 | test 2016) ---")
    tst_cqr = splits['test_cqr']
    tr  = tst_cqr['train_idx']
    ca  = tst_cqr['calib_idx']
    te  = tst_cqr['test_idx']

    X_tr = df.loc[tr, feature_cols].values
    y_tr = df.loc[tr, TARGET].values
    X_ca = df.loc[ca, feature_cols].values
    y_ca = df.loc[ca, TARGET].values
    X_te = df.loc[te, feature_cols].values
    y_te = df.loc[te, TARGET].values

    final_models = fit_models(X_tr, y_tr)
    Q_final, E_final = compute_cqr_Q(final_models, X_ca, y_ca)
    n_ca_final = len(ca)
    q_level_final = cqr_quantile(n_ca_final)

    # Verify E
    assert len(E_final) == n_ca_final, "E length mismatch on final holdout"
    assert not np.any(np.isnan(E_final)), "E contains NaNs on final holdout"
    print(f"  Calib (2015): n={n_ca_final}, q_level={q_level_final:.4f}, Q={Q_final:.4f}")
    print(f"  E range: [{E_final.min():.4f}, {E_final.max():.4f}]  "
          f"median={np.median(E_final):.4f}")

    # Verify Q sanity
    rmse_final = float(np.std(y_ca - final_models['p50'].predict(X_ca)))
    print(f"  P50 calib RMSE: {rmse_final:.4f}  → Q/RMSE ratio: {Q_final/rmse_final:.2f}")
    assert Q_final < 20 * rmse_final, \
        f"Q={Q_final:.4f} is suspiciously large vs RMSE={rmse_final:.4f}"

    p10_raw_te = final_models['p10'].predict(X_te)
    p50_te     = final_models['p50'].predict(X_te)
    p90_raw_te = final_models['p90'].predict(X_te)

    p10_cqr_te = p10_raw_te - Q_final
    p90_cqr_te = p90_raw_te + Q_final

    cov_raw_te   = empirical_coverage(y_te, p10_raw_te, p90_raw_te)
    cov_cqr_te   = empirical_coverage(y_te, p10_cqr_te, p90_cqr_te)
    width_raw_te = float(np.mean(p90_raw_te - p10_raw_te))
    width_cqr_te = float(np.mean(p90_cqr_te - p10_cqr_te))

    assert width_cqr_te > width_raw_te, "CQR width must be wider than raw on final test"

    print(f"\n[VERIFY] 2016 holdout:")
    print(f"  pinball P50   : {pinball_loss(y_te, p50_te, 0.5):.4f}")
    print(f"  coverage raw  : {cov_raw_te:.3f}")
    print(f"  coverage CQR  : {cov_cqr_te:.3f}  (target ~0.80)")
    print(f"  width raw     : {width_raw_te:.4f}")
    print(f"  width CQR     : {width_cqr_te:.4f}  (must be wider ✓)")

    # --- Save models (LightGBM native text format) --------------------------
    model_dir = os.path.join(OUT_DIR, 'models')
    os.makedirs(model_dir, exist_ok=True)
    for name, m in final_models.items():
        m.booster_.save_model(os.path.join(model_dir, f'{name}.txt'))
    with open(os.path.join(model_dir, 'feature_cols.json'), 'w') as f:
        json.dump(feature_cols, f)

    # --- Save all metrics ---------------------------------------------------
    test_metrics = {
        'pinball_p10'      : round(pinball_loss(y_te, p10_raw_te, 0.10), 4),
        'pinball_p50'      : round(pinball_loss(y_te, p50_te,     0.50), 4),
        'pinball_p90'      : round(pinball_loss(y_te, p90_raw_te, 0.90), 4),
        'coverage_raw'     : round(cov_raw_te,   4),
        'coverage_cqr'     : round(cov_cqr_te,   4),
        'width_raw'        : round(width_raw_te, 4),
        'width_cqr'        : round(width_cqr_te, 4),
        'Q_final'          : round(Q_final, 4),
        'q_level'          : round(q_level_final, 4),
        'n_calib'          : n_ca_final,
        'n_test'           : len(te),
    }
    with open(os.path.join(OUT_DIR, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)

    # Test predictions (raw + CQR)
    pred_df = df.loc[te, ['id', 'title', 'release_date', TARGET, 'log_budget_real']].copy()
    pred_df['pred_p10_raw'] = p10_raw_te
    pred_df['pred_p50']     = p50_te
    pred_df['pred_p90_raw'] = p90_raw_te
    pred_df['pred_p10_cqr'] = p10_cqr_te
    pred_df['pred_p90_cqr'] = p90_cqr_te
    pred_df.to_parquet(os.path.join(OUT_DIR, 'test_predictions.parquet'), index=False)

    print(f"\nSaved → {OUT_DIR}/models/  (p10.txt, p50.txt, p90.txt)")
    print(f"Saved → {OUT_DIR}/cv_fold_metrics.csv")
    print(f"Saved → {OUT_DIR}/cv_cqr_metrics.csv")
    print(f"Saved → {OUT_DIR}/test_metrics.json")
    print(f"Saved → {OUT_DIR}/test_predictions.parquet")


if __name__ == '__main__':
    main()
