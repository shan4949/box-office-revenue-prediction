"""
Step 9: Validation Report
Combines CV fold metrics, CQR metrics, SHAP importances, and baseline comparisons
into a Markdown report and machine-readable JSON.
"""
import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STEP6_DIR  = os.path.join(SCRIPT_DIR, '..', 'step6_split',   'output')
STEP7_DIR  = os.path.join(SCRIPT_DIR, '..', 'step7_baselines','output')
STEP8_DIR  = os.path.join(SCRIPT_DIR, '..', 'step8_model',   'output')
OUT_DIR    = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

POST_RELEASE = {
    'vote_average', 'vote_count', 'popularity',
    'homepage', 'overview', 'tagline', 'status',
}


def fmt_table(headers: list[str], rows: list[list]) -> str:
    widths = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    sep  = '| ' + ' | '.join('-' * w for w in widths) + ' |'
    head = '| ' + ' | '.join(str(h).ljust(w) for h, w in zip(headers, widths)) + ' |'
    body = [
        '| ' + ' | '.join(str(r[i]).ljust(w) for i, w in enumerate(widths)) + ' |'
        for r in rows
    ]
    return '\n'.join([head, sep] + body)


def main():
    print("=== Step 9: Validation Report ===")

    # --- Load data ----------------------------------------------------------
    df = pd.read_parquet(os.path.join(STEP6_DIR, 'modeling_df.parquet'))
    df['log_revenue_real'] = np.log(df['revenue_real'])

    with open(os.path.join(STEP6_DIR, 'splits.json')) as f:
        splits = json.load(f)

    with open(os.path.join(STEP8_DIR, 'models', 'feature_cols.json')) as f:
        feature_cols = json.load(f)

    cv_uncal = pd.read_csv(os.path.join(STEP8_DIR, 'cv_fold_metrics.csv'))
    cv_cqr   = pd.read_csv(os.path.join(STEP8_DIR, 'cv_cqr_metrics.csv'))
    cv_base  = pd.read_csv(os.path.join(STEP7_DIR, 'cv_baseline_metrics.csv'))

    with open(os.path.join(STEP8_DIR, 'test_metrics.json')) as f:
        tm = json.load(f)
    with open(os.path.join(STEP7_DIR, 'test_baseline_metrics.json')) as f:
        tm_base = json.load(f)

    # --- SHAP on 2016 test set (P50 model) ----------------------------------
    print("Computing SHAP values …")
    test_idx = splits['test_cqr']['test_idx']
    X_test   = df.loc[test_idx, feature_cols].values

    p50_booster = lgb.Booster(model_file=os.path.join(STEP8_DIR, 'models', 'p50.txt'))
    explainer   = shap.TreeExplainer(p50_booster)
    shap_vals   = explainer.shap_values(X_test)

    mean_abs_shap = pd.Series(np.abs(shap_vals).mean(axis=0), index=feature_cols)
    top10 = mean_abs_shap.sort_values(ascending=False).head(10)

    print(f"\n[VERIFY] Top-10 SHAP (P50, 2016 test):")
    for feat, val in top10.items():
        print(f"  {feat}: {val:.4f}")

    leaked = [f for f in top10.index if f in POST_RELEASE]
    if leaked:
        raise RuntimeError(f"Post-release features in top 10: {leaked}")
    print("[VERIFY] No post-release features in top 10 ✓")

    # --- Coverage / width comparison table (CQR CV) -------------------------
    cv_merged = cv_cqr.merge(
        cv_base.drop(columns=['n_val'], errors='ignore'),
        left_on='test_year', right_on='val_year', how='left'
    )

    # --- Build Markdown report ----------------------------------------------
    n_total = len(df)
    n_feat  = len(feature_cols)

    # Uncalibrated CV summary row
    uncal_mean_cov = cv_uncal['coverage_p10_p90'].mean()
    uncal_mean_pb  = cv_uncal['pinball_p50'].mean()
    cqr_mean_cov   = cv_cqr['cov_cqr'].mean()
    cqr_mean_raw_cov = cv_cqr['cov_raw'].mean()
    cqr_mean_pb    = cv_cqr['pinball_p50'].mean()
    cqr_mean_Q     = cv_cqr['Q'].mean()

    # Build CQR fold table rows
    cqr_fold_headers = ['test_year', 'n_ca', 'Q', 'cov_raw', 'cov_cqr', 'width_raw', 'width_cqr', 'pb_p50', 'lr_pb']
    cqr_fold_rows = []
    for _, row in cv_merged.iterrows():
        lr_pb = f"{row['lr_pinball']:.4f}" if 'lr_pinball' in row.index and not pd.isna(row.get('lr_pinball')) else '—'
        cqr_fold_rows.append([
            int(row['test_year']),
            int(row['n_calib']),
            f"{row['Q']:.4f}",
            f"{row['cov_raw']:.3f}",
            f"{row['cov_cqr']:.3f}",
            f"{row['width_raw']:.4f}",
            f"{row['width_cqr']:.4f}",
            f"{row['pinball_p50']:.4f}",
            lr_pb,
        ])

    shap_rows = [
        [str(i + 1), feat, f"{val:.4f}"]
        for i, (feat, val) in enumerate(top10.items())
    ]

    lr_r2 = tm_base.get('lr_r2', '—')
    report = f"""# Box Office Revenue Prediction — Validation Report

## Modeling Setup

| Parameter | Value |
|-----------|-------|
| N | {n_total:,} films |
| Universe | budget > 0 & revenue > 0, years 1995–2017 |
| Revenue target | log(real revenue, 2017 USD) |
| Train | ≤ 2014 (final model with CQR) |
| Calib | 2015 (CQR conformity scores) |
| Test | 2016 (held-out) |
| CV | 19-fold expanding window, test years 1997–2015 |
| Model | LightGBM quantile (P10 / P50 / P90) |
| Calibration | Conformal Quantile Regression (CQR), α = 0.20 |
| Features | {n_feat} pre-release features |

## 2016 Holdout — Coverage & Width (Raw vs CQR)

| Metric | Raw (uncalibrated) | CQR-adjusted |
|--------|--------------------|--------------|
| P10–P90 coverage | {tm['coverage_raw']:.1%} | **{tm['coverage_cqr']:.1%}** |
| Mean interval width (log-$) | {tm['width_raw']:.4f} | {tm['width_cqr']:.4f} |
| CQR shift Q | — | {tm['Q_final']:.4f} |
| n_calib (2015) | — | {tm['n_calib']} |

## 2016 Holdout — Point Forecasts vs Baselines

| Metric | LightGBM | LR Baseline | Genre-Mean Baseline |
|--------|----------|-------------|---------------------|
| Pinball P50 | **{tm['pinball_p50']:.4f}** | {tm_base['lr_pinball']:.4f} | {tm_base['genre_pinball']:.4f} |
| R² | — | {lr_r2:.3f} | {tm_base['genre_r2']:.3f} |
| N test | {tm['n_test']} | {tm['n_test']} | {tm['n_test']} |

## Rolling-Window CV: Per-Fold CQR Metrics (test years 1997–2015)

{fmt_table(cqr_fold_headers, cqr_fold_rows)}

CV means (19 folds):
- CQR coverage   : **{cqr_mean_cov:.3f}** (raw: {cqr_mean_raw_cov:.3f})
- Mean Q shift   : {cqr_mean_Q:.4f}
- Pinball P50    : {cqr_mean_pb:.4f}
- LR baseline P50: {cv_base['lr_pinball'].mean():.4f}

## Top-10 SHAP Features (P50 model, 2016 test set)

{fmt_table(['Rank', 'Feature', 'Mean |SHAP|'], shap_rows)}

No post-release features in top 10 ✓

---

## Resume-Ready Metrics

```
Model       : LightGBM Quantile + CQR Calibration (P10 / P50 / P90)
N           : {n_total:,} films  |  Train ≤ 2014  |  Calib 2015  |  Test 2016
Features    : {n_feat} pre-release features

Point forecast (P50 pinball, 2016 holdout):
  LightGBM          :  {tm['pinball_p50']:.4f}
  LR baseline        :  {tm_base['lr_pinball']:.4f}
  Genre-mean baseline:  {tm_base['genre_pinball']:.4f}

Prediction interval (2016 holdout):
  Raw P10–P90 coverage : {tm['coverage_raw']:.1%}  (width {tm['width_raw']:.4f})
  CQR P10–P90 coverage : {tm['coverage_cqr']:.1%}  (width {tm['width_cqr']:.4f})
  CQR shift Q          : {tm['Q_final']:.4f}

CV (19-fold rolling origin, test years 1997–2015):
  Mean CQR coverage : {cqr_mean_cov:.3f}
  Mean raw coverage : {cqr_mean_raw_cov:.3f}
  Mean pinball P50  : {cqr_mean_pb:.4f}
```
"""

    out_md = os.path.join(OUT_DIR, 'validation_report.md')
    with open(out_md, 'w') as f:
        f.write(report)

    # --- Resume JSON --------------------------------------------------------
    resume = {
        "model"                       : "LightGBM Quantile + CQR",
        "N_total"                     : n_total,
        "train_window"                : "1995-2014",
        "calib_year"                  : 2015,
        "test_year"                   : 2016,
        "n_features"                  : n_feat,
        "pinball_p50_test"            : tm['pinball_p50'],
        "coverage_raw_test"           : tm['coverage_raw'],
        "coverage_cqr_test"           : tm['coverage_cqr'],
        "width_raw_test"              : tm['width_raw'],
        "width_cqr_test"              : tm['width_cqr'],
        "Q_final"                     : tm['Q_final'],
        "n_calib"                     : tm['n_calib'],
        "lr_baseline_pinball_p50"     : tm_base['lr_pinball'],
        "genre_baseline_pinball_p50"  : tm_base['genre_pinball'],
        "lr_baseline_r2"              : lr_r2,
        "cv_mean_cov_raw"             : round(float(cqr_mean_raw_cov), 4),
        "cv_mean_cov_cqr"             : round(float(cqr_mean_cov), 4),
        "cv_mean_pinball_p50"         : round(float(cqr_mean_pb), 4),
        "cv_mean_Q"                   : round(float(cqr_mean_Q), 4),
        "top10_shap_features"         : top10.index.tolist(),
        "leakage_check_passed"        : True,
    }

    out_json = os.path.join(OUT_DIR, 'resume_metrics.json')
    with open(out_json, 'w') as f:
        json.dump(resume, f, indent=2)

    # --- Console summary ----------------------------------------------------
    bar = '=' * 66
    print(f"\n{bar}")
    print("  RESUME-READY METRICS")
    print(bar)
    print(f"  Model       : LightGBM Quantile + CQR, {n_feat} features")
    print(f"  N           : {n_total:,} | train ≤ 2014 | calib 2015 | test 2016")
    print(f"  Pinball P50 : {tm['pinball_p50']:.4f}")
    print(f"    vs LR     : {tm_base['lr_pinball']:.4f}")
    print(f"    vs genre  : {tm_base['genre_pinball']:.4f}")
    print(f"  Coverage (2016 holdout):")
    print(f"    Raw        : {tm['coverage_raw']:.1%}  width={tm['width_raw']:.4f}")
    print(f"    CQR        : {tm['coverage_cqr']:.1%}  width={tm['width_cqr']:.4f}  Q={tm['Q_final']:.4f}")
    print(f"  CV mean CQR coverage : {cqr_mean_cov:.3f}  (raw: {cqr_mean_raw_cov:.3f})")
    print(f"  Top SHAP feature     : {top10.index[0]}")
    print(bar)

    print(f"\nSaved → {out_md}")
    print(f"Saved → {out_json}")


if __name__ == '__main__':
    main()
