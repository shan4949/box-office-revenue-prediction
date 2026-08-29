# Box Office Revenue Prediction — Validation Report

## Modeling Setup

| Parameter | Value |
|-----------|-------|
| N | 3,991 films |
| Universe | budget > 0 & revenue > 0, years 1995–2017 |
| Revenue target | log(real revenue, 2017 USD) |
| Train | ≤ 2014 (final model with CQR) |
| Calib | 2015 (CQR conformity scores) |
| Test | 2016 (held-out) |
| CV | 19-fold expanding window, test years 1997–2015 |
| Model | LightGBM quantile (P10 / P50 / P90) |
| Calibration | Conformal Quantile Regression (CQR), α = 0.20 |
| Features | 49 pre-release features |

## 2016 Holdout — Coverage & Width (Raw vs CQR)

| Metric | Raw (uncalibrated) | CQR-adjusted |
|--------|--------------------|--------------|
| P10–P90 coverage | 61.3% | **82.5%** |
| Mean interval width (log-$) | 2.3971 | 3.5613 |
| CQR shift Q | — | 0.5821 |
| n_calib (2015) | — | 211 |

## 2016 Holdout — Point Forecasts vs Baselines

| Metric | LightGBM | LR Baseline | Genre-Mean Baseline |
|--------|----------|-------------|---------------------|
| Pinball P50 | **0.5251** | 0.5940 | 0.8716 |
| R² | — | 0.580 | 0.118 |
| N test | 235 | 235 | 235 |

## Rolling-Window CV: Per-Fold CQR Metrics (test years 1997–2015)

| test_year | n_ca | Q      | cov_raw | cov_cqr | width_raw | width_cqr | pb_p50 | lr_pb  |
| --------- | ---- | ------ | ------- | ------- | --------- | --------- | ------ | ------ |
| 1997      | 101  | 0.3922 | 0.728   | 0.816   | 4.6852    | 5.4696    | 0.6818 | 0.6206 |
| 1998      | 114  | 0.6938 | 0.686   | 0.907   | 3.6140    | 5.0016    | 0.5514 | 0.5122 |
| 1999      | 118  | 0.4435 | 0.584   | 0.745   | 2.9720    | 3.8591    | 0.6464 | 0.6004 |
| 2000      | 137  | 0.6874 | 0.560   | 0.881   | 2.3279    | 3.7027    | 0.5575 | 0.4961 |
| 2001      | 134  | 0.3505 | 0.623   | 0.760   | 2.2963    | 2.9973    | 0.4946 | 0.4969 |
| 2002      | 146  | 0.5333 | 0.604   | 0.830   | 2.3449    | 3.4115    | 0.5200 | 0.5271 |
| 2003      | 159  | 0.4841 | 0.564   | 0.757   | 2.0851    | 3.0533    | 0.5950 | 0.6012 |
| 2004      | 140  | 0.5917 | 0.543   | 0.803   | 2.1289    | 3.3123    | 0.6433 | 0.6157 |
| 2005      | 173  | 0.6507 | 0.527   | 0.814   | 1.9340    | 3.2353    | 0.5950 | 0.5954 |
| 2006      | 188  | 0.5394 | 0.576   | 0.802   | 2.1499    | 3.2288    | 0.5555 | 0.5793 |
| 2007      | 217  | 0.5563 | 0.613   | 0.799   | 2.3483    | 3.4609    | 0.6241 | 0.6746 |
| 2008      | 199  | 0.5079 | 0.553   | 0.822   | 2.1024    | 3.1181    | 0.5485 | 0.5871 |
| 2009      | 208  | 0.4772 | 0.560   | 0.801   | 2.1853    | 3.1397    | 0.5345 | 0.5739 |
| 2010      | 216  | 0.4718 | 0.577   | 0.786   | 2.2672    | 3.2108    | 0.5421 | 0.6136 |
| 2011      | 234  | 0.5176 | 0.536   | 0.741   | 2.3885    | 3.4237    | 0.6552 | 0.7231 |
| 2012      | 239  | 0.7312 | 0.522   | 0.809   | 2.4502    | 3.9127    | 0.7108 | 0.7444 |
| 2013      | 209  | 0.6515 | 0.617   | 0.841   | 2.4935    | 3.7966    | 0.6062 | 0.6700 |
| 2014      | 227  | 0.5044 | 0.579   | 0.804   | 2.7041    | 3.7129    | 0.6400 | 0.7188 |
| 2015      | 214  | 0.5689 | 0.569   | 0.801   | 2.6124    | 3.7501    | 0.5945 | 0.6594 |

CV means (19 folds):
- CQR coverage   : **0.806** (raw: 0.585)
- Mean Q shift   : 0.5449
- Pinball P50    : 0.5945
- LR baseline P50: 0.6083

## Top-10 SHAP Features (P50 model, 2016 test set)

| Rank | Feature         | Mean |SHAP| |
| ---- | --------------- | ----------- |
| 1    | log_budget_real | 1.0709      |
| 2    | studio_other    | 0.2506      |
| 3    | runtime         | 0.2443      |
| 4    | franchise_flag  | 0.2342      |
| 5    | log_star_power  | 0.1319      |
| 6    | genre_drama     | 0.0892      |
| 7    | release_month   | 0.0712      |
| 8    | release_dow     | 0.0705      |
| 9    | genre_comedy    | 0.0396      |
| 10   | genre_animation | 0.0257      |

No post-release features in top 10 ✓

---

## Resume-Ready Metrics

```
Model       : LightGBM Quantile + CQR Calibration (P10 / P50 / P90)
N           : 3,991 films  |  Train ≤ 2014  |  Calib 2015  |  Test 2016
Features    : 49 pre-release features

Point forecast (P50 pinball, 2016 holdout):
  LightGBM          :  0.5251
  LR baseline        :  0.5940
  Genre-mean baseline:  0.8716

Prediction interval (2016 holdout):
  Raw P10–P90 coverage : 61.3%  (width 2.3971)
  CQR P10–P90 coverage : 82.5%  (width 3.5613)
  CQR shift Q          : 0.5821

CV (19-fold rolling origin, test years 1997–2015):
  Mean CQR coverage : 0.806
  Mean raw coverage : 0.585
  Mean pinball P50  : 0.5945
```
