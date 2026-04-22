"""
Inhomogeneous Poisson process model for EONET event categories.

Pipeline:
  1. Fetch all events per category (2016-present), cache locally as CSV
  2. Aggregate into monthly event counts
  3. Fit GLM Poisson with seasonal (monthly) + linear trend components
  4. Test trend significance and flag potential reporting artifacts

Usage:
  python eonet_poisson.py                  # run full pipeline, all viable categories
  python eonet_poisson.py --category wildfires severeStorms
  python eonet_poisson.py --refresh        # force re-fetch from API (ignore cache)
  python eonet_poisson.py --no-plots       # skip matplotlib output
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import date, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import eonet_client as eonet

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Fetch window — always pull the full raw history; filtering happens post-load.
FETCH_START = "2016-01-01"
FETCH_END   = date.today().isoformat()

# Per-category config: which slice of the cached data is actually usable.
# "analysis_start" / "analysis_end" filter the DataFrame after loading.
# "skip" marks categories too broken to model (shown in output but not fitted).
CATEGORY_CONFIG = {
    "wildfires": {
        "label":          "Wildfires",
        "analysis_start": "2016-01-01",
        "analysis_end":   "2023-12-31",   # 2024 has 57x jump vs 2023 — reporting expansion
        "note":           "2024 dropped: 5,615 events vs 97 in 2023 (IRWIN expansion suspected)",
    },
    "severeStorms": {
        "label":          "Severe Storms",
        "analysis_start": "2016-01-01",   # single stray event in 2015
        "analysis_end":   None,
        "note":           "Consistent named-storm tracking from 2016",
    },
    "volcanoes": {
        "label":          "Volcanoes",
        "analysis_start": "2019-01-01",   # no data before 2019
        "analysis_end":   None,
        "note":           "Source integration started 2019; earlier years have zero events",
    },
    "floods": {
        "label":          "Floods",
        "analysis_start": "2025-05-01",
        "analysis_end":   None,
        "note":           "GDACS automated alerts dominate from May 2025; pre-2025 has 6-year gap. ~12 months of usable data.",
    },
}

# Minimum non-zero months to attempt a fit
MIN_NONZERO_MONTHS = 24

# ---------------------------------------------------------------------------
# 1. Data fetching with local cache
# ---------------------------------------------------------------------------

def _cache_path(category_id: str) -> Path:
    return CACHE_DIR / f"events_{category_id}_{FETCH_START}_{FETCH_END}.csv"


def _fetch_year(category_id: str, year: int) -> list:
    """Fetch events for one category in one calendar year. Retries on failure."""
    start = f"{year}-01-01"
    end   = f"{year}-12-31" if year < date.today().year else date.today().isoformat()
    for attempt in range(4):
        try:
            data = eonet.get_events(
                status="all",
                start=start,
                end=end,
                category=category_id,
                limit=None,
            )
            return data.get("events", [])
        except Exception:
            if attempt == 3:
                return []
            time.sleep(2 ** attempt)
    return []


def fetch_category(category_id: str, refresh: bool = False) -> pd.DataFrame:
    """
    Pull all events for one category from 2016-present, fetching one year
    at a time to avoid API timeouts on large requests.
    Caches to CSV so repeated runs don't hammer the API.
    """
    path = _cache_path(category_id)

    if path.exists() and not refresh:
        df = pd.read_csv(path, parse_dates=["first_date"])
        print(f"  [{category_id}] loaded {len(df)} events from cache")
        return df

    start_year = int(FETCH_START[:4])
    end_year   = date.today().year
    all_events = []

    print(f"  [{category_id}] fetching {start_year}-{end_year} year by year ...")
    for year in range(start_year, end_year + 1):
        events = _fetch_year(category_id, year)
        all_events.extend(events)
        print(f"    {year}: {len(events)} events", flush=True)
        time.sleep(0.3)  # be polite to the API

    print(f"  [{category_id}] total: {len(all_events)} events")

    if not all_events:
        return pd.DataFrame(columns=["id", "title", "first_date", "closed", "category"])

    rows = []
    for ev in all_events:
        geometries = ev.get("geometry", [])
        if not geometries:
            continue
        dates = [g["date"] for g in geometries if g.get("date")]
        if not dates:
            continue
        first_date = min(pd.to_datetime(d) for d in dates)
        rows.append({
            "id":         ev["id"],
            "title":      ev.get("title", ""),
            "first_date": first_date,
            "closed":     ev.get("closed"),
            "category":   category_id,
            "n_updates":  len(geometries),
        })

    df = pd.DataFrame(rows)
    df["first_date"] = pd.to_datetime(df["first_date"], utc=True).dt.tz_localize(None)
    df.to_csv(path, index=False)
    return df


def filter_to_analysis_window(df: pd.DataFrame, category_id: str) -> pd.DataFrame:
    """Apply per-category date filters to remove periods with inconsistent coverage."""
    cfg = CATEGORY_CONFIG.get(category_id, {})
    if df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    if cfg.get("analysis_start"):
        mask &= df["first_date"] >= pd.Timestamp(cfg["analysis_start"])
    if cfg.get("analysis_end"):
        mask &= df["first_date"] <= pd.Timestamp(cfg["analysis_end"])
    return df[mask].copy()


# ---------------------------------------------------------------------------
# 2. Aggregate into monthly counts
# ---------------------------------------------------------------------------

def to_monthly_counts(df: pd.DataFrame, category_id: str) -> pd.DataFrame:
    """
    Aggregate events to a complete monthly time series over the analysis window
    for this category. Months with zero events are explicitly included.
    """
    cfg        = CATEGORY_CONFIG.get(category_id, {})
    start      = cfg.get("analysis_start") or FETCH_START
    end        = cfg.get("analysis_end")   or FETCH_END
    full_index = pd.period_range(start=start, end=end, freq="M")

    if df.empty:
        counts = pd.Series(0, index=full_index, name="count")
    else:
        df = df.copy()
        df["period"] = df["first_date"].dt.to_period("M")
        counts = df.groupby("period").size().reindex(full_index, fill_value=0)
        counts.name = "count"

    monthly = counts.reset_index()
    monthly.columns = ["period", "count"]
    monthly["year"]       = monthly["period"].dt.year
    monthly["month"]      = monthly["period"].dt.month
    monthly["t"]          = range(len(monthly))          # linear time index (0, 1, 2, ...)
    monthly["t_centered"] = monthly["t"] - monthly["t"].mean()  # centered for numerical stability
    monthly["category"]   = category_id

    return monthly


# ---------------------------------------------------------------------------
# 3. Fit inhomogeneous Poisson (GLM with Poisson family, log link)
# ---------------------------------------------------------------------------
#
# Model:  log(λ_t) = β₀ + Σ_{m=2}^{12} β_m · I(month=m) + β_trend · t_centered
#
# β₀           = log baseline rate (January, at mean time)
# β_m          = log-multiplicative seasonal effect for month m (Jan is reference)
# β_trend      = log-multiplicative trend per month (positive = accelerating)
#
# exp(β_trend * 12) ≈ annual multiplicative change in event rate

def build_design_matrix(monthly: pd.DataFrame) -> pd.DataFrame:
    """Construct GLM design matrix: intercept + month dummies + trend.

    Month dummies are always built over the full range 1-12 (using a Categorical
    with explicit categories) so that train and test slices — which may not contain
    every calendar month — always produce identically-shaped matrices.
    """
    month_cat = pd.Categorical(monthly["month"], categories=range(1, 13))
    month_dummies = pd.get_dummies(month_cat, prefix="m", drop_first=True).astype(float)
    month_dummies.index = monthly.index
    X = pd.concat([
        pd.Series(1.0, index=monthly.index, name="intercept"),
        month_dummies,
        monthly["t_centered"].rename("trend"),
    ], axis=1)
    return X


def fit_poisson(monthly: pd.DataFrame, label: str) -> dict:
    """
    Fit GLM Poisson model. Returns a result dict with model, diagnostics,
    and the trend test outcome.
    """
    nonzero = (monthly["count"] > 0).sum()
    if nonzero < MIN_NONZERO_MONTHS:
        return {"label": label, "status": "insufficient_data", "nonzero_months": nonzero}

    X = build_design_matrix(monthly)
    y = monthly["count"]

    model  = sm.GLM(y, X, family=sm.families.Poisson())
    result = model.fit(disp=False)

    trend_coef  = result.params["trend"]
    trend_se    = result.bse["trend"]
    trend_z     = result.tvalues["trend"]
    trend_pval  = result.pvalues["trend"]

    # Annual multiplicative rate change: exp(β_trend × 12)
    annual_multiplier    = np.exp(trend_coef * 12)
    annual_multiplier_lo = np.exp((trend_coef - 1.96 * trend_se) * 12)
    annual_multiplier_hi = np.exp((trend_coef + 1.96 * trend_se) * 12)

    # Fitted values + 95% CI
    pred        = result.get_prediction(X)
    pred_df     = pred.summary_frame(alpha=0.05)
    monthly     = monthly.copy()
    monthly["fitted"]   = pred_df["mean"].values
    monthly["fit_lo"]   = pred_df["mean_ci_lower"].values
    monthly["fit_hi"]   = pred_df["mean_ci_upper"].values

    # Overdispersion check: if Pearson chi² / df >> 1, data is overdispersed
    # (common for event counts — indicates latent clustering the model misses)
    pearson_chi2 = result.pearson_chi2
    df_resid     = result.df_resid
    dispersion   = pearson_chi2 / df_resid

    return {
        "label":               label,
        "status":              "ok",
        "result":              result,
        "monthly":             monthly,
        "nonzero_months":      nonzero,
        "trend_coef":          trend_coef,
        "trend_se":            trend_se,
        "trend_z":             trend_z,
        "trend_pval":          trend_pval,
        "trend_significant":   trend_pval < 0.05,
        "annual_multiplier":   annual_multiplier,
        "annual_mult_lo":      annual_multiplier_lo,
        "annual_mult_hi":      annual_multiplier_hi,
        "dispersion":          dispersion,
        "aic":                 result.aic,
        "mean_monthly_rate":   monthly["fitted"].mean(),
    }


# ---------------------------------------------------------------------------
# 4. Reporting artifact diagnostic
# ---------------------------------------------------------------------------
#
# Approach: compare trend magnitude across categories.
# A real climate signal should show up in categories with stable long-term
# reporting (volcanoes, severe storms). A trend that only appears in categories
# that gained data sources mid-window is more likely a reporting artifact.
#
# We also inspect the trend *before vs after* the suspected integration date
# to see if the trend is concentrated at the known boundary.

ARTIFACT_BREAKPOINTS = {
    "wildfires": "2017-01",  # IRWIN became dense; pre-period is thinner
    "floods":    "2019-01",  # FloodList coverage expanded
}


def artifact_diagnostic(fit: dict, category_id: str) -> dict:
    """
    For suspected artifact categories, split the series at the known breakpoint
    and compare pre/post trend. If the trend is concentrated post-breakpoint,
    flag it as a likely artifact.
    """
    if category_id not in ARTIFACT_BREAKPOINTS or fit["status"] != "ok":
        return {}

    monthly    = fit["monthly"].copy()
    breakpoint = pd.Period(ARTIFACT_BREAKPOINTS[category_id], freq="M")
    pre        = monthly[monthly["period"] < breakpoint]
    post       = monthly[monthly["period"] >= breakpoint]

    def _mean_rate(df):
        return df["count"].mean() if len(df) > 0 else np.nan

    pre_rate  = _mean_rate(pre)
    post_rate = _mean_rate(post)
    ratio     = post_rate / pre_rate if pre_rate > 0 else np.nan

    # Also fit trend on post-period only to check if trend persists
    # once the data source stabilized
    post_result = None
    if len(post) >= MIN_NONZERO_MONTHS:
        post["t_centered"] = post["t"] - post["t"].mean()
        X_post = build_design_matrix(post)
        try:
            m = sm.GLM(post["count"], X_post, family=sm.families.Poisson()).fit(disp=False)
            post_result = {
                "trend_coef": m.params["trend"],
                "trend_pval": m.pvalues["trend"],
            }
        except Exception:
            pass

    return {
        "breakpoint":           str(breakpoint),
        "pre_mean_rate":        round(pre_rate, 2),
        "post_mean_rate":       round(post_rate, 2),
        "post_pre_ratio":       round(ratio, 2) if ratio else None,
        "post_trend_coef":      post_result["trend_coef"] if post_result else None,
        "post_trend_pval":      post_result["trend_pval"] if post_result else None,
        "artifact_suspected":   ratio is not None and ratio > 2.0,
    }


# ---------------------------------------------------------------------------
# Stationarity tests
# ---------------------------------------------------------------------------
#
# We use two complementary tests:
#
# 1. Mann-Kendall (non-parametric): tests for a monotonic trend in the count
#    series without assuming any distributional form. Based on Kendall's tau
#    between the time index and the observations. p < 0.05 = trend detected.
#    tau > 0 = increasing, tau < 0 = decreasing.
#
# 2. GLM trend test (parametric): the p-value on beta_trend from the fitted
#    Poisson GLM — tests whether the log-rate changes linearly over time after
#    conditioning on the seasonal pattern. Already computed in fit_poisson().
#
# Together: if both agree there is a trend, it is more credible. If only the
# parametric test finds one, the seasonal decomposition may be doing the work.
# If only Mann-Kendall finds one, the trend may be non-linear.

def stationarity_suite(monthly: pd.DataFrame, fit: dict) -> dict:
    """
    Run Mann-Kendall trend test on monthly counts and summarise alongside
    the GLM trend test already stored in `fit`.
    """
    counts = monthly["count"].values.astype(float)
    t      = monthly["t"].values

    # Mann-Kendall via Kendall's tau against the time index
    tau, mk_pval = stats.kendalltau(t, counts)

    mk_direction = (
        "increasing" if tau > 0 and mk_pval < 0.05 else
        "decreasing" if tau < 0 and mk_pval < 0.05 else
        "none detected"
    )

    glm_pval  = fit.get("trend_pval",      float("nan"))
    glm_sig   = fit.get("trend_significant", False)
    glm_mult  = fit.get("annual_multiplier", float("nan"))

    # Year-to-year coefficient of variation in annual totals
    yearly_totals = monthly.groupby("year")["count"].sum()
    yearly_cv     = float(yearly_totals.std() / yearly_totals.mean()) if len(yearly_totals) > 1 else float("nan")

    both_agree = (mk_pval < 0.05) and glm_sig

    verdict = (
        "STATIONARY"     if not glm_sig and mk_pval >= 0.05 else
        "TREND (strong)" if both_agree                       else
        "TREND (GLM only; MK disagrees)" if glm_sig         else
        "TREND (MK only; GLM disagrees)"
    )

    return {
        "mann_kendall_tau":   round(float(tau), 4),
        "mann_kendall_pval":  round(float(mk_pval), 4),
        "mann_kendall_dir":   mk_direction,
        "glm_trend_pval":     round(float(glm_pval), 4),
        "glm_annual_mult":    round(float(glm_mult), 3),
        "yearly_cv":          round(yearly_cv, 3),
        "overdispersion":     round(fit.get("dispersion", float("nan")), 2),
        "verdict":            verdict,
    }


# ---------------------------------------------------------------------------
# Poisson deviance helpers (used by within-cell CV)
# ---------------------------------------------------------------------------

def _poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Poisson deviance. Handles y_true == 0 (0 * log(0) defined as 0)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 1e-10, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.where(y_true > 0, y_true * np.log(y_true / y_pred), 0.0)
    return float(2.0 * np.sum(log_term - (y_true - y_pred)))


def _null_deviance(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """Deviance of a model that predicts the training mean for every test point."""
    mu = np.mean(y_train) if len(y_train) > 0 else 1.0
    return _poisson_deviance(y_test, np.full_like(y_test, mu, dtype=float))


# ---------------------------------------------------------------------------
# Time series cross-validation (expanding window / walk-forward)
# ---------------------------------------------------------------------------
#
# Structure: the monthly series is split chronologically. For fold k:
#
#   [--- training ---][cutoff_k][--- test (test_size months) ---]
#
# The training window expands each fold (cutoff advances). The test window
# slides forward by the same step. This ensures no future data ever leaks
# into training.
#
# Cutoffs are evenly spaced from min_train months to (n_months - test_size).
#
# Two models are fit per fold:
#   - Full model:     intercept + month dummies + trend
#   - Seasonal only:  intercept + month dummies  (no trend)
#
# Skill score = 1 - (model_deviance / null_deviance)
# Null model  = predict the training mean for every test point.
# Skill delta = skill(full) - skill(seasonal) across folds.
#
# t_centered is recomputed relative to each training window's mean so that
# the trend coefficient is always anchored to the center of the seen data,
# not the full series, preventing extrapolation bias across folds.


def cross_validate_timeseries(
    monthly: pd.DataFrame,
    n_splits: int = 5,
    test_size: int = 3,
    min_train: int = 24,
) -> dict:
    """
    Expanding-window (walk-forward) time series cross-validation on monthly counts.

    Cutoffs are evenly spaced from min_train to (n_months - test_size).
    For each cutoff:
      - train = monthly.iloc[:cutoff]          (all history up to cutoff)
      - test  = monthly.iloc[cutoff:cutoff+test_size]  (next test_size months)

    Two models are evaluated per fold:
      - Full:     intercept + month dummies + linear trend
      - Seasonal: intercept + month dummies  (no trend)

    t_centered is re-anchored to each training window's mean to prevent
    extrapolation bias across folds.
    """
    n = len(monthly)
    if n < min_train + test_size:
        return {
            "status": "insufficient_data",
            "reason": f"only {n} months; need at least {min_train + test_size}",
        }

    # Evenly-spaced cutoff points: from min_train to (n - test_size), inclusive
    cutoff_first = min_train
    cutoff_last  = n - test_size
    if n_splits == 1 or cutoff_first == cutoff_last:
        cutoffs = [cutoff_first]
    else:
        step    = (cutoff_last - cutoff_first) / (n_splits - 1)
        cutoffs = sorted(set(int(round(cutoff_first + k * step)) for k in range(n_splits)))

    records_with    = []
    records_without = []
    trend_coefs     = []

    for fold_k, cutoff in enumerate(cutoffs):
        # -- 1. Slice train / test ------------------------------------------
        train = monthly.iloc[:cutoff].copy()
        test  = monthly.iloc[cutoff: cutoff + test_size].copy()
        if len(train) < 13 or len(test) == 0:
            continue

        y_train = train["count"].values.astype(float)
        y_test  = test["count"].values.astype(float)

        # -- 2. Re-center trend on training mean ----------------------------
        # This anchors the trend extrapolation to the midpoint of seen data,
        # not the full series — prevents systematic over/under-prediction.
        t_mean = train["t"].mean()
        train  = train.copy();  train["t_centered"] = train["t"] - t_mean
        test   = test.copy();   test["t_centered"]  = test["t"]  - t_mean

        # -- 3. Build design matrices ---------------------------------------
        X_tr_full = build_design_matrix(train)
        X_te_full = build_design_matrix(test)
        X_tr_seas = X_tr_full.drop(columns=["trend"])
        X_te_seas = X_te_full.drop(columns=["trend"])

        # -- 4. Null model: training mean for all test points ---------------
        null_pred = np.full(len(y_test), max(y_train.mean(), 1e-10))
        null_dev  = _poisson_deviance(y_test, null_pred)

        # -- 5. Fit full and seasonal-only models ---------------------------
        def _fit_predict(X_tr, X_te):
            try:
                m  = sm.GLM(y_train, X_tr, family=sm.families.Poisson()).fit(disp=False)
                mu = np.clip(m.predict(X_te), 1e-10, None)
                return mu, m, True
            except Exception:
                return np.full(len(y_test), np.nan), None, False

        mu_full, mdl_full, ok_full = _fit_predict(X_tr_full, X_te_full)
        mu_seas, _,        ok_seas = _fit_predict(X_tr_seas, X_te_seas)

        if ok_full and mdl_full is not None and "trend" in mdl_full.params:
            trend_coefs.append(float(mdl_full.params["trend"]))

        # -- 6. Metrics per model -------------------------------------------
        def _metrics(mu, ok):
            if not ok:
                return {}
            dev   = _poisson_deviance(y_test, mu)
            mae   = float(np.mean(np.abs(y_test - mu)))
            skill = float(np.clip(1.0 - dev / null_dev if null_dev > 0 else np.nan, -5.0, 1.0))
            return {
                "fold":         fold_k + 1,
                "cutoff":       str(monthly["period"].iloc[cutoff - 1]),
                "train_months": int(cutoff),
                "test_months":  int(len(y_test)),
                "deviance":     round(dev, 3),
                "null_dev":     round(null_dev, 3),
                "skill":        round(skill, 3),
                "mae":          round(mae, 3),
            }

        records_with.append(_metrics(mu_full, ok_full))
        records_without.append(_metrics(mu_seas, ok_seas))

    # -- 7. Aggregate across folds ------------------------------------------
    def _agg(records):
        valid = [r for r in records if r]
        if not valid:
            return {}
        df_r = pd.DataFrame(valid)
        return {
            "mean_skill":    round(float(df_r["skill"].mean()),    3),
            "mean_mae":      round(float(df_r["mae"].mean()),      3),
            "mean_deviance": round(float(df_r["deviance"].mean()), 3),
            "folds":         valid,
        }

    with_agg    = _agg(records_with)
    without_agg = _agg(records_without)

    if not with_agg or not without_agg:
        return {"status": "insufficient_data", "reason": "no valid folds completed"}

    trend_helps = with_agg["mean_deviance"] < without_agg["mean_deviance"]
    skill_delta = round(with_agg["mean_skill"] - without_agg["mean_skill"], 4)

    trend_stability = {}
    if trend_coefs:
        trend_stability = {
            "coefs":  [round(c, 4) for c in trend_coefs],
            "mean":   round(float(np.mean(trend_coefs)), 4),
            "std":    round(float(np.std(trend_coefs)),  4),
            "cv_pct": round(
                float(np.std(trend_coefs) / abs(np.mean(trend_coefs)) * 100), 1
            ) if np.mean(trend_coefs) != 0 else None,
        }

    return {
        "status":             "ok",
        "n_splits":           len(cutoffs),
        "test_size":          test_size,
        "with_trend":         with_agg,
        "without_trend":      without_agg,
        "trend_improves_oos": trend_helps,
        "skill_delta":        skill_delta,
        "trend_stability":    trend_stability,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_category(fit: dict, artifact: dict, category_id: str, save_dir: Path):
    if fit["status"] != "ok":
        return

    monthly      = fit["monthly"]
    label        = fit["label"]
    model        = fit["result"]
    cv_ts        = fit.get("cv_timeseries", {})
    st           = fit.get("stationarity",  {})
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, axes = plt.subplots(
        3, 1, figsize=(13, 11),
        gridspec_kw={"height_ratios": [3, 1.2, 1.4]},
    )
    fig.suptitle(f"Inhomogeneous Poisson  |  {label}", fontsize=13, fontweight="bold")

    # ------------------------------------------------------------------ #
    # Panel 0: time series — observed counts + full-data fitted model
    # ------------------------------------------------------------------ #
    ax = axes[0]
    x  = monthly["period"].dt.to_timestamp()
    ax.fill_between(x, monthly["fit_lo"], monthly["fit_hi"],
                    alpha=0.2, color="steelblue", label="95% CI (full fit)")
    ax.plot(x, monthly["fitted"], color="steelblue", linewidth=1.8, label="Fitted rate")
    ax.bar(x, monthly["count"], width=20, color="gray", alpha=0.4, label="Observed count")

    if artifact.get("breakpoint"):
        bp = pd.Period(artifact["breakpoint"], freq="M").to_timestamp()
        ax.axvline(bp, color="darkorange", linestyle="--", linewidth=1.2,
                   label=f"Suspected source expansion ({artifact['breakpoint']})")

    verdict = st.get("verdict", "")
    ax.set_title(
        f"Trend: x{fit['annual_multiplier']:.2f}/yr  p={fit['trend_pval']:.3f}  "
        f"overdispersion={fit['dispersion']:.1f}  |  {verdict}  "
        f"(MK p={st.get('mann_kendall_pval','?'):.3f})",
        fontsize=9,
    )
    ax.set_ylabel("Events / month")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # ------------------------------------------------------------------ #
    # Panel 1: seasonal profile with SE error bars
    # ------------------------------------------------------------------ #
    ax2 = axes[1]
    seasonal = []
    for m in range(1, 13):
        key  = f"m_{m}"
        coef = model.params.get(key, 0.0)
        se   = model.bse.get(key,   0.0)
        rate = np.exp(coef)
        seasonal.append((rate, rate * se))

    rates  = [s[0] for s in seasonal]
    errs   = [s[1] for s in seasonal]
    colors = ["steelblue" if r >= 1 else "salmon" for r in rates]
    ax2.bar(month_labels, rates, color=colors, alpha=0.8,
            yerr=errs, error_kw={"linewidth": 0.8, "capsize": 3})
    ax2.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Relative rate\n(Jan = 1.0)")
    ax2.set_title("Seasonal profile  (error bars = 1 SE)", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    # ------------------------------------------------------------------ #
    # Panel 2: time series CV — per-fold skill score, with vs without trend
    # ------------------------------------------------------------------ #
    ax3 = axes[2]
    if cv_ts.get("status") == "ok":
        folds_w  = cv_ts["with_trend"].get("folds",    [])
        folds_wo = cv_ts["without_trend"].get("folds", [])
        n_f = min(len(folds_w), len(folds_wo))
        if n_f > 0:
            xlabels  = [f["cutoff"] for f in folds_w[:n_f]]
            skills_w  = [f["skill"] for f in folds_w[:n_f]]
            skills_wo = [f["skill"] for f in folds_wo[:n_f]]
            xf = np.arange(n_f)
            bw = 0.35
            ax3.bar(xf - bw/2, skills_w,  width=bw, color="steelblue", alpha=0.8, label="With trend")
            ax3.bar(xf + bw/2, skills_wo, width=bw, color="salmon",    alpha=0.8, label="Without trend")
            ax3.set_xticks(xf)
            ax3.set_xticklabels(xlabels, fontsize=7, rotation=20, ha="right")
            ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax3.set_ylabel("Skill score")
            ax3.legend(fontsize=8)
            ax3.grid(axis="y", alpha=0.3)
    ts_stab = cv_ts.get("trend_stability", {})
    ax3.set_title(
        f"Time series CV ({cv_ts.get('n_splits','?')} folds, {cv_ts.get('test_size','?')}-month test windows): "
        f"skill per fold  [delta={cv_ts.get('skill_delta', float('nan')):.3f}, "
        f"trend coef cv%={ts_stab.get('cv_pct','?')}%]",
        fontsize=9,
    )

    plt.tight_layout()
    out = save_dir / f"poisson_{category_id}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {out.name}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(fits: dict, artifacts: dict):
    print("\n" + "=" * 90)
    print(f"{'Category':<20} {'Status':<10} {'Mean rate/mo':<14} {'Annual x':<12} "
          f"{'95% CI':<20} {'p-value':<10} {'Overdisp.':<10} {'Artifact?'}")
    print("=" * 90)

    for cat_id, fit in fits.items():
        if fit["status"] != "ok":
            print(f"{fit['label']:<20} {'SKIP':<10} {'—':<12} {'—':<12} {'—':<20} {'—':<10} {'—':<10}")
            continue

        art   = artifacts.get(cat_id, {})
        flag  = "SUSPECTED" if art.get("artifact_suspected") else ""
        sig   = "*" if fit["trend_significant"] else ""
        ci    = f"[{fit['annual_mult_lo']:.2f}–{fit['annual_mult_hi']:.2f}]"

        print(
            f"{fit['label']:<20} {'ok':<10} {fit['mean_monthly_rate']:<14.1f} "
            f"{fit['annual_multiplier']:<12.2f} {ci:<20} "
            f"{fit['trend_pval']:<10.3f} {fit['dispersion']:<10.2f} {flag}{sig}"
        )

    print("=" * 90)
    print("* = trend significant at p<0.05")
    print("Overdispersion > 2.0 suggests clustering — Poisson may underestimate uncertainty")
    print("Artifact? SUSPECTED if post/pre-breakpoint mean rate ratio > 2.0\n")


def _cv_row(fit: dict, cv_key: str) -> str:
    cv = fit.get(cv_key, {})
    if not cv or cv.get("status") != "ok":
        return f"{fit['label']:<20} {'N/A'}"
    w   = cv["with_trend"]
    wo  = cv["without_trend"]
    w_s  = f"{w.get('mean_skill', 0):+.3f} / {w.get('mean_mae', 0):.2f} / {w.get('mean_ci_coverage', 0):.2f}"
    wo_s = f"{wo.get('mean_skill', 0):+.3f} / {wo.get('mean_mae', 0):.2f} / {wo.get('mean_ci_coverage', 0):.2f}"
    helps = "YES" if cv["trend_improves_oos"] else "NO"
    return f"{fit['label']:<20} {w_s:<32} {wo_s:<32} {helps:<18} {cv['skill_delta']:+.4f}"


def print_stationarity_summary(fits: dict):
    print("\nStationarity Analysis")
    print("=" * 105)
    print(f"{'Category':<20} {'MK tau':<10} {'MK p':<10} {'MK direction':<22} "
          f"{'GLM p':<10} {'Annual x':<10} {'Overdisp.':<12} {'Verdict'}")
    print("=" * 105)
    for cat_id, fit in fits.items():
        st = fit.get("stationarity", {})
        cfg = CATEGORY_CONFIG.get(cat_id, {})
        if cfg.get("skip"):
            print(f"{fit['label']:<20} SKIPPED -- {cfg.get('note','')}")
            continue
        if not st:
            print(f"{fit['label']:<20} N/A")
            continue
        print(
            f"{fit['label']:<20} {st['mann_kendall_tau']:<10.4f} {st['mann_kendall_pval']:<10.4f} "
            f"{st['mann_kendall_dir']:<22} {st['glm_trend_pval']:<10.4f} "
            f"{st['glm_annual_mult']:<10.3f} {st['overdispersion']:<12.2f} {st['verdict']}"
        )
    print("=" * 105)
    print("MK = Mann-Kendall (non-parametric). GLM = parametric trend test from Poisson fit.")
    print("Overdispersion > 2 means Poisson CIs are too narrow (clustering present).\n")


def print_cv_summary(fits: dict):
    print("\nTime Series CV (expanding window, 3-month test windows)")
    print("Train = all months up to cutoff | Test = next 3 months after cutoff.")
    print("Skill = 1 - (model deviance / null deviance). Null = training mean.")
    print("Skill delta > 0 means the trend term improves out-of-sample fit.")
    print("=" * 110)
    print(f"{'Category':<20} {'With trend':<24} {'Without trend':<24} "
          f"{'Trend helps?':<14} {'Skill delta':<14} {'Trend coef cv%'}")
    print(f"{'':20} {'skill / MAE':<24} {'skill / MAE':<24}")
    print("=" * 110)

    for cat_id, fit in fits.items():
        cv  = fit.get("cv_timeseries", {})
        cfg = CATEGORY_CONFIG.get(cat_id, {})
        if cfg.get("skip"):
            print(f"{fit['label']:<20} SKIPPED")
            continue
        if not cv or cv.get("status") != "ok":
            reason = cv.get("reason", "insufficient data") if cv else "not run"
            print(f"{fit['label']:<20} N/A  ({reason})")
            continue
        w    = cv["with_trend"]
        wo   = cv["without_trend"]
        ts   = cv.get("trend_stability", {})
        w_s  = f"{w.get('mean_skill',0):+.3f} / {w.get('mean_mae',0):.2f}"
        wo_s = f"{wo.get('mean_skill',0):+.3f} / {wo.get('mean_mae',0):.2f}"
        helps = "YES" if cv["trend_improves_oos"] else "NO"
        stab  = f"{ts['cv_pct']}%" if ts.get("cv_pct") is not None else "N/A"
        print(f"{fit['label']:<20} {w_s:<24} {wo_s:<24} {helps:<14} "
              f"{cv['skill_delta']:<14.4f} {stab}")

        folds_w = w.get("folds", [])
        if folds_w:
            fold_line = "  folds: " + "  ".join(
                f"[cutoff={f['cutoff']}, skill={f['skill']:+.3f}]" for f in folds_w
            )
            print(fold_line)

    print("=" * 110)
    print("Trend coef cv% = std/mean across folds. High (>50%) = unstable trend estimate.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EONET inhomogeneous Poisson model")
    parser.add_argument("--category", nargs="+", default=list(CATEGORY_CONFIG.keys()),
                        help="Category IDs to run")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch from API (ignore cache)")
    parser.add_argument("--no-plots", action="store_true", help="Skip plots")
    args = parser.parse_args()

    categories = {k: CATEGORY_CONFIG[k] for k in args.category if k in CATEGORY_CONFIG}
    if not categories:
        print(f"No valid categories. Choose from: {list(CATEGORY_CONFIG.keys())}")
        sys.exit(1)

    plot_dir = Path(__file__).parent / "plots"
    plot_dir.mkdir(exist_ok=True)

    fits = {}

    print(f"\nFetching events (raw window: {FETCH_START} to {FETCH_END})")
    print("-" * 60)

    for cat_id, cfg in categories.items():
        label = cfg["label"]

        # 1. Fetch raw data (uses cache)
        df_raw = fetch_category(cat_id, refresh=args.refresh)

        # Skip categories flagged as unanalyzable
        if cfg.get("skip"):
            print(f"  [{cat_id}] SKIPPED -- {cfg['note']}")
            fits[cat_id] = {"label": label, "status": "skipped", "note": cfg["note"]}
            continue

        # 2. Apply analysis window filter
        df = filter_to_analysis_window(df_raw, cat_id)
        n_dropped = len(df_raw) - len(df)
        win_start = cfg.get("analysis_start") or FETCH_START
        win_end   = cfg.get("analysis_end")   or FETCH_END
        print(f"  [{cat_id}] analysis window {win_start} to {win_end}  "
              f"({len(df)} events, {n_dropped} dropped)")
        if cfg.get("note"):
            print(f"    note: {cfg['note']}")

        # 3. Aggregate to monthly counts
        monthly = to_monthly_counts(df, cat_id)

        # 4. Fit
        print(f"  [{cat_id}] fitting model ...", end=" ", flush=True)
        fit = fit_poisson(monthly, label)
        fit["_events_df"] = df
        fits[cat_id] = fit

        if fit["status"] == "ok":
            print(f"done  (mean={fit['mean_monthly_rate']:.1f}/mo, "
                  f"trend p={fit['trend_pval']:.3f}, disp={fit['dispersion']:.2f})")
        else:
            print(f"skipped ({fit['nonzero_months']} non-zero months < {MIN_NONZERO_MONTHS})")
            continue

        # 5. Stationarity tests
        st = stationarity_suite(monthly, fit)
        fit["stationarity"] = st
        print(f"  [{cat_id}] stationarity: {st['verdict']}  "
              f"(MK p={st['mann_kendall_pval']:.3f}, GLM p={st['glm_trend_pval']:.3f})")

        # 6. Time series CV (expanding window)
        print(f"  [{cat_id}] time series CV ...", end=" ", flush=True)
        cv = cross_validate_timeseries(fit["monthly"], n_splits=5, test_size=3, min_train=24)
        fit["cv_timeseries"] = cv
        if cv["status"] == "ok":
            ts = cv.get("trend_stability", {})
            print(f"done  (skill delta={cv['skill_delta']:+.4f}, "
                  f"trend cv%={ts.get('cv_pct', 'N/A')}%)")
        else:
            print(f"skipped: {cv.get('reason', 'unknown')}")

        # 7. Plot
        if not args.no_plots:
            plot_category(fit, {}, cat_id, plot_dir)

    # Summary tables
    print_stationarity_summary(fits)
    print_cv_summary(fits)

    # Save JSON
    def _jsonify(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonify(v) for v in obj]
        return obj

    output = {}
    for cat_id, fit in fits.items():
        entry = {k: v for k, v in fit.items() if k not in ("result", "monthly", "_events_df")}
        output[cat_id] = _jsonify(entry)

    out_path = Path(__file__).parent / "poisson_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved: {out_path.name}")


if __name__ == "__main__":
    main()
