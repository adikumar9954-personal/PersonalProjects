"""
models/validator.py  (v2 - unchanged logic, cleaner interface)
"""

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from sklearn.linear_model import LassoCV

# Offensive - leaguehustlestatsplayer + leaguedashplayerptshot
OFFENSIVE_FEATURES = [
    ("SCREEN_ASSISTS",  +1, 2.0, "Screen assists per game"),
    ("DEFLECTIONS",     +1, 1.5, "Deflections per game"),
    ("EFG_PCT",         +1, 2.0, "Effective FG%"),
    ("FG3_PCT",         +1, 1.0, "3pt FG%"),
    ("CHARGES_DRAWN",   +1, 1.0, "Charges drawn per game"),
]
# Defensive - lineup-level team on/off (true on vs same-team off)
DEFENSIVE_FEATURES = [
    ("opp_fg_pct_delta",     -1, 3.0, "Team opp FG% on vs off court"),
    ("opp_fg3a_rate_delta",  -1, 2.0, "Team opp 3PA rate on vs off court"),
    ("opp_tov_rate_delta",   +1, 2.0, "Team forced TOV rate on vs off court"),
    ("D_FGA",                +1, 1.0, "Defended shot attempts per game"),
    ("CONTESTED_SHOTS",      +1, 1.5, "Contested shots per game"),
]
# Playmaking - passing tracking + lineup on/off (team-wide effects)
# Columns sourced from: LeagueDashPtStats (Passing/Drives) + data/playmaking.py
PLAYMAKING_FEATURES = [
    ("POTENTIAL_AST",   +1, 2.5, "Potential assists per game"),
    ("AST_ADJ",         +1, 2.0, "Adjusted assists per game"),
    ("rim_rate_delta",  +1, 2.5, "Team rim rate on vs off court"),
    ("fg3a_rate_delta", +1, 2.5, "Team 3PA rate on vs off court"),
    ("tov_rate_delta",  -1, 2.5, "Team TOV rate on vs off court (lower = better)"),
]


def compute_percentiles(tracking: pd.DataFrame, min_games: int = 20) -> pd.DataFrame:
    if tracking.empty:
        return tracking
    if "GP" in tracking.columns:
        df = tracking[tracking["GP"] >= min_games].copy()
    else:
        df = tracking.copy()
    all_features = OFFENSIVE_FEATURES + DEFENSIVE_FEATURES + PLAYMAKING_FEATURES
    for col, direction, _, _ in all_features:
        if col not in df.columns:
            df[f"{col}_pct"] = np.nan
            continue
        vals = df[col].fillna(df[col].median())
        pcts = vals.apply(lambda v: percentileofscore(vals.dropna(), v, kind="rank"))
        df[f"{col}_pct"] = 100 - pcts if direction == -1 else pcts
    return df


def score_player(player_id: str, rapm: float, tracking_pct: pd.DataFrame) -> dict:
    row = tracking_pct[tracking_pct["PLAYER_ID"].astype(str) == str(player_id)]
    if row.empty:
        return {"support_score": None, "o_support": None, "d_support": None,
                "playmaking_support": None, "red_flags": [], "green_flags": []}
    row = row.iloc[0]

    def weighted_avg(features):
        scores, weights = [], []
        for col, _, w, _ in features:
            v = row.get(f"{col}_pct")
            if v is not None and not pd.isna(v):
                scores.append(v); weights.append(w)
        return np.average(scores, weights=weights) if scores else 50.0

    o_support   = weighted_avg(OFFENSIVE_FEATURES)
    d_support   = weighted_avg(DEFENSIVE_FEATURES)
    pm_support  = weighted_avg(PLAYMAKING_FEATURES)
    support     = 0.40 * o_support + 0.35 * d_support + 0.25 * pm_support

    # Columns that are 0-1 proportions - display as "45.1%"
    _PCT_COLS   = {"EFG_PCT", "FG3_PCT"}
    # Columns that are small decimal deltas - display as "+2.3%"
    _DELTA_COLS = {
        "fg3a_rate_delta", "tov_rate_delta", "rim_rate_delta",
        "opp_fg_pct_delta", "opp_fg3a_rate_delta", "opp_tov_rate_delta",
    }

    def _fmt_raw(col, val):
        if val is None or pd.isna(val):
            return "-"
        if col in _PCT_COLS:
            return f"{val * 100:.1f}%"
        if col in _DELTA_COLS:
            return f"{val * 100:+.1f}%"
        return f"{val:.2f}"

    # Emit every metric as "{label}: {raw_value} ({pct}th pct)"
    # green if good percentile (>=65), red if poor (<=35), skipped if missing
    green, red = [], []
    for col, _, _, label in OFFENSIVE_FEATURES + DEFENSIVE_FEATURES + PLAYMAKING_FEATURES:
        pct = row.get(f"{col}_pct")
        if pct is None or pd.isna(pct):
            continue
        raw_str = _fmt_raw(col, row.get(col))
        entry = f"{label}: {raw_str} ({pct:.0f}th pct)"
        if pct >= 65:
            green.append(entry)
        elif pct <= 35:
            red.append(entry)

    return {"support_score": round(support, 1), "o_support": round(o_support, 1),
            "d_support": round(d_support, 1), "playmaking_support": round(pm_support, 1),
            "red_flags": red, "green_flags": green}


def fit_support_model(tracking_pct: pd.DataFrame, rapm_qualified: pd.DataFrame):
    """
    Fit a LassoCV to predict RAPM from tracking percentile features.
    Automatically zeros out uninformative features.

    Returns (model, feature_cols) - model is None if insufficient data.
    """
    all_features = OFFENSIVE_FEATURES + DEFENSIVE_FEATURES + PLAYMAKING_FEATURES
    pct_cols = [f"{col}_pct" for col, _, _, _ in all_features]
    avail = [c for c in pct_cols if c in tracking_pct.columns]

    tp = tracking_pct.copy()
    tp["_pid"] = tp["PLAYER_ID"].astype(str)
    rq = rapm_qualified.copy()
    rq["_pid"] = rq["player_id"].astype(str)

    merged = tp.merge(rq[["_pid", "rapm"]], on="_pid", how="inner")
    # Fill missing percentile features with 50 (league-average proxy) rather than
    # dropping the whole row - only require the target to be present.
    for c in avail:
        if c in merged.columns:
            merged[c] = merged[c].fillna(50.0)
        else:
            merged[c] = 50.0
    merged = merged.dropna(subset=["rapm"])

    if len(merged) < 20:
        print(f"[lasso] only {len(merged)} complete rows - falling back to manual weights")
        return None, avail

    X = merged[avail].values
    y = merged["rapm"].values

    model = LassoCV(cv=5, max_iter=10000, random_state=42)
    model.fit(X, y)

    n_selected = int((np.abs(model.coef_) > 1e-6).sum())
    r2 = model.score(X, y)
    print(f"[lasso] alpha={model.alpha_:.4f}  "
          f"{n_selected}/{len(avail)} features selected  R²={r2:.3f}")
    ranked = sorted(zip(avail, model.coef_), key=lambda x: abs(x[1]), reverse=True)
    for col, w in ranked:
        if abs(w) > 1e-6:
            orig = col.replace("_pct", "")
            label = next((l for c, _, _, l in all_features if c == orig), orig)
            print(f"  {label}: {w:+.5f}")

    return model, avail


def validate_all_players(
    rapm_results: pd.DataFrame,
    tracking: pd.DataFrame,
    playmaking_df: pd.DataFrame = None,
    min_games: int = 5,
    min_minutes: float = 10.0,
) -> pd.DataFrame:
    # Merge lineup on/off playmaking metrics into tracking before computing percentiles
    if playmaking_df is not None and not playmaking_df.empty and "PLAYER_ID" in tracking.columns:
        pm_cols = ["PLAYER_ID"] + [c for c in playmaking_df.columns if c != "PLAYER_ID"]
        pm = playmaking_df[pm_cols].copy()
        # Normalise both sides to str so int64 vs str doesn't break the join
        tracking = tracking.copy()
        tracking["PLAYER_ID"] = tracking["PLAYER_ID"].astype(str)
        pm["PLAYER_ID"] = pm["PLAYER_ID"].astype(str)
        tracking = tracking.merge(pm, on="PLAYER_ID", how="left")
        print(f"[validator] merged playmaking on/off ({len(playmaking_df)} players)")
    print("[validator] Computing tracking percentiles ...")
    tracking_pct = compute_percentiles(tracking, min_games)
    qualified    = rapm_results[rapm_results["minutes"] >= min_minutes]
    rapm_dist    = qualified["rapm"].values

    # Fit Lasso support model and batch-predict for all players in tracking_pct
    lasso_model, lasso_cols = fit_support_model(tracking_pct, qualified)
    lasso_pct_map = {}
    if lasso_model is not None:
        tp_pred = tracking_pct.copy()
        for c in lasso_cols:
            if c not in tp_pred.columns:
                tp_pred[c] = 50.0
            else:
                tp_pred[c] = tp_pred[c].fillna(50.0)
        raw_preds = lasso_model.predict(tp_pred[lasso_cols].values)
        pred_pcts = np.array([
            percentileofscore(raw_preds, v, kind="rank") for v in raw_preds
        ])
        lasso_pct_map = dict(zip(tp_pred["PLAYER_ID"].astype(str), pred_pcts))

    rows = []
    for _, p in qualified.iterrows():
        s = score_player(p["player_id"], p["rapm"], tracking_pct)
        # Override support_score with Lasso prediction if available
        pid_str = str(p["player_id"])
        if pid_str in lasso_pct_map:
            s["support_score"] = round(lasso_pct_map[pid_str], 1)
        rp = percentileofscore(rapm_dist, p["rapm"], kind="rank")
        rows.append({
            "player_id": p["player_id"], "player_name": p["player_name"],
            "minutes": p["minutes"], "prior": p["prior"], "rapm": p["rapm"],
            "rapm_adjustment": p.get("rapm_adjustment", p["rapm"] - p["prior"]),
            "rapm_pct": round(rp, 1), **s,
        })

    df = pd.DataFrame(rows)
    df["mispriced_up"]   = (df["rapm_pct"] > 65) & (df["support_score"] < 40)
    df["mispriced_down"] = (df["rapm_pct"] < 35) & (df["support_score"] > 60)
    print(f"[validator] {len(df)} players  "
          f"overvalued={df['mispriced_up'].sum()}  "
          f"undervalued={df['mispriced_down'].sum()}")
    return df.sort_values("rapm", ascending=False).reset_index(drop=True)
