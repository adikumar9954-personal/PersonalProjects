"""
data/date_split.py
==================
Runs the pipeline over two date windows and compares results.
Useful for before/after analysis - e.g. pre vs post injury return.

Usage in your test file:
    from data.date_split import compare_windows

    results = compare_windows(
        season="2025-26",
        split_date="2025-12-01",   # date dividing early vs recent
        max_games=100,
    )
    # results["early"]  - pipeline output for games before split_date
    # results["recent"] - pipeline output for games after split_date
    # results["comparison"] - side-by-side player DataFrame
"""

import time
import pandas as pd
from pathlib import Path
from data.ingest import (
    CACHE_DIR, NBA_API_DELAY, _load_or_fetch,
    get_box_scores, get_advanced_box, get_all_tracking,
    get_best_prior_target,
)
from data.stint_matrix import build_stint_matrix
from models.rapm import build_prior_model, predict_prior, fit_rapm
from models.validator import validate_all_players
from models.lineup_synergy import compute_synergy, compute_pairwise_compatibility


def get_stints_daterange(
    season: str,
    date_from: str = "",
    date_to: str = "",
    label: str = "window",
) -> pd.DataFrame:
    """
    Fetch lineup stints for a specific date window using
    LeagueDashLineups date_from_nullable / date_to_nullable params.

    date_from / date_to format: "MM/DD/YYYY"  e.g. "12/01/2025"
    """
    from nba_api.stats.endpoints import leaguedashlineups

    safe_label = label.replace("/", "-").replace(" ", "_")
    path = CACHE_DIR / f"stints_lineups_{season.replace('-','_')}_{safe_label}.parquet"
    if path.exists():
        print(f"  [cache] stints ({label})")
        return pd.read_parquet(path)

    print(f"  [fetch] stints for window: {date_from or 'season start'} -> {date_to or 'today'}")

    df = None
    for attempt in range(3):
        try:
            ep = leaguedashlineups.LeagueDashLineups(
                season=season,
                season_type_all_star="Regular Season",
                per_mode_detailed="Totals",
                measure_type_detailed_defense="Base",
                group_quantity="5",
                date_from_nullable=date_from,
                date_to_nullable=date_to,
            )
            df = ep.get_data_frames()[0]
            break
        except Exception as e:
            wait = (attempt + 1) * 20
            print(f"  [retry {attempt+1}/3] {e.__class__.__name__} - waiting {wait}s")
            time.sleep(wait)

    if df is None or df.empty:
        print(f"  [warning] no data for window {label}")
        return pd.DataFrame()

    time.sleep(NBA_API_DELAY)
    print(f"  [lineups] {len(df)} rows for {label}")

    def parse_ids(group_id):
        try:
            return [int(x) for x in str(group_id).split("-") if x.strip()]
        except Exception:
            return []

    id_col = "GROUP_ID" if "GROUP_ID" in df.columns else df.columns[1]
    df["home_players"] = df[id_col].apply(parse_ids)
    df["away_players"] = df[id_col].apply(lambda x: [])
    df["game_id"]      = f"agg_{label}"

    poss_col = next((c for c in ["POSS", "FGA", "FGM"] if c in df.columns), None)
    df["possessions"] = df[poss_col].fillna(50) if poss_col else 50

    pt_col = next((c for c in ["PLUS_MINUS"] if c in df.columns), None)
    df["point_diff"] = df[pt_col].fillna(0) if pt_col else 0

    min_col = next((c for c in ["MIN", "MINUTES"] if c in df.columns), None)
    df["minutes"] = df[min_col].fillna(0) if min_col else 0

    df = df[df["minutes"] >= 2].reset_index(drop=True)
    df.to_parquet(path, index=False)
    return df


def run_window(
    season: str,
    date_from: str,
    date_to: str,
    label: str,
    box100: pd.DataFrame,
    box_adv: pd.DataFrame,
    tracking: pd.DataFrame,
    prior_targets: pd.DataFrame,
) -> dict:
    """Run the full RAPM + validation pipeline for one date window."""
    print(f"\n--Window: {label} ({date_from or 'start'} -> {date_to or 'now'}) --")

    stints = get_stints_daterange(season, date_from, date_to, label)
    if stints.empty:
        return {}

    pm, scaler, feature_cols = build_prior_model(box100, box_adv, prior_targets)
    prior_df = predict_prior(pm, scaler, feature_cols, box100, box_adv)

    X, y, weights, enc = build_stint_matrix(stints)
    rapm_results, alpha = fit_rapm(X, y, weights, enc, prior_df)
    validated = validate_all_players(rapm_results, tracking)
    synergy   = compute_synergy(stints, rapm_results, min_poss=30.0)
    pairs     = compute_pairwise_compatibility(stints, rapm_results, min_shared_poss=20.0)

    return {
        "label":     label,
        "rapm":      rapm_results,
        "validated": validated,
        "synergy":   synergy,
        "pairs":     pairs,
        "alpha":     alpha,
    }


def compare_windows(
    season: str = "2025-26",
    split_date: str = "2025-12-01",
    date_fmt: str = "%Y-%m-%d",
    early_start: str = "",
    recent_end: str = "",
) -> dict:
    """
    Run pipeline on two windows split at split_date and compare.

    split_date: "YYYY-MM-DD" - divides early vs recent
    Returns dict with "early", "recent", "comparison" keys.
    """
    from datetime import datetime

    # Convert YYYY-MM-DD to MM/DD/YYYY for nba_api
    def to_nba_fmt(d):
        if not d: return ""
        return datetime.strptime(d, "%Y-%m-%d").strftime("%m/%d/%Y")

    date_from_early  = to_nba_fmt(early_start)
    date_to_early    = to_nba_fmt(split_date)
    date_from_recent = to_nba_fmt(split_date)
    date_to_recent   = to_nba_fmt(recent_end)

    # Shared data - fetch once
    print("\n--Shared data --")
    box100        = get_box_scores(season)
    box_adv       = get_advanced_box(season)
    tracking      = get_all_tracking(season)
    prior_targets = get_best_prior_target(season, box100)

    # Run both windows
    early  = run_window(season, date_from_early,  date_to_early,
                        f"early_{split_date}",  box100, box_adv, tracking, prior_targets)
    time.sleep(3)
    recent = run_window(season, date_from_recent, date_to_recent,
                        f"recent_{split_date}", box100, box_adv, tracking, prior_targets)

    # Build comparison DataFrame
    comparison = pd.DataFrame()
    if early and recent:
        e = early["rapm"][["player_id","player_name","rapm","prior"]].copy()
        r = recent["rapm"][["player_id","player_name","rapm","prior"]].copy()
        e = e.rename(columns={"rapm": "rapm_early"})
        r = r.rename(columns={"rapm": "rapm_recent", "prior": "prior_recent"})
        comparison = e.merge(r[["player_id","rapm_recent"]], on="player_id", how="inner")
        comparison["rapm_delta"] = comparison["rapm_recent"] - comparison["rapm_early"]
        comparison = comparison.sort_values("rapm_delta", ascending=False)
        print(f"\n--Comparison: {len(comparison)} players in both windows --")

    return {
        "early":      early,
        "recent":     recent,
        "comparison": comparison,
    }


def print_player_comparison(
    name: str,
    results: dict,
    top_n_pairs: int = 8,
) -> None:
    """Print a side-by-side breakdown of a player across two windows."""
    first = name.split()[0]
    comp  = results["comparison"]
    early  = results["early"]
    recent = results["recent"]

    print(f"\n{'='*68}")
    print(f"  {name.upper()} - WINDOW COMPARISON")
    print(f"{'='*68}")

    # RAPM comparison
    row = comp[comp["player_name"].str.contains(first, case=False)]
    if row.empty:
        print("  Not found in both windows")
        return
    row = row.iloc[0]

    print(f"\n  {'Metric':<20} {'Early':>10}  {'Recent':>10}  {'Delta':>10}")
    print(f"  {'-'*20} {'-'*10}  {'-'*10}  {'-'*10}")
    print(f"  {'RAPM':<20} {row['rapm_early']:>+10.2f}  {row['rapm_recent']:>+10.2f}  "
          f"{row['rapm_delta']:>+10.2f}")

    # Validation scores
    for label, window in [("Early", early), ("Recent", recent)]:
        if not window: continue
        v = window["validated"]
        vrow = v[v["player_name"].str.contains(first, case=False)]
        if vrow.empty: continue
        vrow = vrow.iloc[0]
        print(f"  {label+' support':<20} {vrow.get('support_score') or 'N/A':>10}  "
              f"off={vrow.get('o_support') or 'N/A'}  def={vrow.get('d_support') or 'N/A'}")

    # Best pairings in each window
    for label, window in [("Early", early), ("Recent", recent)]:
        if not window: continue
        pairs = window["pairs"]
        p = pairs[
            pairs["player_a"].str.contains(first, case=False) |
            pairs["player_b"].str.contains(first, case=False)
        ].sort_values("compat_shrunk", ascending=False).head(top_n_pairs)

        print(f"\n  Best pairings - {label}:")
        print(f"  {'Partner':<26} {'Poss':>5}  {'Shrunk':>7}  {'Conf':>5}")
        print(f"  {'-'*26} {'-'*5}  {'-'*7}  {'-'*5}")
        for _, pr in p.iterrows():
            partner = (pr["player_b"]
                      if first.lower() in pr["player_a"].lower()
                      else pr["player_a"])
            print(f"  {partner:<26} {pr['shared_poss']:>5.0f}  "
                  f"{pr['compat_shrunk']:>+7.2f}  {pr['confidence']:>4.0f}%")

    # Most improved lineups
    if early and recent:
        e_syn = early["synergy"].copy()
        r_syn = recent["synergy"].copy()
        e_syn["key"] = e_syn["lineup_key"].astype(str)
        r_syn["key"] = r_syn["lineup_key"].astype(str)

        e_p = e_syn[e_syn["player_names"].apply(
            lambda names: any(first.lower() in n.lower() for n in names)
        )][["key","synergy_shrunk"]].rename(columns={"synergy_shrunk":"early_syn"})
        r_p = r_syn[r_syn["player_names"].apply(
            lambda names: any(first.lower() in n.lower() for n in names)
        )][["key","synergy_shrunk","player_names","possessions"]].rename(
            columns={"synergy_shrunk":"recent_syn"})

        merged = r_p.merge(e_p, on="key", how="left").fillna(0)
        merged["improvement"] = merged["recent_syn"] - merged["early_syn"]
        merged = merged.sort_values("recent_syn", ascending=False).head(5)

        print(f"\n  Best lineups - Recent window:")
        print(f"  {'Players':<55} {'Poss':>5}  {'Shrunk':>7}")
        print(f"  {'-'*55} {'-'*5}  {'-'*7}")
        for _, lr in merged.iterrows():
            players = ", ".join(lr["player_names"])
            if len(players) > 55: players = players[:52] + "..."
            print(f"  {players:<55} {lr['possessions']:>5.0f}  "
                  f"{lr['recent_syn']:>+7.2f}")

    print(f"{'='*68}\n")
