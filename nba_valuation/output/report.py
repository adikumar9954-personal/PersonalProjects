"""
output/report.py  (v2)
======================
Player reports, league screener, and lineup synergy outputs.
Single entry point: run_full_pipeline()
"""

import textwrap
import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Player report ──────────────────────────────────────────────────────────

def player_report(
    name_or_id: str,
    validated: pd.DataFrame,
    pair_df: pd.DataFrame = None,
    synergy_df: pd.DataFrame = None,
) -> str:
    mask = (validated["player_name"].str.lower().str.contains(name_or_id.lower()) |
            (validated["player_id"].astype(str) == str(name_or_id)))
    if not mask.any():
        return f"Player '{name_or_id}' not found."
    row = validated[mask].iloc[0]

    support = row.get("support_score")
    rapm    = row.get("rapm", 0)
    prior   = row.get("prior", 0)

    if support is None:   verdict, sym = "Insufficient tracking data", "?"
    elif support >= 70:   verdict, sym = "RAPM well-supported by tracking", "+"
    elif support >= 45:   verdict, sym = "Mixed tracking support", "~"
    else:                 verdict, sym = "RAPM not well-supported", "-"

    support_str  = f"{support:.0f}/100" if support is not None else "N/A"
    o_sup_str    = f"{row.get('o_support'):.0f}/100" if row.get('o_support') is not None else "N/A"
    d_sup_str    = f"{row.get('d_support'):.0f}/100" if row.get('d_support') is not None else "N/A"
    pm_sup_str   = f"{row.get('playmaking_support'):.0f}/100" if row.get('playmaking_support') is not None else "N/A"
    rapm_pct_str = f"{row.get('rapm_pct', 0):.0f}" if row.get('rapm_pct') is not None else "?"

    lines = [
        "=" * 62,
        f"  {row['player_name'].upper()}",
        "=" * 62,
        f"  RAPM            {rapm:+.2f}  ({rapm_pct_str}th pct)",
        f"  Prior estimate  {prior:+.2f}",
        f"  On/off adjust   {rapm-prior:+.2f}",
        f"  Minutes         {row['minutes']:.0f}",
        "",
        f"  Tracking support: {support_str} [{sym}]  {verdict}",
        f"    Offense: {o_sup_str}   Defense: {d_sup_str}   Playmaking: {pm_sup_str}",
        "",
    ]
    for f in row.get("green_flags", []):
        lines.append(f"  + {f}")
    for f in row.get("red_flags", []):
        lines.append(f"  ! {f}")

    if pair_df is not None and not pair_df.empty:
        lines.append("")
        lines.append("  Best pairings:")
        from models.lineup_synergy import compatibility_for_player
        compat = compatibility_for_player(row["player_name"], pair_df, top_n=5)
        for _, cr in compat.iterrows():
            lines.append(f"    [{cr['compatibility']:+.1f}]  {cr['player_b']}")

    lines.append("=" * 62)
    return "\n".join(str(l) for l in lines)


# ── League screener ────────────────────────────────────────────────────────

def league_screener(
    validated: pd.DataFrame,
    min_minutes: float = 20.0,
    top_n: int = 15,
) -> dict[str, pd.DataFrame]:
    df = validated[validated["minutes"] >= min_minutes].copy()
    cols = ["player_name","minutes","rapm","rapm_pct","prior",
            "rapm_adjustment","support_score","o_support","d_support","playmaking_support"]
    cols = [c for c in cols if c in df.columns]

    return {
        "overvalued":      df[df["mispriced_up"]].sort_values("support_score").head(top_n)[cols],
        "undervalued":     df[df["mispriced_down"]].sort_values("support_score", ascending=False).head(top_n)[cols],
        "confirmed_elite": df[(df["rapm_pct"]>70)&(df["support_score"]>65)].sort_values("rapm",ascending=False).head(top_n)[cols],
        "hidden_gems":     df[(df["rapm_pct"]<50)&(df["support_score"]>70)].sort_values("support_score",ascending=False).head(top_n)[cols],
    }


def print_screener(screens: dict) -> None:
    labels = {
        "overvalued":      "OVERVALUED — high RAPM, low tracking support",
        "undervalued":     "UNDERVALUED — low RAPM, high tracking support",
        "confirmed_elite": "CONFIRMED ELITE — high RAPM + high support",
        "hidden_gems":     "HIDDEN GEMS — modest RAPM, tracking says more",
    }
    for key, df in screens.items():
        print(f"\n{'='*62}\n  {labels[key]}\n{'='*62}")
        if df.empty:
            print("  (none)")
            continue
        fmt = df.copy()
        for c in ["rapm","prior","rapm_adjustment"]:
            if c in fmt.columns: fmt[c] = fmt[c].apply(lambda v: f"{v:+.2f}")
        for c in ["rapm_pct","support_score","o_support","d_support"]:
            if c in fmt.columns: fmt[c] = fmt[c].apply(lambda v: f"{v:.0f}")
        fmt["minutes"] = fmt["minutes"].apply(lambda v: f"{v:.0f}")
        print(fmt.to_string(index=False))


def export_results(validated: pd.DataFrame, pair_df: pd.DataFrame = None) -> None:
    out = validated.copy()
    for col in ["red_flags","green_flags"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: " | ".join(x) if isinstance(x, list) else x)
    p = OUTPUT_DIR / "player_valuations.csv"
    out.to_csv(p, index=False)
    print(f"[export] {len(out)} players → {p}")

    if pair_df is not None and not pair_df.empty:
        p2 = OUTPUT_DIR / "pairwise_compatibility.csv"
        pair_df.to_csv(p2, index=False)
        print(f"[export] {len(pair_df)} pairs → {p2}")


# ═══════════════════════════════════════════════════════════════════════════
# Full pipeline
# ═══════════════════════════════════════════════════════════════════════════

def _prior_seasons(season: str, n: int) -> list[str]:
    """Return n seasons ending at (and including) season, newest first.
    e.g. _prior_seasons('2025-26', 3) → ['2025-26', '2024-25', '2023-24']
    """
    y = int(season.split("-")[0])
    return [f"{y - i}-{str(y - i + 1)[-2:]}" for i in range(n)]


def run_full_pipeline(
    season: str = "2023-24",
    n_seasons: int = 3,
    season_weights: list = None,
    max_games: int = None,
    prior_weight: float = 1000.0,
    synergy_min_poss: float = 50.0,
    pair_min_poss: float = 30.0,
    roster_for_lineup: list[str] = None,
) -> dict:
    """
    End-to-end pipeline. Returns dict with all output DataFrames.

    n_seasons:       how many seasons of stints to stack for RAPM (default 3 — the
                     accepted stabilisation window). Box scores / tracking / priors
                     always use the current season only.
    season_weights:  possession multipliers per season, newest-first.
                     Defaults to [1.00, 0.75, 0.50] for 3 seasons.
    roster_for_lineup: list of player_ids — if provided, prints optimal lineup.
    """
    from data.ingest import (get_stints, get_box_scores, get_advanced_box,
                              get_all_tracking, get_best_prior_target,
                              get_passing_stats, get_lineup_shot_profile)
    from data.playmaking import compute_playmaking_onoff
    from data.stint_matrix import build_stint_matrix
    from models.rapm import build_prior_model, predict_prior, fit_rapm
    from models.validator import validate_all_players
    from models.lineup_synergy import (compute_synergy, compute_pairwise_compatibility,
                                        find_best_lineup, print_synergy_report)

    # Resolve seasons + weights
    if season_weights is None:
        season_weights = [max(0.25, 1.0 - 0.25 * i) for i in range(n_seasons)]
    seasons_list   = _prior_seasons(season, n_seasons)
    season_weights = list(season_weights)[:len(seasons_list)]

    print(f"\n{'='*62}")
    print(f"  NBA Valuation Pipeline v2 — {season}")
    wstr = "  +  ".join(f"{s} (×{w:.2f})" for s, w in zip(seasons_list, season_weights))
    print(f"  Seasons: {wstr}")
    print(f"{'='*62}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    print("── 1. Data ──")
    box100         = get_box_scores(season)
    box_adv        = get_advanced_box(season)
    tracking       = get_all_tracking(season)
    lineup_profile = get_lineup_shot_profile(season)

    # Multi-year stints: fetch each season, scale possessions, then stack
    all_stints = []
    for s, w in zip(seasons_list, season_weights):
        s_stints = get_stints(s, max_games=max_games)
        if not s_stints.empty:
            scaled = s_stints.copy()
            scaled["possessions"] = scaled["possessions"] * w
            all_stints.append(scaled)
            print(f"  [{s}] {len(scaled)} stints  weight={w:.2f}")
    stints = pd.concat(all_stints, ignore_index=True) if all_stints else pd.DataFrame()
    if len(seasons_list) > 1:
        print(f"  [combined] {len(stints)} total stints across {len(seasons_list)} seasons")

    # ── Playmaking on/off (team-wide effects) ─────────────────────────────
    print("\n── 1b. Playmaking on/off ──")
    playmaking_df = compute_playmaking_onoff(lineup_profile)

    # ── Prior ─────────────────────────────────────────────────────────────
    # Merge passing stats into box_adv so POTENTIAL_AST / AST_ADJ are available
    # as prior model features without changing the model signature.
    print("\n── 2. Prior model ──")
    passing_stats = get_passing_stats(season)
    if not passing_stats.empty and "PLAYER_ID" in passing_stats.columns:
        pass_cols = ["PLAYER_ID"] + [c for c in ["POTENTIAL_AST", "AST_ADJ"]
                                      if c in passing_stats.columns]
        if len(pass_cols) > 1:
            box_adv = box_adv.merge(passing_stats[pass_cols], on="PLAYER_ID", how="left")
            print(f"  [prior] merged passing features: {pass_cols[1:]}")
    prior_targets = get_best_prior_target(season, box100)
    pm, scaler, feature_cols = build_prior_model(box100, box_adv, prior_targets)
    prior_df = predict_prior(pm, scaler, feature_cols, box100, box_adv)

    # ── Stint matrix + RAPM ───────────────────────────────────────────────
    print("\n── 3. Stint matrix ──")
    X, y, weights, enc = build_stint_matrix(stints)

    print("\n── 4. RAPM ──")
    rapm_results, alpha = fit_rapm(X, y, weights, enc, prior_df, prior_weight)

    # ── Validation ────────────────────────────────────────────────────────
    print("\n── 5. Tracking validation ──")
    validated = validate_all_players(
        rapm_results, tracking,
        playmaking_df=playmaking_df,
        min_games=5, min_minutes=10.0,
    )

    # ── Lineup synergy ────────────────────────────────────────────────────
    print("\n── 6. Lineup synergy ──")
    synergy_df = compute_synergy(stints, rapm_results, min_poss=synergy_min_poss)
    pair_df    = compute_pairwise_compatibility(stints, rapm_results, min_shared_poss=pair_min_poss)

    # ── Outputs ───────────────────────────────────────────────────────────
    print("\n── 7. Outputs ──")
    screens = league_screener(validated)
    print_screener(screens)
    export_results(validated, pair_df)

    if roster_for_lineup:
        print("\n── Optimal lineup from provided roster ──")
        best = find_best_lineup(roster_for_lineup, pair_df, rapm_results)
        print(best[["lineup_str","individual_rapm","compat_bonus","predicted_nr"]]
              .to_string(index=False))

    return {
        "rapm":     rapm_results,
        "validated": validated,
        "synergy":  synergy_df,
        "pairs":    pair_df,
        "screens":  screens,
        "alpha":    alpha,
    }


# ── entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    season   = sys.argv[1] if len(sys.argv) > 1 else "2023-24"
    max_games = int(sys.argv[2]) if len(sys.argv) > 2 else None

    out = run_full_pipeline(season=season, max_games=max_games)

    # Example reports
    from models.lineup_synergy import print_synergy_report
    # Print top 10 players by RAPM
    print("\n=== TOP 10 BY RAPM ===")
    top10 = out["rapm"].head(10)[["player_name","minutes","prior","rapm","rapm_adjustment"]]
    print(top10.to_string(index=False))

    print("\n=== BOTTOM 10 BY RAPM ===")
    bot10 = out["rapm"].tail(10)[["player_name","minutes","prior","rapm","rapm_adjustment"]]
    print(bot10.to_string(index=False))

    print("\n=== TOP 10 PAIRWISE COMPATIBILITY ===")
    print(out["pairs"].head(10)[["player_a","player_b","shared_poss",
                                  "compatibility","pair_net_rating"]].to_string(index=False))

    # Try individual player reports for whoever is in the data
    sample_players = out["rapm"]["player_name"].head(5).tolist()
    for name in sample_players:
        try:
            print(player_report(name, out["validated"], out["pairs"], out["synergy"]))
        except Exception as e:
            print(f"  [skip] {name}: {e}")
