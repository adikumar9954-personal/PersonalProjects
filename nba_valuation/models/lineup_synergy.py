"""
models/lineup_synergy.py
========================
Lineup synergy model — answers:
  "What is player X worth specifically when paired with player Y?"

Architecture:
  1. Lineup RAPM — same ridge regression as player RAPM but run on
     5-man lineup stints. Each lineup gets its own coefficient.

  2. Synergy decomposition — for each player, compute:
       baseline_rapm      their individual RAPM (from models/rapm.py)
       lineup_rapm        avg RAPM of lineups they appear in (weighted by poss)
       synergy_delta      lineup_rapm - sum(individual RAPMs of all 5 players)
                          positive = lineup is better than parts suggest
                          negative = lineup is worse than parts suggest

  3. Pairwise compatibility — for any two players A and B:
       shared_poss         possessions they've played together
       pair_net_rating     pts/100 when both on floor
       pair_vs_individual  how pair net rating compares to A_rapm + B_rapm
       compatibility_score -3 to +3 (pos = good fit, neg = redundant/conflicting)

  4. Best lineup builder — given a roster, find the predicted best 5
     by optimising the sum of pairwise compatibility scores.

All outputs are DataFrames you can filter, sort, and export.
"""

import numpy as np
import pandas as pd
import itertools
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge, RidgeCV
from models.rapm import inject_prior


# ═══════════════════════════════════════════════════════════════════════════
# 1. Lineup RAPM
# ═══════════════════════════════════════════════════════════════════════════

def fit_lineup_rapm(
    X_lineup: csr_matrix,
    y_lineup: np.ndarray,
    weights_lineup: np.ndarray,
    enc_lineup,
    alpha: float = None,
    min_possessions_for_report: float = 100.0,
) -> tuple[pd.DataFrame, float]:
    """
    Fit ridge regression on lineup-level stints.
    Each column = one unique 5-man lineup (encoded as a tuple of player IDs).

    Returns lineup_rapm_df with columns:
      lineup_id, player_ids (tuple), rapm, possessions
    """
    if alpha is None:
        print("[lineup_rapm] RidgeCV selecting alpha ...")
        cv = RidgeCV(alphas=np.logspace(2, 7, 30), fit_intercept=False)
        cv.fit(X_lineup, y_lineup, sample_weight=weights_lineup)
        alpha = cv.alpha_
        print(f"[lineup_rapm] alpha={alpha:.0f}")

    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X_lineup, y_lineup, sample_weight=weights_lineup)

    lineup_ids = enc_lineup.classes_   # tuples of player IDs
    coefs      = model.coef_

    df = pd.DataFrame({
        "lineup_key": lineup_ids,
        "lineup_rapm": coefs,
    })
    print(f"[lineup_rapm] {len(df)} lineups fitted  "
          f"mean={coefs.mean():.2f}  std={coefs.std():.2f}")
    return df, alpha


# ═══════════════════════════════════════════════════════════════════════════
# 2. Synergy decomposition
# ═══════════════════════════════════════════════════════════════════════════

def compute_synergy(
    stints: pd.DataFrame,
    player_rapm: pd.DataFrame,
    min_poss: float = 50.0,
) -> pd.DataFrame:
    """
    For each lineup that has enough possessions, compute:
      expected_rapm   = sum of individual RAPMs of all 5 players
      actual_rapm     = observed net rating per 100 poss for that lineup
      synergy_delta   = actual - expected
                        > 0: these players make each other better
                        < 0: skill sets conflict or overlap badly

    player_rapm: output of models/rapm.fit_rapm()
    """
    rapm_map = dict(zip(
        player_rapm["player_id"].astype(str),
        player_rapm["rapm"]
    ))
    name_map = dict(zip(
        player_rapm["player_id"].astype(str),
        player_rapm["player_name"]
    ))

    # Group stints by exact 5-man home lineup
    stints = stints.copy()
    stints["home_key"] = stints["home_players"].apply(lambda x: tuple(sorted(x)))
    stints["away_key"] = stints["away_players"].apply(lambda x: tuple(sorted(x)))

    # Focus on home lineups (symmetric — same logic applies to away)
    home_groups = (
        stints.groupby("home_key")
        .agg(
            possessions=("possessions", "sum"),
            point_diff=("point_diff", "sum"),
        )
        .reset_index()
    )
    home_groups = home_groups[home_groups["possessions"] >= min_poss]
    home_groups["actual_rapm"] = (home_groups["point_diff"] /
                                   home_groups["possessions"]) * 100

    rows = []
    for _, row in home_groups.iterrows():
        lineup = row["home_key"]
        if len(lineup) != 5:
            continue
        pid_strs = [str(p) for p in lineup]
        individual_rapms = [rapm_map.get(p, 0.0) for p in pid_strs]
        expected = sum(individual_rapms)
        delta    = row["actual_rapm"] - expected

        # Shrink delta toward zero based on sample size
        # k=500: need ~500 poss before fully trusting the number
        k = 500.0
        poss = row["possessions"]
        shrink_factor = poss / (poss + k)
        shrunk_delta  = round(delta * shrink_factor, 2)
        confidence    = round(shrink_factor * 100, 1)   # 0-100%

        rows.append({
            "lineup_key":     lineup,
            "player_ids":     pid_strs,
            "player_names":   [name_map.get(p, p) for p in pid_strs],
            "possessions":    row["possessions"],
            "actual_rapm":    round(row["actual_rapm"], 2),
            "expected_rapm":  round(expected, 2),
            "synergy_delta":  round(delta, 2),
            "synergy_shrunk": shrunk_delta,
            "confidence":     confidence,
            "individual_rapms": [round(r, 2) for r in individual_rapms],
        })

    df = pd.DataFrame(rows).sort_values("synergy_shrunk", ascending=False)
    print(f"[synergy] {len(df)} lineups with >={min_poss} poss")
    print(f"  Best delta:  {df['synergy_delta'].max():.2f}")
    print(f"  Worst delta: {df['synergy_delta'].min():.2f}")
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Pairwise compatibility
# ═══════════════════════════════════════════════════════════════════════════

def compute_pairwise_compatibility(
    stints: pd.DataFrame,
    player_rapm: pd.DataFrame,
    min_shared_poss: float = 30.0,
) -> pd.DataFrame:
    """
    For every pair of players who've shared the floor enough, compute:
      shared_possessions
      pair_net_rating    (pts/100 when both on floor, same team)
      sum_individual     (player_A_rapm + player_B_rapm)
      compatibility      pair_net_rating - sum_individual
                         > 0: additive or superadditive — good fit
                         = 0: exactly additive — neither helps nor hurts
                         < 0: subadditive — skills conflict or redundant roles

    Note: this uses raw on/off data which is noisy for short co-stints.
    The min_shared_poss filter prevents spurious readings.
    """
    rapm_map = dict(zip(
        player_rapm["player_id"].astype(str),
        player_rapm["rapm"]
    ))
    name_map = dict(zip(
        player_rapm["player_id"].astype(str),
        player_rapm["player_name"]
    ))

    # Aggregate: for each pair of home players, sum poss + pts
    pair_stats: dict = {}

    for _, row in stints.iterrows():
        home = [str(p) for p in row["home_players"]]
        poss = row["possessions"]
        diff = row["point_diff"]

        for a, b in itertools.combinations(home, 2):
            key = tuple(sorted([a, b]))
            if key not in pair_stats:
                pair_stats[key] = {"poss": 0.0, "pts_diff": 0.0}
            pair_stats[key]["poss"]     += poss
            pair_stats[key]["pts_diff"] += diff

    rows = []
    for (a, b), stats in pair_stats.items():
        if stats["poss"] < min_shared_poss:
            continue
        pair_nr    = (stats["pts_diff"] / stats["poss"]) * 100
        rapm_a     = rapm_map.get(a, 0.0)
        rapm_b     = rapm_map.get(b, 0.0)
        compat     = pair_nr - (rapm_a + rapm_b)

        # Shrink compatibility toward zero based on sample size
        # k=300: pairs stabilise faster than lineups (2 players vs 5)
        k = 300.0
        poss = stats["poss"]
        shrink_factor    = poss / (poss + k)
        compat_shrunk    = round(compat * shrink_factor, 2)
        pair_confidence  = round(shrink_factor * 100, 1)

        rows.append({
            "player_a_id":       a,
            "player_b_id":       b,
            "player_a":          name_map.get(a, a),
            "player_b":          name_map.get(b, b),
            "shared_poss":       round(poss, 1),
            "pair_net_rating":   round(pair_nr, 2),
            "rapm_a":            round(rapm_a, 2),
            "rapm_b":            round(rapm_b, 2),
            "sum_individual":    round(rapm_a + rapm_b, 2),
            "compatibility":     round(compat, 2),
            "compat_shrunk":     compat_shrunk,
            "confidence":        pair_confidence,
        })

    df = pd.DataFrame(rows).sort_values("compat_shrunk", ascending=False)
    print(f"[pairwise] {len(df)} pairs with >={min_shared_poss} shared poss")
    return df.reset_index(drop=True)


def compatibility_for_player(
    player_name: str,
    pair_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Show how a specific player fits with everyone else they've played with.
    Useful for "who should we surround player X with?"
    """
    name_lower = player_name.lower()
    mask = (pair_df["player_a"].str.lower().str.contains(name_lower) |
            pair_df["player_b"].str.lower().str.contains(name_lower))
    df = pair_df[mask].copy()

    # Normalise so target player is always in player_a column
    swap = df["player_b"].str.lower().str.contains(name_lower)
    df.loc[swap, ["player_a_id","player_b_id","player_a","player_b","rapm_a","rapm_b"]] = (
        df.loc[swap, ["player_b_id","player_a_id","player_b","player_a","rapm_b","rapm_a"]].values
    )
    return (df[["player_a","player_b","shared_poss","rapm_a","rapm_b",
                "pair_net_rating","sum_individual","compatibility"]]
            .sort_values("compatibility", ascending=False)
            .head(top_n)
            .reset_index(drop=True))


# ═══════════════════════════════════════════════════════════════════════════
# 4. Best lineup builder
# ═══════════════════════════════════════════════════════════════════════════

def find_best_lineup(
    roster_player_ids: list[str],
    pair_df: pd.DataFrame,
    player_rapm: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Given a roster (list of player IDs), find the top-N predicted lineups
    by summing pairwise compatibility scores + individual RAPMs.

    Score = sum(rapm_i for i in lineup) + sum(compat(i,j) for all pairs i,j)

    This is a simple combinatorial search — feasible for rosters ≤ 15 players
    (C(15,5) = 3003 combinations). For larger search spaces use beam search.

    Returns DataFrame of top-N lineups with their predicted net ratings.
    """
    rapm_map = dict(zip(
        player_rapm["player_id"].astype(str),
        player_rapm["rapm"]
    ))
    name_map = dict(zip(
        player_rapm["player_id"].astype(str),
        player_rapm["player_name"]
    ))

    # Build pairwise compatibility lookup
    compat_map: dict = {}
    for _, row in pair_df.iterrows():
        key = tuple(sorted([row["player_a_id"], row["player_b_id"]]))
        compat_map[key] = row["compatibility"]

    results = []
    for combo in itertools.combinations(roster_player_ids, 5):
        combo = [str(p) for p in combo]

        # Individual RAPM sum
        indiv_sum = sum(rapm_map.get(p, 0.0) for p in combo)

        # Pairwise compatibility sum
        compat_sum = sum(
            compat_map.get(tuple(sorted([a, b])), 0.0)
            for a, b in itertools.combinations(combo, 2)
        )

        total_score = indiv_sum + compat_sum * 0.5   # compat weighted at 50%

        results.append({
            "lineup":           [name_map.get(p, p) for p in combo],
            "player_ids":       combo,
            "individual_rapm":  round(indiv_sum, 2),
            "compat_bonus":     round(compat_sum, 2),
            "predicted_nr":     round(total_score, 2),
        })

    df = (pd.DataFrame(results)
          .sort_values("predicted_nr", ascending=False)
          .head(top_n)
          .reset_index(drop=True))

    df["lineup_str"] = df["lineup"].apply(lambda x: ", ".join(x))
    return df[["lineup_str","individual_rapm","compat_bonus","predicted_nr","player_ids"]]


# ═══════════════════════════════════════════════════════════════════════════
# Pretty printer
# ═══════════════════════════════════════════════════════════════════════════

def print_synergy_report(
    player_name: str,
    player_rapm: pd.DataFrame,
    synergy_df: pd.DataFrame,
    pair_df: pd.DataFrame,
) -> None:
    """Print a full lineup synergy report for one player."""
    name_lower = player_name.lower()

    rapm_row = player_rapm[
        player_rapm["player_name"].str.lower().str.contains(name_lower)
    ]
    if rapm_row.empty:
        print(f"Player '{player_name}' not found in RAPM results.")
        return
    rapm_row = rapm_row.iloc[0]

    print(f"\n{'='*62}")
    print(f"  Lineup Synergy Report — {rapm_row['player_name'].upper()}")
    print(f"{'='*62}")
    print(f"  Individual RAPM : {rapm_row['rapm']:+.2f}")
    print(f"  Prior           : {rapm_row['prior']:+.2f}")
    print(f"  Minutes         : {rapm_row['minutes']:.0f}")
    print()

    # Best lineups this player appeared in
    mask = synergy_df["player_names"].apply(
        lambda names: any(name_lower in n.lower() for n in names)
    )
    player_lineups = synergy_df[mask].head(5)
    if not player_lineups.empty:
        print("  Best lineups by synergy delta:")
        for _, lr in player_lineups.iterrows():
            names = ", ".join(lr["player_names"])
            print(f"    [{lr['synergy_delta']:+.1f}] {lr['possessions']:.0f} poss  {names}")
    print()

    # Best individual pairings
    compat = compatibility_for_player(player_name, pair_df, top_n=8)
    if not compat.empty:
        print("  Best individual pairings (compatibility score):")
        for _, cr in compat.iterrows():
            print(f"    [{cr['compatibility']:+.1f}]  {cr['player_b']:<24} "
                  f"{cr['shared_poss']:.0f} poss")
    print()

    print("  Interpretation:")
    avg_compat = compat["compatibility"].mean() if not compat.empty else 0
    if avg_compat > 1.5:
        print("    Strongly superadditive — consistently makes lineups better than")
        print("    the sum of individual parts. High fit flexibility.")
    elif avg_compat > 0.3:
        print("    Moderately positive fit — tends to be additive or slightly")
        print("    superadditive. Solid roster piece in most contexts.")
    elif avg_compat > -0.5:
        print("    Roughly additive — fits depend heavily on specific co-players.")
        print("    Look at individual pairings above for guidance.")
    else:
        print("    Below-additive on average — may have skill overlap or usage")
        print("    conflicts in common lineup configurations.")
    print(f"{'='*62}\n")
