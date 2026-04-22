"""
data/stint_matrix.py  (v2)
==========================
Builds the sparse design matrix from pbpstats stints.
Possession-weighted — no more minutes approximation.
"""

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix, vstack
from sklearn.preprocessing import LabelEncoder


def build_stint_matrix(
    stints: pd.DataFrame,
    min_possessions: float = 2.0,
) -> tuple[csr_matrix, np.ndarray, np.ndarray, LabelEncoder]:
    """
    Convert stints DataFrame into RAPM design matrix.

    stints columns required:
      home_players (list[int]), away_players (list[int]),
      point_diff (int), possessions (float)

    Returns:
      X       sparse (n_stints, n_players)  +1 / -1 / 0
      y       (n_stints,)  point diff per 100 possessions
      weights (n_stints,)  possession counts
      enc     LabelEncoder  player_id -> column index
    """
    # Filter noise stints
    stints = stints[stints["possessions"] >= min_possessions].reset_index(drop=True)
    print(f"[matrix] {len(stints)} stints after filtering (min {min_possessions} poss)")

    # Collect all player IDs
    all_players: set = set()
    for _, row in stints.iterrows():
        all_players.update(row["home_players"])
        all_players.update(row["away_players"])

    enc = LabelEncoder()
    enc.fit(sorted(all_players))
    n_players = len(enc.classes_)
    n_stints  = len(stints)
    print(f"[matrix] {n_players} unique players across {n_stints} stints")

    # Build sparse X
    X = lil_matrix((n_stints, n_players), dtype=np.float32)
    classes_set = set(enc.classes_)

    for i, (_, row) in enumerate(stints.iterrows()):
        home_ids = enc.transform([p for p in row["home_players"] if p in classes_set])
        away_ids = enc.transform([p for p in row["away_players"] if p in classes_set])
        for j in home_ids: X[i, j] =  1.0
        for j in away_ids: X[i, j] = -1.0

    X = X.tocsr()

    # Target: pts/100poss (possession-weighted)
    poss = stints["possessions"].values.astype(np.float32)
    y    = (stints["point_diff"].values.astype(np.float32) / poss) * 100.0
    y    = np.clip(y, -150, 150)   # clip extreme short stints

    print(f"[matrix] y: mean={y.mean():.2f}  std={y.std():.2f}")
    print(f"[matrix] X density: {X.nnz / (n_stints * n_players):.5f}")

    return X.astype(np.float32), y, poss, enc


def build_lineup_matrix(
    stints: pd.DataFrame,
    min_possessions: float = 10.0,
) -> tuple[csr_matrix, np.ndarray, np.ndarray, LabelEncoder]:
    """
    Build a lineup-level design matrix — one row per unique 5-man lineup pair
    instead of one row per stint.

    This is the input for the lineup synergy model. We group stints by
    the exact 10-man combination and aggregate possessions + point diff.
    """
    stints = stints[stints["possessions"] >= 1].copy()

    # Create hashable lineup keys
    stints["home_key"] = stints["home_players"].apply(lambda x: tuple(sorted(x)))
    stints["away_key"] = stints["away_players"].apply(lambda x: tuple(sorted(x)))
    stints["matchup_key"] = list(zip(stints["home_key"], stints["away_key"]))

    # Group by exact 10-man matchup
    grouped = (
        stints.groupby("matchup_key")
        .agg(
            home_players=("home_key", "first"),
            away_players=("away_key", "first"),
            possessions=("possessions", "sum"),
            point_diff=("point_diff",  "sum"),
        )
        .reset_index(drop=True)
    )

    grouped = grouped[grouped["possessions"] >= min_possessions]
    print(f"[lineup_matrix] {len(grouped)} unique matchups (min {min_possessions} poss)")

    return build_stint_matrix(grouped, min_possessions=min_possessions)


def temporal_split(
    stints: pd.DataFrame,
    test_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal split — use end of season as test set to prevent leakage."""
    stints = stints.sort_values("game_id").reset_index(drop=True)
    cut = int(len(stints) * (1 - test_frac))
    return stints.iloc[:cut].copy(), stints.iloc[cut:].copy()
