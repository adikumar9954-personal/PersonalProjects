"""
models/rapm.py  (v2)
====================
Prior model now trains against RAPTOR/LEBRON instead of raw +/-.
Everything else structurally the same — just a better target means
a better prior, which means better small-sample estimates.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler


PRIOR_FEATURES = [
    # from box_per100
    "PTS", "STL", "BLK", "TOV",
    # from box_advanced
    "TS_PCT", "USG_PCT", "AST_PCT", "OREB_PCT", "DREB_PCT",
    "DEF_RATING", "OFF_RATING", "E_TOV_PCT",
    # from passing tracking (merged into box_adv in report.py before calling build_prior_model)
    "POTENTIAL_AST", "AST_ADJ",
]


def build_prior_model(
    box_per100: pd.DataFrame,
    box_advanced: pd.DataFrame,
    prior_targets: pd.DataFrame,
    min_minutes: float = 100.0,
) -> tuple:
    """
    Train ridge regression on box score features → prior RAPM estimate.
    Uses actual column names confirmed from nba_api 1.11.4.
    """
    # Merge per100 + advanced on PLAYER_ID
    p100_cols = ["PLAYER_ID", "PLAYER_NAME", "MIN", "PTS", "STL", "BLK", "TOV"]
    p100_cols = [c for c in p100_cols if c in box_per100.columns]
    df = box_per100[p100_cols].copy()

    adv_cols = ["PLAYER_ID", "TS_PCT", "USG_PCT", "AST_PCT", "OREB_PCT",
                "DREB_PCT", "DEF_RATING", "OFF_RATING", "E_TOV_PCT"]
    adv_cols = [c for c in adv_cols if c in box_advanced.columns]
    df = df.merge(box_advanced[adv_cols], on="PLAYER_ID", how="left")

    df = df.merge(
        prior_targets[["PLAYER_ID", "prior_target", "prior_source"]],
        on="PLAYER_ID", how="left"
    )
    df["prior_target"] = df["prior_target"].fillna(0.0)
    # Scale raw per-100 +/- to reasonable RAPM range (~-5 to +5)
    mask = df["prior_source"] == "raw_plus_minus"
    df.loc[mask, "prior_target"] = df.loc[mask, "prior_target"] / 10.0

    # Use lower min_minutes threshold — per100 MIN is minutes per game not total
    df = df[df["MIN"] >= min_minutes].reset_index(drop=True)
    print(f"[prior] {len(df)} players >= {min_minutes} MIN")
    if len(df) == 0:
        # Fallback: use all players
        df = box_per100[p100_cols].copy()
        df = df.merge(box_advanced[adv_cols], on="PLAYER_ID", how="left")
        df = df.merge(prior_targets[["PLAYER_ID","prior_target","prior_source"]],
                      on="PLAYER_ID", how="left")
        df["prior_target"] = df["prior_target"].fillna(0.0)
    # Scale raw per-100 +/- to reasonable RAPM range (~-5 to +5)
    mask = df["prior_source"] == "raw_plus_minus"
    df.loc[mask, "prior_target"] = df.loc[mask, "prior_target"] / 10.0
    print(f"[prior] fallback: using all {len(df)} players")

    print(f"[prior] target mix: {df['prior_source'].value_counts().to_dict()}")

    feature_cols = [c for c in PRIOR_FEATURES if c in df.columns]
    print(f"[prior] features found: {feature_cols}")
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    X = df[feature_cols].values
    y = df["prior_target"].values

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    model = RidgeCV(alphas=np.logspace(-1, 4, 50), cv=5)
    model.fit(X_sc, y)

    print(f"[prior] alpha={model.alpha_:.2f}  R²={model.score(X_sc, y):.3f}")
    return model, scaler, feature_cols


def predict_prior(model, scaler, feature_cols, box_per100, box_advanced) -> pd.DataFrame:
    """Generate prior estimates for ALL players."""
    df = box_per100[["PLAYER_ID","PLAYER_NAME","MIN"]].copy()

    # Merge advanced features
    adv_avail = ["PLAYER_ID"] + [c for c in feature_cols if c in box_advanced.columns]
    df = df.merge(box_advanced[adv_avail], on="PLAYER_ID", how="left")

    # Merge per100 features not already in df
    p100_avail = ["PLAYER_ID"] + [c for c in feature_cols
                                   if c in box_per100.columns and c not in df.columns]
    if len(p100_avail) > 1:
        df = df.merge(box_per100[p100_avail], on="PLAYER_ID", how="left")

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    df[feature_cols] = df[feature_cols].fillna(0.0)

    X_sc = scaler.transform(df[feature_cols].values)
    df["prior"] = model.predict(X_sc)
    return df[["PLAYER_ID","PLAYER_NAME","MIN","prior"]]


def inject_prior(X, y, weights, enc, prior_df, prior_weight=1000.0):
    """
    Add synthetic virtual stints pulling each player toward their prior.
    prior_weight = number of possession-equivalents the prior is worth.
    1000 ≈ one full season of average playing time.
    """
    from scipy.sparse import eye as sp_eye
    n_players = X.shape[1]
    virtual_X = sp_eye(n_players, format="csr", dtype=np.float32)

    pid_to_prior = dict(zip(
        prior_df["PLAYER_ID"].astype(str),
        prior_df["prior"]
    ))
    player_ids = enc.classes_.astype(str)
    virtual_y = np.array([pid_to_prior.get(p, 0.0) for p in player_ids], dtype=np.float32)
    virtual_w = np.full(n_players, prior_weight, dtype=np.float32)

    X_aug = vstack([X, virtual_X])
    y_aug = np.concatenate([y, virtual_y])
    w_aug = np.concatenate([weights, virtual_w])
    return X_aug, y_aug, w_aug


def fit_rapm(
    X: csr_matrix,
    y: np.ndarray,
    weights: np.ndarray,
    enc,
    prior_df: pd.DataFrame,
    prior_weight: float = 1000.0,
    alpha: float = None,
) -> tuple[pd.DataFrame, float]:
    """
    Fit RAPM with prior injection.
    Returns (results_df, chosen_alpha).
    """
    X_aug, y_aug, w_aug = inject_prior(X, y, weights, enc, prior_df, prior_weight)

    if alpha is None:
        print("[rapm] RidgeCV selecting alpha ...")
        cv = RidgeCV(alphas=np.logspace(1, 6, 40), fit_intercept=False)
        cv.fit(X_aug, y_aug, sample_weight=w_aug)
        alpha = cv.alpha_
        print(f"[rapm] alpha={alpha:.1f}")

    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X_aug, y_aug, sample_weight=w_aug)

    player_ids = enc.classes_.astype(str)
    pid_map  = dict(zip(prior_df["PLAYER_ID"].astype(str), prior_df["prior"]))
    name_map = dict(zip(prior_df["PLAYER_ID"].astype(str), prior_df["PLAYER_NAME"]))
    min_map  = dict(zip(prior_df["PLAYER_ID"].astype(str), prior_df["MIN"]))

    df = pd.DataFrame({
        "player_id":        player_ids,
        "player_name":      [name_map.get(p, p) for p in player_ids],
        "minutes":          [min_map.get(p, 0.0) for p in player_ids],
        "prior":            [pid_map.get(p, 0.0) for p in player_ids],
        "rapm":             model.coef_,
    })
    df["rapm_adjustment"] = df["rapm"] - df["prior"]

    print(f"[rapm] mean={df['rapm'].mean():.3f}  std={df['rapm'].std():.3f}")
    return df.sort_values("rapm", ascending=False).reset_index(drop=True), alpha


def bootstrap_se(X, y, weights, enc, prior_df, alpha, prior_weight=1000.0, n=50):
    """Bootstrap standard errors. n=50 is fast; n=200 for publication quality."""
    n_stints = X.shape[0]
    coefs = []
    for b in range(n):
        idx = np.random.choice(n_stints, n_stints, replace=True)
        X_b, y_b, w_b = X[idx], y[idx], weights[idx]
        X_aug, y_aug, w_aug = inject_prior(X_b, y_b, w_b, enc, prior_df, prior_weight)
        m = Ridge(alpha=alpha, fit_intercept=False)
        m.fit(X_aug, y_aug, sample_weight=w_aug)
        coefs.append(m.coef_)
        if (b+1) % 10 == 0:
            print(f"  bootstrap {b+1}/{n}")
    se = np.array(coefs).std(axis=0)
    return pd.DataFrame({"player_id": enc.classes_.astype(str), "rapm_se": se})
