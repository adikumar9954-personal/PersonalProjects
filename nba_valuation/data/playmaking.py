"""
data/playmaking.py
==================
Compute per-player team on/off effects on six lineup-level metrics, all using
true team on/off (same team's lineups without the player as the off-court
baseline, not league average):

  Offensive
  ---------
  fg3a_rate_delta   - team 3PA rate on vs off court
  tov_rate_delta    - team TOV rate on vs off court  (negative = fewer TOs)
  rim_rate_delta    - team PTS_PAINT/FGA on vs off court

  Defensive
  ---------
  opp_fg_pct_delta     - team opponent FG% on vs off court  (negative = better)
  opp_fg3a_rate_delta  - team opponent 3PA rate on vs off  (negative = better)
  opp_tov_rate_delta   - team forced TOV rate on vs off     (positive = better)

League-average baselines are computed but only used as a fallback when a
team has no off-court sample for a given player.
"""

import numpy as np
import pandas as pd


def _isnan(v) -> bool:
    """Return True if v is NaN (handles both Python float and numpy scalar)."""
    try:
        return np.isnan(v)
    except (TypeError, ValueError):
        return False


def compute_playmaking_onoff(lineup_shot_profile: pd.DataFrame) -> pd.DataFrame:
    """
    From lineup-level shot profile data, compute per-player weighted team on/off
    effects on six metrics (three offensive, three defensive).

    Returns DataFrame with one row per player and columns:
        PLAYER_ID, fg3a_rate, tov_rate,
        fg3a_rate_delta, tov_rate_delta, rim_rate_delta,
        opp_fg_pct_delta, opp_fg3a_rate_delta, opp_tov_rate_delta
    """
    if lineup_shot_profile.empty:
        print("[playmaking] no lineup shot profile data - skipping")
        return pd.DataFrame()

    df = lineup_shot_profile.copy()

    # ── Lineup-level offensive rates ─────────────────────────────────────────
    df["fg3a_rate"] = df["FG3A"] / df["FGA"].replace(0, np.nan)

    tov_denom = (df["FGA"]
                 + 0.44 * df.get("FTA", pd.Series(0, index=df.index))
                 + df["TOV"])
    df["tov_rate"] = df["TOV"] / tov_denom.replace(0, np.nan)

    has_rim = "PTS_PAINT" in df.columns and "FGA" in df.columns
    df["rim_rate"] = (
        df["PTS_PAINT"] / df["FGA"].replace(0, np.nan)
        if has_rim else np.nan
    )
    if not has_rim:
        print("[playmaking] PTS_PAINT not available - rim_rate_delta will be NaN")

    # ── Lineup-level defensive rates ─────────────────────────────────────────
    has_def = all(c in df.columns for c in ["OPP_FGM", "OPP_FGA"])
    if has_def:
        df["opp_fg_pct"]    = df["OPP_FGM"] / df["OPP_FGA"].replace(0, np.nan)
        df["opp_fg3a_rate"] = df.get("OPP_FG3A", 0) / df["OPP_FGA"].replace(0, np.nan)
        opp_tov_denom = (
            df["OPP_FGA"]
            + 0.44 * df.get("OPP_FTA", pd.Series(0, index=df.index))
            + df.get("OPP_TOV", pd.Series(0, index=df.index))
        )
        df["opp_tov_rate"] = (
            df.get("OPP_TOV", pd.Series(0, index=df.index))
            / opp_tov_denom.replace(0, np.nan)
        )
    else:
        df["opp_fg_pct"] = df["opp_fg3a_rate"] = df["opp_tov_rate"] = np.nan
        print("[playmaking] OPP_FGM/FGA not available - defensive on/off will be NaN")

    # ── League-average baselines (fallback only) ─────────────────────────────
    total_fga  = df["FGA"].sum()
    total_fg3a = df["FG3A"].sum()
    total_tov  = df["TOV"].sum()
    total_fta  = df.get("FTA", pd.Series(0, index=df.index)).sum()
    lg_fg3a_rate = total_fg3a / max(total_fga, 1)
    lg_tov_rate  = total_tov  / max(total_fga + 0.44 * total_fta + total_tov, 1)
    lg_rim_rate  = (df["PTS_PAINT"].sum() / max(total_fga, 1)) if has_rim else np.nan

    if has_def:
        total_opp_fga  = df["OPP_FGA"].sum()
        total_opp_fgm  = df["OPP_FGM"].sum()
        total_opp_fg3a = df.get("OPP_FG3A", pd.Series(0, index=df.index)).sum()
        total_opp_fta  = df.get("OPP_FTA", pd.Series(0, index=df.index)).sum()
        total_opp_tov  = df.get("OPP_TOV", pd.Series(0, index=df.index)).sum()
        lg_opp_fg_pct    = total_opp_fgm  / max(total_opp_fga, 1)
        lg_opp_fg3a_rate = total_opp_fg3a / max(total_opp_fga, 1)
        lg_opp_tov_rate  = total_opp_tov  / max(
            total_opp_fga + 0.44 * total_opp_fta + total_opp_tov, 1)
    else:
        lg_opp_fg_pct = lg_opp_fg3a_rate = lg_opp_tov_rate = np.nan

    print(
        f"[playmaking] league avg - 3PA rate: {lg_fg3a_rate:.3f}  "
        f"TOV rate: {lg_tov_rate:.3f}"
        + (f"  rim rate: {lg_rim_rate:.3f}" if has_rim else "")
        + (f"  opp FG%: {lg_opp_fg_pct:.3f}" if has_def else "")
    )

    df["weight"] = df["MIN"].fillna(0)
    has_team = "team_id" in df.columns and df["team_id"].notna().any()

    # ── Single pass: build per-player accumulators and team-lineup index ─────
    #
    # player_data[pid]  - on-court weighted sums for display values + fallback
    # team_lineups[tid] - list of tuples:
    #   (set(pids), fg3a_r, tov_r, rim_r, opp_fg_r, opp_f3r, opp_tvr, w)
    # player_team_w[pid]- {team_id: total on-court minutes} -> primary team

    player_data:   dict = {}
    team_lineups:  dict = {}
    player_team_w: dict = {}

    def _f(v):
        """Return float or NaN, safe for any scalar type."""
        return float(v) if not _isnan(v) else np.nan

    for _, row in df.iterrows():
        pids = row["player_ids"]
        if hasattr(pids, "tolist"):
            pids = pids.tolist()
        if not isinstance(pids, list) or len(pids) != 5:
            continue
        w = float(row["weight"])
        if w <= 0:
            continue

        fg3a_r  = _f(row["fg3a_rate"])
        tov_r   = _f(row["tov_rate"])
        rim_r   = _f(row["rim_rate"])
        opp_fg  = _f(row["opp_fg_pct"])
        opp_f3  = _f(row["opp_fg3a_rate"])
        opp_tv  = _f(row["opp_tov_rate"])

        if has_team:
            tid = row["team_id"]
            if tid not in team_lineups:
                team_lineups[tid] = []
            team_lineups[tid].append(
                (set(pids), fg3a_r, tov_r, rim_r, opp_fg, opp_f3, opp_tv, w)
            )

        for pid in pids:
            if pid not in player_data:
                player_data[pid] = {k: 0.0 for k in [
                    "fg3a_w", "fg3a_v", "tov_w", "tov_v",
                    "rim_w",  "rim_v",
                    "opp_fg_w", "opp_fg_v", "opp_f3_w", "opp_f3_v",
                    "opp_tv_w", "opp_tv_v",
                ]}
            d = player_data[pid]
            if not _isnan(fg3a_r): d["fg3a_w"] += fg3a_r * w; d["fg3a_v"] += w
            if not _isnan(tov_r):  d["tov_w"]  += tov_r  * w; d["tov_v"]  += w
            if not _isnan(rim_r):  d["rim_w"]  += rim_r  * w; d["rim_v"]  += w
            if not _isnan(opp_fg): d["opp_fg_w"] += opp_fg * w; d["opp_fg_v"] += w
            if not _isnan(opp_f3): d["opp_f3_w"] += opp_f3 * w; d["opp_f3_v"] += w
            if not _isnan(opp_tv): d["opp_tv_w"] += opp_tv * w; d["opp_tv_v"] += w
            if has_team:
                if pid not in player_team_w:
                    player_team_w[pid] = {}
                player_team_w[pid][tid] = player_team_w[pid].get(tid, 0) + w

    # Primary team = team with the most on-court minutes
    player_primary_team = {
        pid: max(tw, key=tw.get) for pid, tw in player_team_w.items()
    } if has_team else {}

    # ── Per-player on/off deltas ──────────────────────────────────────────────
    rows = []
    for pid, d in player_data.items():
        def _avg(w_key, v_key):
            return d[w_key] / d[v_key] if d[v_key] > 0 else np.nan

        fg3a_rate = _avg("fg3a_w", "fg3a_v")
        tov_rate  = _avg("tov_w",  "tov_v")

        fg3a_delta = tov_delta = rim_delta = np.nan
        opp_fg_delta = opp_f3_delta = opp_tv_delta = np.nan

        rid = player_primary_team.get(pid)
        if rid is not None:
            on_f3_w = on_f3_v = off_f3_w = off_f3_v = 0.0
            on_tv_w = on_tv_v = off_tv_w = off_tv_v = 0.0
            on_rm_w = on_rm_v = off_rm_w = off_rm_v = 0.0
            on_og_w = on_og_v = off_og_w = off_og_v = 0.0
            on_o3_w = on_o3_v = off_o3_w = off_o3_v = 0.0
            on_ot_w = on_ot_v = off_ot_w = off_ot_v = 0.0

            for lp, f3r, tvr, rmr, ofg, of3, otv, w in team_lineups.get(rid, []):
                is_on = pid in lp
                if not _isnan(f3r):
                    if is_on: on_f3_w  += f3r * w; on_f3_v  += w
                    else:     off_f3_w += f3r * w; off_f3_v += w
                if not _isnan(tvr):
                    if is_on: on_tv_w  += tvr * w; on_tv_v  += w
                    else:     off_tv_w += tvr * w; off_tv_v += w
                if not _isnan(rmr):
                    if is_on: on_rm_w  += rmr * w; on_rm_v  += w
                    else:     off_rm_w += rmr * w; off_rm_v += w
                if not _isnan(ofg):
                    if is_on: on_og_w  += ofg * w; on_og_v  += w
                    else:     off_og_w += ofg * w; off_og_v += w
                if not _isnan(of3):
                    if is_on: on_o3_w  += of3 * w; on_o3_v  += w
                    else:     off_o3_w += of3 * w; off_o3_v += w
                if not _isnan(otv):
                    if is_on: on_ot_w  += otv * w; on_ot_v  += w
                    else:     off_ot_w += otv * w; off_ot_v += w

            def _delta(on_w, on_v, off_w, off_v):
                if on_v > 0 and off_v > 0:
                    return round(on_w / on_v - off_w / off_v, 4)
                return np.nan

            fg3a_delta   = _delta(on_f3_w, on_f3_v, off_f3_w, off_f3_v)
            tov_delta    = _delta(on_tv_w, on_tv_v, off_tv_w, off_tv_v)
            rim_delta    = _delta(on_rm_w, on_rm_v, off_rm_w, off_rm_v)
            opp_fg_delta = _delta(on_og_w, on_og_v, off_og_w, off_og_v)
            opp_f3_delta = _delta(on_o3_w, on_o3_v, off_o3_w, off_o3_v)
            opp_tv_delta = _delta(on_ot_w, on_ot_v, off_ot_w, off_ot_v)

        # Fallbacks - league-average baseline when team on/off is unavailable
        if _isnan(fg3a_delta) and not _isnan(fg3a_rate):
            fg3a_delta = round(fg3a_rate - lg_fg3a_rate, 4)
        if _isnan(tov_delta) and not _isnan(tov_rate):
            tov_delta  = round(tov_rate  - lg_tov_rate,  4)
        if _isnan(rim_delta) and d["rim_v"] > 0:
            rim_delta  = round(_avg("rim_w", "rim_v") - lg_rim_rate, 4)
        if _isnan(opp_fg_delta) and d["opp_fg_v"] > 0:
            opp_fg_delta = round(_avg("opp_fg_w", "opp_fg_v") - lg_opp_fg_pct, 4)
        if _isnan(opp_f3_delta) and d["opp_f3_v"] > 0:
            opp_f3_delta = round(_avg("opp_f3_w", "opp_f3_v") - lg_opp_fg3a_rate, 4)
        if _isnan(opp_tv_delta) and d["opp_tv_v"] > 0:
            opp_tv_delta = round(_avg("opp_tv_w", "opp_tv_v") - lg_opp_tov_rate, 4)

        rows.append({
            "PLAYER_ID":          str(pid),
            "fg3a_rate":          round(fg3a_rate, 4) if not _isnan(fg3a_rate) else np.nan,
            "tov_rate":           round(tov_rate,  4) if not _isnan(tov_rate)  else np.nan,
            "fg3a_rate_delta":    fg3a_delta,
            "tov_rate_delta":     tov_delta,
            "rim_rate_delta":     rim_delta,
            "opp_fg_pct_delta":   opp_fg_delta,
            "opp_fg3a_rate_delta":opp_f3_delta,
            "opp_tov_rate_delta": opp_tv_delta,
        })

    result = pd.DataFrame(rows)
    print(f"[playmaking] on/off computed for {len(result)} players")
    return result
