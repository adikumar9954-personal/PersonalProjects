"""
data/ingest.py  (v2)
====================
Fixes from v1:
  1. Stints via pbpstats.com API - no fragile PBP parser
  2. Prior target: RAPTOR > LEBRON > raw +/- (in priority order)
  3. Full caching - every endpoint hits the network once per season

Install: pip install nba_api pandas pyarrow requests
"""

import time
import requests
import pandas as pd
from pathlib import Path
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguedashptstats,
    leaguegamelog,
)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
NBA_API_DELAY = 0.65


# ── helpers ────────────────────────────────────────────────────────────────

def _cache_path(name: str, season: str) -> Path:
    return CACHE_DIR / f"{name}_{season.replace('-', '_')}.parquet"


def _load_or_fetch(name: str, season: str, fetch_fn) -> pd.DataFrame:
    path = _cache_path(name, season)
    if path.exists():
        print(f"  [cache] {name}")
        return pd.read_parquet(path)
    print(f"  [fetch] {name} ...")
    df = fetch_fn()
    df.to_parquet(path, index=False)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# FIX 1 - Stints via pbpstats.com
# ═══════════════════════════════════════════════════════════════════════════

def get_game_ids(season: str = "2023-24") -> list[str]:
    path = CACHE_DIR / f"game_ids_{season.replace('-','_')}.txt"
    if path.exists():
        return path.read_text().splitlines()
    log = leaguegamelog.LeagueGameLog(
        season=season, season_type_all_star="Regular Season"
    )
    df = log.get_data_frames()[0]
    ids = df["GAME_ID"].unique().tolist()
    path.write_text("\n".join(ids))
    print(f"  [fetch] {len(ids)} game IDs")
    return ids


def _fetch_pbpstats_game(game_id: str) -> list[dict]:
    """
    Pull lineup stints for one game from pbpstats.com.
    Handles double-subs, ejections, OT, technicals - all the edge cases
    the hand-rolled PBP parser missed.
    """
    url = "https://api.pbpstats.com/get-game-stats/nba"
    params = {
        "GameId":    game_id,
        "Type":      "lineup",
        "StartType": "All",
        "EndType":   "All",
    }
    try:
        r = requests.get(
            url, params=params,
            headers={"User-Agent": "nba-valuation-research/2.0",
                     "Referer": "https://pbpstats.com"},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"    [pbpstats] {game_id}: {e}")
        return []

    stints = []
    for row in data.get("multi_row_table_data", []):
        home_pids = row.get("home_player_ids", "")
        away_pids = row.get("away_player_ids", "")
        poss      = float(row.get("possessions", 0) or 0)
        if not home_pids or not away_pids or poss < 1:
            continue
        home_pts = int(row.get("home_points", 0) or 0)
        away_pts = int(row.get("away_points", 0) or 0)
        stints.append({
            "game_id":      game_id,
            "home_players": [int(p) for p in str(home_pids).split(",") if p.strip()],
            "away_players": [int(p) for p in str(away_pids).split(",") if p.strip()],
            "home_pts":     home_pts,
            "away_pts":     away_pts,
            "possessions":  poss,
            "point_diff":   home_pts - away_pts,
        })
    return stints


def get_stints(season: str = "2023-24", max_games: int = None) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguedashlineups

    suffix = f"_top{max_games}" if max_games else "_full"
    path = CACHE_DIR / f"stints_lineups_{season.replace('-','_')}{suffix}.parquet"
    if path.exists():
        print(f"  [cache] stints ({suffix})")
        return pd.read_parquet(path)

    print(f"  [fetch] lineup stints via LeagueDashLineups ...")

    df = None
    for attempt in range(3):
        try:
            ep = leaguedashlineups.LeagueDashLineups(
                season=season,
                season_type_all_star="Regular Season",
                per_mode_detailed="Totals",
                measure_type_detailed_defense="Base",
                group_quantity="5",
            )
            df = ep.get_data_frames()[0]
            break
        except Exception as e:
            wait = (attempt + 1) * 20
            print(f"  [retry {attempt+1}/3] failed: {e.__class__.__name__}: {e} - waiting {wait}s")
            time.sleep(wait)

    if df is None or df.empty:
        print("  [warning] LeagueDashLineups failed - returning empty")
        return pd.DataFrame()

    time.sleep(NBA_API_DELAY)
    print(f"  [lineups] {len(df)} lineup rows, columns: {df.columns.tolist()[:8]}")

    # Parse player IDs from GROUP_ID (format: "203999-201939-...")
    def parse_ids(group_id):
        try:
            return [int(x) for x in str(group_id).split("-") if x.strip()]
        except Exception:
            return []

    id_col = "GROUP_ID" if "GROUP_ID" in df.columns else df.columns[1]
    df["home_players"] = df[id_col].apply(parse_ids)
    df["away_players"] = df[id_col].apply(lambda x: [])
    df["game_id"]      = "season_agg"

    # Use W_PCT proxy for possessions if POSS not available
    poss_col = next((c for c in ["POSS", "FGA", "FGM"] if c in df.columns), None)
    df["possessions"] = df[poss_col].fillna(50) if poss_col else 50

    pt_col = next((c for c in ["PLUS_MINUS", "PLUS_MINUS_RANK"] if c in df.columns), None)
    df["point_diff"] = df[pt_col].fillna(0) if pt_col else 0

    min_col = next((c for c in ["MIN", "MINUTES"] if c in df.columns), None)
    df["minutes"] = df[min_col].fillna(0) if min_col else 0

    df = df[df["minutes"] >= 2].reset_index(drop=True)

    if max_games:
        df = df.head(max_games * 15)

    df.to_parquet(path, index=False)
    print(f"  [stints] {len(df)} lineups cached")
    return df


def get_raptor(season: str = "2023-24") -> pd.DataFrame | None:
    """
    Download RAPTOR from 538's public GitHub repo.
    Returns columns: player_name, raptor_total, season
    """
    year = int(season.split("-")[0]) + 1
    path = CACHE_DIR / f"raptor_{year}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        return df[df["season"] == year] if "season" in df.columns else df
    try:
        url = ("https://raw.githubusercontent.com/fivethirtyeight/data"
               "/master/nba-raptor/modern_RAPTOR_by_player.csv")
        df = pd.read_csv(url)
        df.to_parquet(path, index=False)
        result = df[df["season"] == year] if "season" in df.columns else df
        print(f"  [raptor] {len(result)} player-seasons for {year}")
        return result
    except Exception as e:
        print(f"  [raptor] unavailable: {e}")
        return None


def get_lebron(season: str = "2023-24") -> pd.DataFrame | None:
    """
    Load LEBRON from a locally saved CSV.
    Download from: https://www.basketball-reference.com/friv/bpm2.fcgi
    Save as: data/cache/lebron_{year}.csv
    Expected columns: player_id (int), player_name, lebron
    """
    year = int(season.split("-")[0]) + 1
    path = CACHE_DIR / f"lebron_{year}.csv"
    if not path.exists():
        print(f"  [lebron] Not at {path} - skipping. "
              f"Download from BBRef and save there to use LEBRON as prior.")
        return None
    df = pd.read_csv(path)
    print(f"  [lebron] {len(df)} players")
    return df


def get_darko(season: str = "2025-26") -> pd.DataFrame | None:
    """Load DARKO from data/cache/darko_{year}.csv"""
    year = int(season.split("-")[0]) + 1
    path = CACHE_DIR / f"darko_{year}.csv"
    if not path.exists():
        print(f"  [darko] Not found at {path}")
        return None
    df = pd.read_csv(path)
    df = df.rename(columns={"nba_id": "player_id", "Player": "player_name", "DPM": "darko"})
    df["player_id"] = df["player_id"].astype(str)
    print(f"  [darko] {len(df)} players loaded")
    return df


def get_best_prior_target(season: str, box_per100: pd.DataFrame) -> pd.DataFrame:
    """
    For every player, find the best available prior target.
    Priority: RAPTOR > LEBRON > raw +/-

    Returns: PLAYER_ID, PLAYER_NAME, prior_target, prior_source
    """
    base = box_per100[["PLAYER_ID", "PLAYER_NAME", "PLUS_MINUS"]].copy()
    base = base.rename(columns={"PLUS_MINUS": "prior_target"})
    base["prior_source"] = "raw_plus_minus"

    # Try RAPTOR (name-matched)
    raptor = get_raptor(season)
    if raptor is not None:
        col = next((c for c in ["raptor_total", "war_total"] if c in raptor.columns), None)
        if col:
            rmap = dict(zip(
                raptor["player_name"].str.lower().str.strip(),
                raptor[col]
            ))
            mask = base["PLAYER_NAME"].str.lower().str.strip().isin(rmap)
            base.loc[mask, "prior_target"] = (
                base.loc[mask, "PLAYER_NAME"].str.lower().str.strip().map(rmap)
            )
            base.loc[mask, "prior_source"] = "raptor"
            print(f"  [prior] RAPTOR matched {mask.sum()}/{len(base)} players")

    # Try LEBRON (id-matched - more precise, overwrites RAPTOR)
    lebron = get_lebron(season)
    if lebron is not None and "lebron" in lebron.columns and "player_id" in lebron.columns:
        lmap = dict(zip(lebron["player_id"].astype(str), lebron["lebron"]))
        mask = base["PLAYER_ID"].astype(str).isin(lmap)
        base.loc[mask, "prior_target"] = (
            base.loc[mask, "PLAYER_ID"].astype(str).map(lmap)
        )
        base.loc[mask, "prior_source"] = "lebron"
        print(f"  [prior] LEBRON matched {mask.sum()}/{len(base)} players")

    # DARKO (id-matched, overwrites all - most current and accurate)
    darko = get_darko(season)
    if darko is not None and "darko" in darko.columns:
        dmap = dict(zip(darko["player_id"].astype(str), darko["darko"]))
        mask = base["PLAYER_ID"].astype(str).isin(dmap)
        base.loc[mask, "prior_target"] = base.loc[mask, "PLAYER_ID"].astype(str).map(dmap)
        base.loc[mask, "prior_source"] = "darko"
        print(f"  [prior] DARKO matched {mask.sum()}/{len(base)} players")

    print(f"  [prior] sources: {base['prior_source'].value_counts().to_dict()}")
    return base


# ═══════════════════════════════════════════════════════════════════════════
# Box scores + tracking
# ═══════════════════════════════════════════════════════════════════════════

def get_box_scores(season: str = "2023-24") -> pd.DataFrame:
    def fetch():
        ep = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="Per100Possessions",
            measure_type_detailed_defense="Base",
        )
        df = ep.get_data_frames()[0]; time.sleep(NBA_API_DELAY); return df
    return _load_or_fetch("box_per100", season, fetch)


def get_advanced_box(season: str = "2023-24") -> pd.DataFrame:
    def fetch():
        ep = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Advanced",
        )
        df = ep.get_data_frames()[0]; time.sleep(NBA_API_DELAY); return df
    return _load_or_fetch("box_advanced", season, fetch)


def _tracking_endpoint(name: str, pt_type: str, season: str) -> pd.DataFrame:
    def fetch():
        ep = leaguedashptstats.LeagueDashPtStats(
            season=season,
            season_type_all_star="Regular Season",
            pt_measure_type=pt_type,
        )
        df = ep.get_data_frames()[0]; time.sleep(NBA_API_DELAY); return df
    try:
        return _load_or_fetch(name, season, fetch)
    except Exception as e:
        print(f"  [skip] {name} failed: {e}")
        return pd.DataFrame()


def _get_player_defense(season: str) -> pd.DataFrame:
    """Player-level defensive tracking via LeagueDashPtDefend."""
    from nba_api.stats.endpoints import leaguedashptdefend
    def fetch():
        ep = leaguedashptdefend.LeagueDashPtDefend(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_simple="PerGame",
            defense_category="Overall",
        )
        df = ep.get_data_frames()[0]
        # Rename the player ID column so merges work downstream
        df = df.rename(columns={"CLOSE_DEF_PERSON_ID": "PLAYER_ID"})
        time.sleep(NBA_API_DELAY)
        return df
    try:
        return _load_or_fetch("tracking_player_defense", season, fetch)
    except Exception as e:
        print(f"  [skip] player defense failed: {e}")
        return pd.DataFrame()

def _get_hustle_stats(season: str) -> pd.DataFrame:
    """Player hustle stats - contested shots, screen assists, deflections."""
    from nba_api.stats.endpoints import leaguehustlestatsplayer
    def fetch():
        ep = leaguehustlestatsplayer.LeagueHustleStatsPlayer(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_time="PerGame",
        )
        df = ep.get_data_frames()[0]
        time.sleep(NBA_API_DELAY)
        return df
    try:
        return _load_or_fetch("tracking_hustle", season, fetch)
    except Exception as e:
        print(f"  [skip] hustle stats failed: {e}")
        return pd.DataFrame()


def _get_ptshot_stats(season: str) -> pd.DataFrame:
    """Player shot quality - FG%, EFG%, 2pt/3pt splits."""
    from nba_api.stats.endpoints import leaguedashplayerptshot
    def fetch():
        ep = leaguedashplayerptshot.LeagueDashPlayerPtShot(
            season=season,
            season_type_all_star="Regular Season",
        )
        df = ep.get_data_frames()[0]
        time.sleep(NBA_API_DELAY)
        return df
    try:
        return _load_or_fetch("tracking_ptshot", season, fetch)
    except Exception as e:
        print(f"  [skip] ptshot stats failed: {e}")
        return pd.DataFrame()



def get_passing_stats(season: str = "2023-24") -> pd.DataFrame:
    """Passing tracking: potential assists, adjusted assists, passes made."""
    return _tracking_endpoint("tracking_passing", "Passing", season)


def get_lineup_shot_profile(season: str = "2023-24") -> pd.DataFrame:
    """
    Fetch lineup-level shot profile from LeagueDashLineups.

    Makes three calls:
      - Base:     FGA, FG3A, FTA, TOV, MIN (team offensive counting stats)
      - Misc:     PTS_PAINT (paint points - lineup-level rim proxy)
      - Opponent: OPP_FGA, OPP_FGM, OPP_FG3A, OPP_FTA, OPP_TOV
                  (what the opponent did against each lineup - used for
                   defensive on/off: opp FG%, opp 3PA rate, forced TOV rate)

    All three are merged on GROUP_ID and cached as a single parquet.
    Used by data/playmaking.py to compute true team on/off for six metrics
    (3PA rate, TOV rate, rim rate on offense; opp FG%, opp 3PA rate,
    forced TOV rate on defense).
    """
    from nba_api.stats.endpoints import leaguedashlineups

    def fetch():
        def _call(measure_type, retries=3):
            for attempt in range(retries):
                try:
                    ep = leaguedashlineups.LeagueDashLineups(
                        season=season,
                        season_type_all_star="Regular Season",
                        per_mode_detailed="Totals",
                        measure_type_detailed_defense=measure_type,
                        group_quantity="5",
                        timeout=60,
                    )
                    df = ep.get_data_frames()[0]
                    time.sleep(NBA_API_DELAY)
                    return df
                except Exception as e:
                    wait = (attempt + 1) * 20
                    print(f"  [retry {attempt+1}/{retries}] lineup {measure_type} failed: "
                          f"{e.__class__.__name__} - waiting {wait}s")
                    time.sleep(wait)
            return None

        df_base = _call("Base")
        if df_base is None:
            print("  [warning] lineup Base fetch failed - returning empty")
            return pd.DataFrame()

        time.sleep(2)
        df_misc = _call("Misc")      # PTS_PAINT

        time.sleep(2)
        df_opp  = _call("Opponent")  # OPP_FGA, OPP_FGM, OPP_FG3A, OPP_FTA, OPP_TOV

        id_col = "GROUP_ID" if "GROUP_ID" in df_base.columns else df_base.columns[1]

        def parse_ids(group_id):
            try:
                return [int(x) for x in str(group_id).split("-") if x.strip()]
            except Exception:
                return []

        result = pd.DataFrame()
        result["group_id"]   = df_base[id_col]
        result["player_ids"] = df_base[id_col].apply(parse_ids)
        result["team_id"]    = df_base["TEAM_ID"] if "TEAM_ID" in df_base.columns else None

        for col in ["FGA", "FG3A", "FG3M", "FTA", "FTM", "TOV", "MIN"]:
            if col in df_base.columns:
                result[col] = df_base[col].fillna(0).astype(float)

        # Misc -> PTS_PAINT
        if df_misc is not None and not df_misc.empty:
            misc_id   = "GROUP_ID" if "GROUP_ID" in df_misc.columns else df_misc.columns[1]
            misc_keep = [misc_id] + [c for c in ["PTS_PAINT"] if c in df_misc.columns]
            if len(misc_keep) > 1:
                result = result.merge(
                    df_misc[misc_keep].rename(columns={misc_id: "group_id"}),
                    on="group_id", how="left",
                )
        else:
            print("  [warning] lineup Misc fetch failed - rim_rate_delta will be unavailable")

        # Opponent -> defensive counting stats
        opp_want = ["OPP_FGA", "OPP_FGM", "OPP_FG3A", "OPP_FG3M", "OPP_FTA", "OPP_TOV"]
        if df_opp is not None and not df_opp.empty:
            opp_id    = "GROUP_ID" if "GROUP_ID" in df_opp.columns else df_opp.columns[1]
            avail_opp = [c for c in opp_want if c in df_opp.columns]
            if avail_opp:
                result = result.merge(
                    df_opp[[opp_id] + avail_opp].rename(columns={opp_id: "group_id"}),
                    on="group_id", how="left",
                )
            else:
                print(f"  [warning] Opponent cols not found; got: {df_opp.columns.tolist()[:12]}")
        else:
            print("  [warning] lineup Opponent fetch failed - defensive on/off unavailable")

        if "MIN" in result.columns:
            result = result[result["MIN"] >= 2].reset_index(drop=True)

        result = result.drop(columns=["group_id"])
        return result

    return _load_or_fetch("lineup_shot_profile_v4", season, fetch)


def get_all_tracking(season: str = "2023-24") -> pd.DataFrame:
    path = _cache_path("tracking_merged", season)
    if path.exists():
        print("  [cache] tracking_merged")
        return pd.read_parquet(path)

    defense = _get_player_defense(season)   # D_FG_PCT, D_FGA, PCT_PLUSMINUS
    time.sleep(3)
    hustle  = _get_hustle_stats(season)     # CONTESTED_SHOTS, SCREEN_ASSISTS, DEFLECTIONS
    time.sleep(3)
    ptshot  = _get_ptshot_stats(season)     # EFG_PCT, FG3_PCT, FG2_PCT
    time.sleep(3)
    passing = get_passing_stats(season)     # POTENTIAL_AST, AST_ADJ, PASSES_MADE

    tables = [t for t in [defense, hustle, ptshot, passing]
              if not t.empty and "PLAYER_ID" in t.columns]
    if not tables:
        print("  [warning] all tracking failed - returning empty")
        return pd.DataFrame()

    merged = tables[0].copy()
    for tbl in tables[1:]:
        new_cols = ["PLAYER_ID"] + [col for col in tbl.columns
                                     if col not in merged.columns and col != "PLAYER_ID"]
        if len(new_cols) > 1:
            merged = merged.merge(tbl[new_cols], on="PLAYER_ID", how="left")

    merged.to_parquet(path, index=False)
    print(f"  [tracking] {len(merged)} players, {merged.shape[1]} cols")
    print(f"  [tracking] columns: {merged.columns.tolist()}")
    return merged

