"""
output/team_report.py
=====================
Generates a team-specific HTML report with:
  - Date range filter (early vs recent window comparison)
  - Roster RAPM rankings
  - Best lineups for that team
  - Best pairings among teammates
  - Player deep-dives with tracking flags

Usage:
    from output.team_report import generate_team_report

    generate_team_report(
        team="Hornets",
        season="2025-26",
        split_date="2025-12-15",
        path=r"C:\nba_valuation\output\hornets_report.html",
    )
"""

import time
import html as html_lib
from pathlib import Path


TEAM_COLORS = {
    "hornets": {"primary": "#00788C", "secondary": "#1D1160", "accent": "#A1A1A4"},
    "lakers":  {"primary": "#FDB927", "secondary": "#552583", "accent": "#ffffff"},
    "celtics": {"primary": "#007A33", "secondary": "#BA9653", "accent": "#ffffff"},
    "warriors":{"primary": "#1D428A", "secondary": "#FFC72C", "accent": "#ffffff"},
    "nuggets": {"primary": "#0E2240", "secondary": "#FEC524", "accent": "#8B2131"},
    "heat":    {"primary": "#98002E", "secondary": "#F9A01B", "accent": "#ffffff"},
    "knicks":  {"primary": "#006BB6", "secondary": "#F58426", "accent": "#ffffff"},
    "bucks":   {"primary": "#00471B", "secondary": "#EEE1C6", "accent": "#ffffff"},
    "suns":    {"primary": "#1D1160", "secondary": "#E56020", "accent": "#63727A"},
    "clippers":{"primary": "#C8102E", "secondary": "#1D428A", "accent": "#BEC0C2"},
    "default": {"primary": "#3b82f6", "secondary": "#1e293b", "accent": "#94a3b8"},
}

TEAM_ABBREVS = {
    "hornets": "CHA", "lakers": "LAL", "celtics": "BOS",
    "warriors": "GSW", "nuggets": "DEN", "heat": "MIA",
    "knicks": "NYK", "bucks": "MIL", "suns": "PHX",
    "clippers": "LAC", "nets": "BKN", "bulls": "CHI",
    "hawks": "ATL", "sixers": "PHI", "raptors": "TOR",
    "pacers": "IND", "cavaliers": "CLE", "pistons": "DET",
    "wizards": "WAS", "magic": "ORL", "thunder": "OKC",
    "blazers": "POR", "jazz": "UTA", "kings": "SAC",
    "wolves": "MIN", "timberwolves": "MIN", "spurs": "SAS",
    "mavericks": "DAL", "rockets": "HOU", "grizzlies": "MEM",
    "pelicans": "NOP",
}


def _rapm_color(v):
    if v is None: return "#888"
    if v >= 2.5:  return "#22c55e"
    if v >= 1.0:  return "#86efac"
    if v >= 0:    return "#a7f3d0"
    if v >= -1.0: return "#fca5a5"
    return "#ef4444"

def _support_color(v):
    if v is None: return "#888"
    if v >= 70: return "#22c55e"
    if v >= 45: return "#f59e0b"
    return "#ef4444"

def _delta_color(v):
    if v >= 5:  return "#22c55e"
    if v >= 0:  return "#86efac"
    if v >= -5: return "#fca5a5"
    return "#ef4444"


def _get_team_players(validated, team_abbrev):
    """Filter validated players to those on the target team."""
    import pandas as pd
    # Try matching on TEAM_ABBREVIATION if available, else fall back to name search
    if "team" in validated.columns:
        return validated[validated["team"].str.upper() == team_abbrev.upper()]
    # Fall back: match via rapm df which has player_id - just return all and note
    return validated


def generate_team_report(
    team: str,
    season: str = "2025-26",
    split_date: str = None,
    path: str = None,
    max_games: int = 100,
) -> str:
    """
    Generate a team-specific HTML report.

    team:        Team name e.g. "hornets", "lakers"
    split_date:  "YYYY-MM-DD" - if provided, shows early vs recent comparison
    path:        Where to save the HTML file
    """
    import pandas as pd
    from data.date_split import compare_windows, get_stints_daterange
    from output.report import run_full_pipeline

    team_key    = team.lower().replace(" ", "")
    colors      = TEAM_COLORS.get(team_key, TEAM_COLORS["default"])
    team_abbrev = TEAM_ABBREVS.get(team_key, team.upper()[:3])
    team_title  = team.title()

    print(f"\n[team_report] Generating {team_title} report for {season}...")

    # ── Run full pipeline ────────────────────────────────────────────────
    out = run_full_pipeline(season=season, max_games=max_games)

    # ── Get team roster from tracking/box data ───────────────────────────
    from data.ingest import get_box_scores
    box = get_box_scores(season)
    team_players = box[
        box["TEAM_ABBREVIATION"].str.upper() == team_abbrev.upper()
    ]["PLAYER_ID"].astype(str).tolist()

    if not team_players:
        print(f"  [warning] No players found for {team_abbrev} - showing all players")

    # Filter validated to team
    validated = out["validated"].copy()
    if team_players:
        validated = validated[validated["player_id"].astype(str).isin(team_players)]
    validated = validated.sort_values("rapm", ascending=False).reset_index(drop=True)

    # Filter pairs to team (both players on same team)
    pairs = out["pairs"].copy()
    if team_players:
        pairs = pairs[
            pairs["player_a_id"].astype(str).isin(team_players) &
            pairs["player_b_id"].astype(str).isin(team_players)
        ]
    pairs = pairs.sort_values("compat_shrunk", ascending=False)

    # Filter synergy to team lineups
    synergy = out["synergy"].copy()
    if team_players:
        synergy = synergy[synergy["player_ids"].apply(
            lambda ids: all(str(p) in team_players for p in ids)
        )]
    synergy["players"] = synergy["player_names"].apply(lambda x: ", ".join(x))
    synergy = synergy.sort_values("synergy_shrunk", ascending=False)

    # ── Date split if requested ──────────────────────────────────────────
    split_results = None
    if split_date:
        print(f"  [split] Running date windows around {split_date}...")
        split_results = compare_windows(
            season=season,
            split_date=split_date,
        )

    # ── Build roster table ───────────────────────────────────────────────
    roster_rows = []
    for i, (_, row) in enumerate(validated.iterrows(), 1):
        rapm_v  = row.get("rapm")
        supp_v  = row.get("support_score")
        flags_g = row.get("green_flags", []) or []
        flags_r = row.get("red_flags", []) or []
        flags_html = ""
        for f in flags_g:
            flags_html += f'<span class="flag flag-green">{html_lib.escape(str(f))}</span>'
        for f in flags_r:
            flags_html += f'<span class="flag flag-red">{html_lib.escape(str(f))}</span>'

        # Split comparison delta
        delta_cell = ""
        if split_results and not split_results["comparison"].empty:
            comp = split_results["comparison"]
            crow = comp[comp["player_name"] == row.get("player_name","")]
            if not crow.empty:
                d = crow.iloc[0]["rapm_delta"]
                col = _rapm_color(d)
                delta_cell = f'<span style="color:{col};font-weight:600">{d:+.2f}</span>'

        roster_rows.append(f"""
        <tr>
          <td class="num">{i}</td>
          <td class="player-name">{html_lib.escape(str(row.get("player_name","-")))}</td>
          <td class="num"><span style="color:{_rapm_color(rapm_v)};font-weight:600">{f"{rapm_v:+.2f}" if rapm_v is not None else "-"}</span></td>
          <td class="num">{f"{row.get('prior',0):+.2f}" if row.get('prior') is not None else "-"}</td>
          <td class="num"><span style="color:{_support_color(supp_v)}">{f"{supp_v:.0f}" if supp_v is not None else "N/A"}</span></td>
          <td class="num">{f"{row.get('o_support',0):.0f}" if row.get('o_support') is not None else "-"}</td>
          <td class="num">{f"{row.get('d_support',0):.0f}" if row.get('d_support') is not None else "-"}</td>
          <td class="num">{delta_cell or "-"}</td>
          <td class="flags">{flags_html}</td>
        </tr>""")

    # ── Lineup table ─────────────────────────────────────────────────────
    lineup_rows = []
    for i, (_, row) in enumerate(synergy.head(20).iterrows(), 1):
        delta  = row.get("synergy_delta", 0)
        shrunk = row.get("synergy_shrunk", delta)
        conf   = row.get("confidence", 0)
        conf_color = "#22c55e" if conf >= 50 else "#f59e0b" if conf >= 25 else "#ef4444"
        lineup_rows.append(f"""
        <tr>
          <td class="num">{i}</td>
          <td class="player-name">{html_lib.escape(str(row.get("players","-")))}</td>
          <td class="num">{row.get("possessions",0):.0f}</td>
          <td class="num">{row.get("actual_rapm",0):+.1f}</td>
          <td class="num" style="color:#666">{delta:+.1f}</td>
          <td class="num"><span style="color:{_delta_color(shrunk)};font-weight:600">{shrunk:+.1f}</span></td>
          <td class="num"><span style="color:{conf_color}">{conf:.0f}%</span></td>
        </tr>""")

    # ── Pairs table ───────────────────────────────────────────────────────
    pair_rows = []
    for i, (_, row) in enumerate(pairs.head(20).iterrows(), 1):
        compat = row.get("compatibility", 0)
        shrunk = row.get("compat_shrunk", compat)
        conf   = row.get("confidence", 0)
        conf_color = "#22c55e" if conf >= 50 else "#f59e0b" if conf >= 25 else "#ef4444"
        pair_rows.append(f"""
        <tr>
          <td class="num">{i}</td>
          <td class="player-name">{html_lib.escape(str(row.get("player_a","-")))}</td>
          <td class="player-name">{html_lib.escape(str(row.get("player_b","-")))}</td>
          <td class="num">{row.get("shared_poss",0):.0f}</td>
          <td class="num">{row.get("rapm_a",0):+.2f}</td>
          <td class="num">{row.get("rapm_b",0):+.2f}</td>
          <td class="num" style="color:#666">{compat:+.2f}</td>
          <td class="num"><span style="color:{_delta_color(shrunk)};font-weight:600">{shrunk:+.2f}</span></td>
          <td class="num"><span style="color:{conf_color}">{conf:.0f}%</span></td>
        </tr>""")

    # ── Split section ─────────────────────────────────────────────────────
    split_html = ""
    if split_results and not split_results["comparison"].empty:
        comp = split_results["comparison"].copy()
        # Filter to team players
        if team_players:
            comp = comp[comp["player_id"].astype(str).isin(team_players)]
        comp = comp.sort_values("rapm_delta", ascending=False)

        split_rows = []
        for i, (_, row) in enumerate(comp.iterrows(), 1):
            d = row["rapm_delta"]
            split_rows.append(f"""
            <tr>
              <td class="num">{i}</td>
              <td class="player-name">{html_lib.escape(str(row.get("player_name","-")))}</td>
              <td class="num">{row.get("rapm_early",0):+.2f}</td>
              <td class="num">{row.get("rapm_recent",0):+.2f}</td>
              <td class="num"><span style="color:{_rapm_color(d)};font-weight:600">{d:+.2f}</span></td>
            </tr>""")

        split_html = f"""
        <div id="split" class="section">
          <div class="section-title">Early vs recent comparison</div>
          <div class="section-desc">Split at {split_date}. Delta = recent RAPM minus early RAPM - positive means player improved.</div>
          <table class="data-table">
            <thead><tr>
              <th>#</th><th>Player</th>
              <th style="text-align:right">Early RAPM</th>
              <th style="text-align:right">Recent RAPM</th>
              <th style="text-align:right">Delta</th>
            </tr></thead>
            <tbody>{"".join(split_rows)}</tbody>
          </table>
        </div>"""

    split_nav = '<button onclick="show(\'split\', this)">Early vs Recent</button>' if split_date else ""

    # ── Summary stats ────────────────────────────────────────────────────
    best_player = validated.iloc[0]["player_name"] if len(validated) else "-"
    best_rapm   = validated.iloc[0]["rapm"] if len(validated) else 0
    n_players   = len(validated)
    avg_rapm    = validated["rapm"].mean() if len(validated) else 0
    best_lineup = synergy.iloc[0]["players"][:50] if len(synergy) else "-"

    # ── Assemble HTML ─────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{team_title} - NBA Valuation {season}</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --primary:  {colors["primary"]};
    --secondary:{colors["secondary"]};
    --bg:       #0e0f11;
    --surface:  #16181c;
    --border:   #2a2d35;
    --text:     #e8eaf0;
    --muted:    #6b7280;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:'DM Sans',sans-serif; font-size:14px; line-height:1.6; }}
  .header {{
    padding: 0;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(135deg, var(--secondary) 0%, var(--primary) 100%);
  }}
  .header-inner {{ padding: 40px 40px 32px; }}
  .header h1 {{ font-size:34px; font-weight:300; letter-spacing:-0.5px; color:#fff; }}
  .header h1 strong {{ font-weight:700; }}
  .header p {{ color:rgba(255,255,255,0.65); margin-top:6px; font-size:13px; }}
  .summary-cards {{
    display:grid; grid-template-columns:repeat(4,1fr);
    gap:16px; padding:28px 40px; border-bottom:1px solid var(--border);
  }}
  .card {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:18px 20px; }}
  .card.accent {{ border-top:3px solid var(--primary); }}
  .card .label {{ font-size:11px; text-transform:uppercase; letter-spacing:1px; color:var(--muted); }}
  .card .value {{ font-size:24px; font-weight:600; margin-top:4px; color:var(--text); }}
  .card .sub {{ font-size:12px; color:var(--muted); margin-top:2px; }}
  nav {{ display:flex; gap:4px; padding:20px 40px 0; border-bottom:1px solid var(--border); }}
  nav button {{
    background:none; border:none; color:var(--muted);
    padding:10px 18px; font-family:'DM Sans',sans-serif; font-size:13px;
    cursor:pointer; border-bottom:2px solid transparent; transition:all .15s;
  }}
  nav button:hover {{ color:var(--text); }}
  nav button.active {{ color:var(--primary); border-bottom-color:var(--primary); }}
  .section {{ display:none; padding:32px 40px; }}
  .section.active {{ display:block; }}
  .section-title {{ font-size:18px; font-weight:500; margin-bottom:6px; }}
  .section-desc {{ color:var(--muted); font-size:13px; margin-bottom:20px; }}
  .data-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  .data-table th {{
    text-align:left; padding:10px 12px; color:var(--muted);
    font-size:11px; text-transform:uppercase; letter-spacing:0.5px;
    font-weight:500; border-bottom:1px solid var(--border); white-space:nowrap;
  }}
  .data-table td {{ padding:10px 12px; border-bottom:1px solid var(--border); vertical-align:top; }}
  .data-table tr:hover td {{ background:var(--surface); }}
  .data-table td.num {{ text-align:right; font-family:'DM Mono',monospace; white-space:nowrap; }}
  .data-table td.player-name {{ font-weight:500; white-space:nowrap; }}
  .flag {{ display:inline-block; font-size:11px; padding:2px 8px; border-radius:4px; margin:2px; line-height:1.4; }}
  .flag-green {{ background:#052e16; color:#86efac; }}
  .flag-red   {{ background:#2c0a0a; color:#fca5a5; }}
  @media(max-width:900px) {{
    .summary-cards {{ grid-template-columns:1fr 1fr; }}
    .header-inner, nav, .section {{ padding-left:20px; padding-right:20px; }}
  }}
</style>
</head>
<body>

<div class="header">
  <div class="header-inner">
    <h1><strong>{team_title}</strong> Valuation Report</h1>
    <p>Season {season} &nbsp;·&nbsp; RAPM + tracking + lineup synergy{f' &nbsp;·&nbsp; split at {split_date}' if split_date else ''}</p>
  </div>
</div>

<div class="summary-cards">
  <div class="card accent">
    <div class="label">Best player (RAPM)</div>
    <div class="value" style="font-size:16px;margin-top:6px">{html_lib.escape(str(best_player))}</div>
    <div class="sub">{best_rapm:+.2f} RAPM</div>
  </div>
  <div class="card">
    <div class="label">Roster rated</div>
    <div class="value">{n_players}</div>
    <div class="sub">players with data</div>
  </div>
  <div class="card">
    <div class="label">Team avg RAPM</div>
    <div class="value" style="color:{_rapm_color(avg_rapm)}">{avg_rapm:+.2f}</div>
    <div class="sub">across roster</div>
  </div>
  <div class="card">
    <div class="label">Best lineup</div>
    <div class="value" style="font-size:11px;margin-top:8px;line-height:1.5">{html_lib.escape(best_lineup)}</div>
    <div class="sub">by shrunk synergy</div>
  </div>
</div>

<nav>
  <button class="active" onclick="show('roster', this)">Roster</button>
  <button onclick="show('lineups', this)">Lineups</button>
  <button onclick="show('pairs', this)">Pairings</button>
  {split_nav}
</nav>

<!-- ROSTER -->
<div id="roster" class="section active">
  <div class="section-title">{team_title} roster ratings</div>
  <div class="section-desc">RAPM = on/off impact per 100 possessions. Support = how well tracking explains the RAPM. Delta = recent minus early RAPM (if split enabled).</div>
  <table class="data-table">
    <thead><tr>
      <th>#</th><th>Player</th>
      <th style="text-align:right">RAPM</th>
      <th style="text-align:right">Prior</th>
      <th style="text-align:right">Support</th>
      <th style="text-align:right">Off</th>
      <th style="text-align:right">Def</th>
      <th style="text-align:right">Delta</th>
      <th>Flags</th>
    </tr></thead>
    <tbody>{"".join(roster_rows)}</tbody>
  </table>
</div>

<!-- LINEUPS -->
<div id="lineups" class="section">
  <div class="section-title">Best {team_title} lineups</div>
  <div class="section-desc">Synergy delta = actual net rating minus sum of individual RAPMs. Shrunk adjusts for sample size.</div>
  <table class="data-table">
    <thead><tr>
      <th>#</th><th>Players</th>
      <th style="text-align:right">Poss</th>
      <th style="text-align:right">Actual NR</th>
      <th style="text-align:right">Raw delta</th>
      <th style="text-align:right">Shrunk</th>
      <th style="text-align:right">Confidence</th>
    </tr></thead>
    <tbody>{"".join(lineup_rows)}</tbody>
  </table>
</div>

<!-- PAIRS -->
<div id="pairs" class="section">
  <div class="section-title">Best {team_title} pairings</div>
  <div class="section-desc">Compatibility = pair net rating minus sum of individual RAPMs. Shrunk adjusts for sample size.</div>
  <table class="data-table">
    <thead><tr>
      <th>#</th><th>Player A</th><th>Player B</th>
      <th style="text-align:right">Poss</th>
      <th style="text-align:right">RAPM A</th>
      <th style="text-align:right">RAPM B</th>
      <th style="text-align:right">Raw compat</th>
      <th style="text-align:right">Shrunk</th>
      <th style="text-align:right">Conf</th>
    </tr></thead>
    <tbody>{"".join(pair_rows)}</tbody>
  </table>
</div>

{split_html}

<script>
function show(id, btn) {{
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
}}
</script>
</body>
</html>"""

    if path:
        Path(path).write_text(html, encoding="utf-8")
        print(f"[team_report] saved -> {path}")

    return html
