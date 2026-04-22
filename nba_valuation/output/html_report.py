

from pathlib import Path
import html as html_lib


def _pct_bar(value, max_val=100, color="#3b82f6"):
    pct = max(0, min(100, (value / max_val) * 100))
    return f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.0f}%;background:{color}"></div></div>'


def _rapm_color(v):
    if v is None: return "#888"
    if v >= 2.5:  return "#22c55e"
    if v >= 1.0:  return "#86efac"
    if v >= 0:    return "#d1fae5"
    if v >= -1.0: return "#fecaca"
    return "#ef4444"


def _support_color(v):
    if v is None: return "#888"
    if v >= 70: return "#22c55e"
    if v >= 45: return "#f59e0b"
    return "#ef4444"


def _delta_color(v):
    if v >= 5:   return "#22c55e"
    if v >= 0:   return "#86efac"
    if v >= -5:  return "#fca5a5"
    return "#ef4444"


def generate_report(
    out: dict,
    path: str = None,
    season: str = "2023-24",
    top_n_lineups: int = 50,
    top_n_players: int = 30,
    top_n_pairs: int = 30,
    min_pair_poss: float = 60.0,
) -> str:
    """
    Generate a self-contained HTML report.
    Returns the HTML string and optionally saves to path.
    """
    import pandas as pd

    rapm      = out["rapm"].copy()
    validated = out["validated"].copy()
    synergy   = out["synergy"].copy()
    pairs     = out["pairs"].copy()
    screens   = out["screens"]

    synergy["players"] = synergy["player_names"].apply(lambda x: ", ".join(x))
    synergy = synergy.sort_values("synergy_delta", ascending=False)

    reliable_pairs = pairs[pairs["shared_poss"] >= min_pair_poss].sort_values(
        "compatibility", ascending=False
    )

    # ── Section builders ──────────────────────────────────────────────────

    def player_rows(df, cols, formatters=None):
        rows = []
        for i, (_, row) in enumerate(df.iterrows(), 1):
            cells = [f'<td class="num">{i}</td>']
            for col in cols:
                v = row.get(col)
                fmt = formatters.get(col) if formatters else None
                if fmt:
                    cells.append(f'<td>{fmt(v, row)}</td>')
                else:
                    cells.append(f'<td>{v if v is not None else "—"}</td>')
            rows.append(f'<tr>{"".join(cells)}</tr>')
        return "\n".join(rows)

    # ── Players table ─────────────────────────────────────────────────────
    def rapm_cell(v, row): return f'<span style="color:{_rapm_color(v)};font-weight:600">{v:+.2f}</span>' if v is not None else "—"
    def support_cell(v, row):
        if v is None: return '<span style="color:#888">N/A</span>'
        return f'<span style="color:{_support_color(v)}">{v:.0f}</span>'
    def name_cell(v, row): return f'<td class="player-name">{html_lib.escape(str(v))}</td>'

    # Build rows for ALL players; rows beyond top_n_players get hidden-player class
    player_table_rows = []
    for i, (_, row) in enumerate(validated.iterrows(), 1):
        rapm_v  = row.get("rapm")
        supp_v  = row.get("support_score")
        flags_r = row.get("red_flags", [])
        flags_g = row.get("green_flags", [])
        flags_html = ""
        for f in (flags_g or []):
            flags_html += f'<span class="flag flag-green">{html_lib.escape(str(f))}</span>'
        for f in (flags_r or []):
            flags_html += f'<span class="flag flag-red">{html_lib.escape(str(f))}</span>'

        hidden_cls = ' class="hidden-player"' if i > top_n_players else ""
        name_lower = html_lib.escape(str(row.get("player_name", "")).lower())

        player_table_rows.append(f"""
        <tr{hidden_cls} data-name="{name_lower}">
          <td class="num">{i}</td>
          <td class="player-name">{html_lib.escape(str(row.get("player_name","—")))}</td>
          <td class="num"><span style="color:{_rapm_color(rapm_v)};font-weight:600">{f"{rapm_v:+.2f}" if rapm_v is not None else "—"}</span></td>
          <td class="num">{f"{row.get('prior',0):+.2f}" if row.get('prior') is not None else "—"}</td>
          <td class="num">{f"{row.get('rapm_adjustment',0):+.2f}" if row.get('rapm_adjustment') is not None else "—"}</td>
          <td class="num"><span style="color:{_support_color(supp_v)}">{f"{supp_v:.0f}" if supp_v is not None else "N/A"}</span></td>
          <td class="num">{f"{row.get('o_support',0):.0f}" if row.get('o_support') is not None else "—"}</td>
          <td class="num">{f"{row.get('d_support',0):.0f}" if row.get('d_support') is not None else "—"}</td>
          <td class="num">{f"{row.get('playmaking_support',0):.0f}" if row.get('playmaking_support') is not None else "—"}</td>
          <td class="flags">{flags_html}</td>
        </tr>""")

    # ── Lineup table ──────────────────────────────────────────────────────
    lineup_rows = []
    for i, (_, row) in enumerate(synergy.head(top_n_lineups).iterrows(), 1):
        delta   = row.get("synergy_delta", 0)
        shrunk  = row.get("synergy_shrunk", delta)
        conf    = row.get("confidence", 0)
        conf_color = "#22c55e" if conf >= 50 else "#f59e0b" if conf >= 25 else "#ef4444"
        lineup_rows.append(f"""
        <tr>
          <td class="num">{i}</td>
          <td class="player-name">{html_lib.escape(str(row.get("players","—")))}</td>
          <td class="num">{row.get("possessions",0):.0f}</td>
          <td class="num">{row.get("actual_rapm",0):+.1f}</td>
          <td class="num">{row.get("expected_rapm",0):+.1f}</td>
          <td class="num" style="color:#666">{delta:+.1f}</td>
          <td class="num"><span style="color:{_delta_color(shrunk)};font-weight:600">{shrunk:+.1f}</span></td>
          <td class="num"><span style="color:{conf_color}">{conf:.0f}%</span></td>
        </tr>""")

    # ── Pairs table ───────────────────────────────────────────────────────
    pair_rows = []
    for i, (_, row) in enumerate(reliable_pairs.head(top_n_pairs).iterrows(), 1):
        compat  = row.get("compatibility", 0)
        shrunk  = row.get("compat_shrunk", compat)
        conf    = row.get("confidence", 0)
        conf_color = "#22c55e" if conf >= 50 else "#f59e0b" if conf >= 25 else "#ef4444"
        pair_rows.append(f"""
        <tr>
          <td class="num">{i}</td>
          <td class="player-name">{html_lib.escape(str(row.get("player_a","—")))}</td>
          <td class="player-name">{html_lib.escape(str(row.get("player_b","—")))}</td>
          <td class="num">{row.get("shared_poss",0):.0f}</td>
          <td class="num">{row.get("rapm_a",0):+.2f}</td>
          <td class="num">{row.get("rapm_b",0):+.2f}</td>
          <td class="num" style="color:#666">{compat:+.2f}</td>
          <td class="num"><span style="color:{_delta_color(shrunk)};font-weight:600">{shrunk:+.2f}</span></td>
          <td class="num"><span style="color:{conf_color}">{conf:.0f}%</span></td>
        </tr>""")

    # ── Screener section ──────────────────────────────────────────────────
    def screener_table(df, label, color):
        if df.empty:
            return f'<p style="color:#666;font-style:italic">No players found</p>'
        rows_html = []
        for i, (_, row) in enumerate(df.iterrows(), 1):
            rapm_v = row.get("rapm")
            supp_v = row.get("support_score")
            rows_html.append(f"""
            <tr>
              <td class="num">{i}</td>
              <td class="player-name">{html_lib.escape(str(row.get("player_name","—")))}</td>
              <td class="num"><span style="color:{_rapm_color(rapm_v)};font-weight:600">{f"{rapm_v:+.2f}" if rapm_v is not None else "—"}</span></td>
              <td class="num">{f"{row.get('rapm_pct',0):.0f}" if row.get('rapm_pct') is not None else "—"}</td>
              <td class="num"><span style="color:{_support_color(supp_v)}">{f"{supp_v:.0f}" if supp_v is not None else "N/A"}</span></td>
            </tr>""")
        return f"""
        <table class="data-table">
          <thead><tr>
            <th>#</th><th>Player</th>
            <th>RAPM</th><th>RAPM pct</th><th>Support</th>
          </tr></thead>
          <tbody>{"".join(rows_html)}</tbody>
        </table>"""

    over_html   = screener_table(screens.get("overvalued",   pd.DataFrame()), "Overvalued",      "#ef4444")
    under_html  = screener_table(screens.get("undervalued",  pd.DataFrame()), "Undervalued",     "#22c55e")
    elite_html  = screener_table(screens.get("confirmed_elite", pd.DataFrame()), "Elite",         "#3b82f6")
    hidden_html = screener_table(screens.get("hidden_gems",  pd.DataFrame()), "Hidden Gems",     "#f59e0b")

    # ── Stats summary cards ───────────────────────────────────────────────
    top_player  = rapm.iloc[0]["player_name"] if len(rapm) else "—"
    top_rapm    = rapm.iloc[0]["rapm"] if len(rapm) else 0
    n_players   = len(validated)
    n_lineups   = len(synergy)
    n_pairs     = len(reliable_pairs)

    # ── Assemble HTML ─────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NBA Valuation Report — {season}</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:       #0e0f11;
    --surface:  #16181c;
    --border:   #2a2d35;
    --text:     #e8eaf0;
    --muted:    #6b7280;
    --accent:   #3b82f6;
    --green:    #22c55e;
    --red:      #ef4444;
    --amber:    #f59e0b;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    line-height: 1.6;
  }}
  .header {{
    padding: 48px 40px 32px;
    border-bottom: 1px solid var(--border);
  }}
  .header h1 {{
    font-size: 32px;
    font-weight: 300;
    letter-spacing: -0.5px;
    color: var(--text);
  }}
  .header h1 span {{ color: var(--accent); font-weight: 600; }}
  .header p {{ color: var(--muted); margin-top: 6px; font-size: 13px; }}
  .summary-cards {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    padding: 28px 40px;
    border-bottom: 1px solid var(--border);
  }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
  }}
  .card .label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); }}
  .card .value {{ font-size: 26px; font-weight: 600; margin-top: 4px; color: var(--text); }}
  .card .sub {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}
  nav {{
    display: flex;
    gap: 4px;
    padding: 20px 40px 0;
    border-bottom: 1px solid var(--border);
  }}
  nav button {{
    background: none;
    border: none;
    color: var(--muted);
    padding: 10px 18px;
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all .15s;
  }}
  nav button:hover {{ color: var(--text); }}
  nav button.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
  .section {{ display: none; padding: 32px 40px; }}
  .section.active {{ display: block; }}
  .section-title {{
    font-size: 18px;
    font-weight: 500;
    margin-bottom: 6px;
  }}
  .section-desc {{ color: var(--muted); font-size: 13px; margin-bottom: 20px; }}
  .data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }}
  .data-table th {{
    text-align: left;
    padding: 10px 12px;
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 500;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }}
  .data-table td {{
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }}
  .data-table tr:hover td {{ background: var(--surface); }}
  .data-table td.num {{
    text-align: right;
    font-family: 'DM Mono', monospace;
    white-space: nowrap;
    color: var(--text);
  }}
  .data-table td.player-name {{
    font-weight: 500;
    white-space: nowrap;
    color: var(--text);
  }}
  .flag {{
    display: inline-block;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    margin: 2px 2px;
    line-height: 1.4;
  }}
  .flag-green {{ background: #052e16; color: #86efac; }}
  .flag-red   {{ background: #2c0a0a; color: #fca5a5; }}
  .bar-track  {{ height: 4px; background: var(--border); border-radius: 2px; width: 80px; }}
  .bar-fill   {{ height: 4px; border-radius: 2px; }}
  .screener-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }}
  .screener-block h3 {{
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }}
  .overvalued-title  {{ color: var(--red); }}
  .undervalued-title {{ color: var(--green); }}
  .elite-title       {{ color: var(--accent); }}
  .hidden-title      {{ color: var(--amber); }}
  .hidden-player {{ display: none; }}
  .search-bar {{
    display: flex;
    gap: 10px;
    align-items: center;
    margin-bottom: 18px;
  }}
  .search-bar input {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    padding: 8px 14px;
    width: 280px;
    outline: none;
    transition: border-color .15s;
  }}
  .search-bar input:focus {{ border-color: var(--accent); }}
  .search-bar input::placeholder {{ color: var(--muted); }}
  .search-bar button {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--muted);
    cursor: pointer;
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    padding: 8px 14px;
    transition: all .15s;
  }}
  .search-bar button:hover {{ color: var(--text); border-color: var(--accent); }}
  @media (max-width: 900px) {{
    .summary-cards {{ grid-template-columns: 1fr 1fr; }}
    .screener-grid {{ grid-template-columns: 1fr; }}
    .header, nav, .section {{ padding-left: 20px; padding-right: 20px; }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>NBA <span>Valuation</span> Report</h1>
  <p>Season {season} &nbsp;·&nbsp; RAPM + tracking validation + lineup synergy</p>
</div>

<div class="summary-cards">
  <div class="card">
    <div class="label">Top player</div>
    <div class="value" style="font-size:18px;margin-top:6px">{html_lib.escape(str(top_player))}</div>
    <div class="sub">RAPM {top_rapm:+.2f}</div>
  </div>
  <div class="card">
    <div class="label">Players rated</div>
    <div class="value">{n_players}</div>
    <div class="sub">with on/off + tracking</div>
  </div>
  <div class="card">
    <div class="label">Lineups scored</div>
    <div class="value">{n_lineups}</div>
    <div class="sub">synergy decomposed</div>
  </div>
  <div class="card">
    <div class="label">Reliable pairs</div>
    <div class="value">{n_pairs}</div>
    <div class="sub">≥{min_pair_poss:.0f} shared poss</div>
  </div>
</div>

<nav>
  <button class="active" onclick="show('players', this)">Players</button>
  <button onclick="show('screener', this)">Screener</button>
  <button onclick="show('lineups', this)">Lineups</button>
  <button onclick="show('pairs', this)">Pairings</button>
</nav>

<!-- PLAYERS -->
<div id="players" class="section active">
  <div class="section-title">Player ratings</div>
  <div class="section-desc">Sorted by RAPM. Support score measures how well tracking data explains the RAPM estimate.</div>
  <div class="search-bar">
    <input type="text" id="player-search" placeholder="Search any player…" oninput="filterPlayers(this.value)" />
    <button onclick="showAllPlayers()" id="show-all-btn">Show all {n_players}</button>
  </div>
  <table class="data-table">
    <thead><tr>
      <th>#</th>
      <th>Player</th>
      <th style="text-align:right">RAPM</th>
      <th style="text-align:right">Prior</th>
      <th style="text-align:right">Adjust</th>
      <th style="text-align:right">Support</th>
      <th style="text-align:right">Off</th>
      <th style="text-align:right">Def</th>
      <th style="text-align:right">Play</th>
      <th>Flags</th>
    </tr></thead>
    <tbody id="player-tbody">
      {"".join(player_table_rows)}
    </tbody>
  </table>
</div>

<!-- SCREENER -->
<div id="screener" class="section">
  <div class="section-title">Mispricing screener</div>
  <div class="section-desc">Players where RAPM and tracking support diverge — potential over or undervaluation signals.</div>
  <div class="screener-grid">
    <div class="screener-block">
      <h3 class="overvalued-title">Overvalued — high RAPM, low tracking support</h3>
      {over_html}
    </div>
    <div class="screener-block">
      <h3 class="undervalued-title">Undervalued — low RAPM, high tracking support</h3>
      {under_html}
    </div>
    <div class="screener-block">
      <h3 class="elite-title">Confirmed elite — high RAPM + high support</h3>
      {elite_html}
    </div>
    <div class="screener-block">
      <h3 class="hidden-title">Hidden gems — modest RAPM, tracking says more</h3>
      {hidden_html}
    </div>
  </div>
</div>

<!-- LINEUPS -->
<div id="lineups" class="section">
  <div class="section-title">Top {top_n_lineups} lineups by synergy</div>
  <div class="section-desc">Synergy delta = actual net rating minus sum of individual RAPMs. Shrunk = delta adjusted for sample size (k=500). Confidence = how much to trust the number.</div>
  <table class="data-table">
    <thead><tr>
      <th>#</th>
      <th>Players</th>
      <th style="text-align:right">Poss</th>
      <th style="text-align:right">Actual NR</th>
      <th style="text-align:right">Expected</th>
      <th style="text-align:right">Raw delta</th>
      <th style="text-align:right">Shrunk</th>
      <th style="text-align:right">Confidence</th>
    </tr></thead>
    <tbody>
      {"".join(lineup_rows)}
    </tbody>
  </table>
</div>

<!-- PAIRS -->
<div id="pairs" class="section">
  <div class="section-title">Pairwise compatibility</div>
  <div class="section-desc">Compatibility = pair net rating minus sum of individual RAPMs. Shrunk = sample-size adjusted estimate (k=300). Min {min_pair_poss:.0f} shared possessions.</div>
  <table class="data-table">
    <thead><tr>
      <th>#</th>
      <th>Player A</th>
      <th>Player B</th>
      <th style="text-align:right">Poss</th>
      <th style="text-align:right">RAPM A</th>
      <th style="text-align:right">RAPM B</th>
      <th style="text-align:right">Raw compat</th>
      <th style="text-align:right">Shrunk</th>
      <th style="text-align:right">Confidence</th>
    </tr></thead>
    <tbody>
      {"".join(pair_rows)}
    </tbody>
  </table>
</div>

<script>
function show(id, btn) {{
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
}}

function filterPlayers(query) {{
  const q = query.toLowerCase().trim();
  const rows = document.querySelectorAll('#player-tbody tr');
  const showAllBtn = document.getElementById('show-all-btn');
  if (!q) {{
    // Reset: show top-N, hide the rest (use inline style to override CSS class)
    rows.forEach(row => {{
      row.style.display = row.classList.contains('hidden-player') ? 'none' : 'table-row';
    }});
    showAllBtn.style.display = '';
    return;
  }}
  // Search mode: explicitly set display so inline style overrides the hidden-player CSS class
  showAllBtn.style.display = 'none';
  rows.forEach(row => {{
    row.style.display = (row.dataset.name || '').includes(q) ? 'table-row' : 'none';
  }});
}}

function showAllPlayers() {{
  document.querySelectorAll('#player-tbody tr').forEach(row => row.style.display = 'table-row');
  document.getElementById('show-all-btn').style.display = 'none';
}}
</script>

</body>
</html>"""

    if path:
        Path(path).write_text(html, encoding="utf-8")
        print(f"[report] saved → {path}")

    return html
