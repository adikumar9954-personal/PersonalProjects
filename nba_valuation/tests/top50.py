import sys
sys.path.insert(0, r"C:\nba_valuation")

from output.report import run_full_pipeline
from output.html_report import generate_report

out = run_full_pipeline(season="2025-26", max_games=100)

generate_report(out, path=r"C:\nba_valuation\output\nba_report.html", season="2025-26")

import pandas as pd
from output.report import run_full_pipeline

out = run_full_pipeline(season="2025-26", max_games=100)

# ── Lineup synergy top 50 ──────────────────────────────────────────────────

synergy = out["synergy"].copy()
synergy["players"] = synergy["player_names"].apply(lambda x: ", ".join(x))

def print_lineup_table(df, title, n=50):
    rows = df.head(n).copy()
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"  synergy delta = actual net rating minus sum of individual RAPMs")
    print(f"{'='*90}")
    print(f"  {'#':<4} {'Players':<52} {'Poss':>5}  {'Actual':>7}  {'Expected':>8}  {'Delta':>7}")
    print(f"  {'-'*4} {'-'*52} {'-'*5}  {'-'*7}  {'-'*8}  {'-'*7}")
    for i, (_, row) in enumerate(rows.iterrows(), 1):
        players = row["players"]
        if len(players) > 52:
            players = players[:49] + "..."
        print(f"  {i:<4} {players:<52} {row['possessions']:>5.0f}  "
              f"{row['actual_rapm']:>+7.1f}  {row['expected_rapm']:>+8.1f}  "
              f"{row['synergy_delta']:>+7.1f}")
    print(f"{'='*90}")

print_lineup_table(
    synergy.sort_values("synergy_delta", ascending=False),
    "TOP 50 LINEUPS BY SYNERGY DELTA",
)

print_lineup_table(
    synergy.sort_values("synergy_delta", ascending=True),
    "BOTTOM 10 LINEUPS BY SYNERGY DELTA",
    n=10,
)

# ── Top 20 players ────────────────────────────────────────────────────────

rapm = out["rapm"]
print(f"\n{'='*72}")
print(f"  TOP 20 PLAYERS BY RAPM  (2025-26)")
print(f"{'='*72}")
print(f"  {'#':<4} {'Player':<26} {'RAPM':>6}  {'Prior':>6}  {'Adjust':>7}  {'Min':>5}")
print(f"  {'-'*4} {'-'*26} {'-'*6}  {'-'*6}  {'-'*7}  {'-'*5}")
for i, (_, row) in enumerate(rapm.head(20).iterrows(), 1):
    print(f"  {i:<4} {row['player_name']:<26} {row['rapm']:>+6.2f}  "
          f"{row['prior']:>+6.2f}  {row['rapm_adjustment']:>+7.2f}  {row['minutes']:>5.0f}")
print(f"{'='*72}")

# ── Top pairings (min 60 shared poss) ─────────────────────────────────────

pairs = out["pairs"]
reliable = pairs[pairs["shared_poss"] >= 60].sort_values("compatibility", ascending=False)

print(f"\n{'='*80}")
print(f"  TOP 20 PAIRINGS  (min 60 shared possessions)")
print(f"{'='*80}")
print(f"  {'#':<4} {'Player A':<24} {'Player B':<24} {'Poss':>5}  {'Compat':>7}")
print(f"  {'-'*4} {'-'*24} {'-'*24} {'-'*5}  {'-'*7}")
for i, (_, row) in enumerate(reliable.head(20).iterrows(), 1):
    print(f"  {i:<4} {row['player_a']:<24} {row['player_b']:<24} "
          f"{row['shared_poss']:>5.0f}  {row['compatibility']:>+7.2f}")
print(f"{'='*80}")