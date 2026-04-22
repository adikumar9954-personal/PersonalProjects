import sys
sys.path.insert(0, r"C:\nba_valuation")

import pandas as pd
from output.report import run_full_pipeline
from output.html_report import generate_report

out = run_full_pipeline(season="2025-26", max_games=100)

validated = out["validated"]
pairs     = out["pairs"]
synergy   = out["synergy"]

for name in ["LaMelo Ball", "Luka Doncic"]:
    print(f"\n{'='*65}")
    print(f"  {name.upper()}")
    print(f"{'='*65}")

    # Player row
    row = validated[validated["player_name"].str.contains(name.split()[0], case=False)]
    if not row.empty:
        r = row.iloc[0]
        print(f"\n  RAPM          {r['rapm']:+.2f}  ({r['rapm_pct']:.0f}th pct)")
        print(f"  Prior         {r['prior']:+.2f}")
        rapm_row = out["rapm"][out["rapm"]["player_name"].str.contains(name.split()[0], case=False)]
        adj = rapm_row.iloc[0]["rapm_adjustment"] if not rapm_row.empty else 0
        print(f"  Adjustment    {adj:+.2f}")
        print(f"  Support       {r.get('support_score') or 'N/A'}")
        print(f"  Off support   {r.get('o_support') or 'N/A'}")
        print(f"  Def support   {r.get('d_support') or 'N/A'}")
        print(f"\n  Green flags:")
        for f in (r.get("green_flags") or []):
            print(f"    + {f}")
        print(f"\n  Red flags:")
        for f in (r.get("red_flags") or []):
            print(f"    ! {f}")
    else:
        print("  Not found in validated results")

    # Best pairings
    first = name.split()[0]
    player_pairs = pairs[
        pairs["player_a"].str.contains(first, case=False) |
        pairs["player_b"].str.contains(first, case=False)
    ].sort_values("compat_shrunk", ascending=False).head(10)

    print(f"\n  Best pairings (shrunk compat, min data):")
    print(f"  {'Partner':<26} {'Poss':>5}  {'Raw':>6}  {'Shrunk':>7}  {'Conf':>5}")
    print(f"  {'-'*26} {'-'*5}  {'-'*6}  {'-'*7}  {'-'*5}")
    for _, pr in player_pairs.iterrows():
        partner = pr["player_b"] if first.lower() in pr["player_a"].lower() else pr["player_a"]
        print(f"  {partner:<26} {pr['shared_poss']:>5.0f}  "
              f"{pr['compatibility']:>+6.2f}  {pr['compat_shrunk']:>+7.2f}  "
              f"{pr['confidence']:>4.0f}%")

    # Best lineups
    name_lineups = synergy[synergy["player_names"].apply(
        lambda names: any(first.lower() in n.lower() for n in names)
    )].sort_values("synergy_shrunk", ascending=False).head(5)

    print(f"\n  Best lineups by shrunk synergy:")
    print(f"  {'Players':<55} {'Poss':>5}  {'Shrunk':>7}  {'Conf':>5}")
    print(f"  {'-'*55} {'-'*5}  {'-'*7}  {'-'*5}")
    for _, lr in name_lineups.iterrows():
        players = ", ".join(lr["player_names"])
        if len(players) > 55: players = players[:52] + "..."
        print(f"  {players:<55} {lr['possessions']:>5.0f}  "
              f"{lr['synergy_shrunk']:>+7.2f}  {lr['confidence']:>4.0f}%")