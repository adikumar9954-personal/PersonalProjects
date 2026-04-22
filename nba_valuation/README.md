# NBA Player Valuation System v2

RAPM + tracking validation + lineup synergy. All three v1 weaknesses fixed.

## What's new in v2

| v1 weakness | v2 fix |
|---|---|
| Fragile PBP stint parser | pbpstats.com API — handles double-subs, ejections, OT |
| Raw +/- as prior target | RAPTOR (auto-downloaded) > LEBRON (local CSV) > raw +/- |
| No lineup context | Synergy decomposition, pairwise compatibility, optimal lineup builder |

## Structure

```
nba_valuation/
├── data/
│   ├── ingest.py          # Data + caching
│   └── stint_matrix.py    # Design matrix (player + lineup level)
├── models/
│   ├── rapm.py            # Prior model + ridge RAPM
│   ├── validator.py       # Tracking validation scorer
│   └── lineup_synergy.py  # Synergy + pairwise compat + lineup builder
├── output/
│   └── report.py          # Reports, screener, full pipeline
└── README.md
```

## Install

```bash
pip install nba_api pandas numpy scipy scikit-learn pyarrow requests
```

## Quickstart

```python
from output.report import run_full_pipeline, player_report

out = run_full_pipeline(season="2023-24", max_games=200)

# Player report with best pairings
print(player_report("Nikola Jokic", out["validated"], out["pairs"]))

# Who fits best with a player?
from models.lineup_synergy import compatibility_for_player
print(compatibility_for_player("Stephen Curry", out["pairs"], top_n=10))

# Best lineup from a roster
from models.lineup_synergy import find_best_lineup
ids = ["201939", "203110", "1629029", "203648", "1628384"]
print(find_best_lineup(ids, out["pairs"], out["rapm"]))

# Full synergy report
from models.lineup_synergy import print_synergy_report
print_synergy_report("Draymond Green", out["rapm"], out["synergy"], out["pairs"])
```

## Optional: LEBRON prior

1. Download from https://www.basketball-reference.com/friv/bpm2.fcgi
2. Save as `data/cache/lebron_2024.csv` with columns: `player_id, player_name, lebron`
3. Pipeline auto-detects and uses it over RAPTOR

## How lineup synergy works

**Synergy delta**: `actual_lineup_net_rating - sum(individual_RAPMs)`
Positive = superadditive (players elevate each other). Negative = subadditive (overlap/conflict).

**Pairwise compatibility**: `pair_net_rating - (rapm_a + rapm_b)`
The most actionable number for roster construction.

**Lineup optimizer**: evaluates all C(N,5) combinations from a roster,
scores by `sum(individual_rapms) + 0.5 * sum(pairwise_compat)`.

## Limitations

- Pairwise compat is noisy for pairs with < 100 shared possessions.
- Lineup optimizer is combinatorial — works for rosters up to ~15 players.
- Synergy effects are within-season; mid-season trades split the sample.

## Next modules to build

- `models/salary.py` — $/RAPM cap efficiency
- `models/causal.py` — DoWhy causal graph to separate mechanism from confounding  
- `models/projection.py` — age curves + multi-season stacking for contract valuation
