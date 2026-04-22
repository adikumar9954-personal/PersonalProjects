# NBA Player Valuation & Lineup Synergy

A statistical model that estimates how much each NBA player contributes to winning — and how well specific combinations of players work together.

---

## What This Does (Plain English)

Standard NBA stats (points, rebounds, assists) tell you what a player *did*, not how much they *helped their team win*. This project builds a more rigorous picture using three components:

1. **RAPM (Regularized Adjusted Plus/Minus)** — estimates each player's net impact on points scored vs. allowed per 100 possessions while on the court, controlling for the quality of teammates and opponents faced.
2. **Lineup Synergy** — detects which *combinations* of players produce results better or worse than their individual ratings would predict (e.g., two good players who don't fit together).
3. **HTML Reports** — presents everything in a readable, shareable report for any team.

---

## Reports (Start Here)

Open any of these files in a browser:

```
reports/
  report_2025-26.html           Full league player valuations, current season
  nba_report.html               League-wide screener
  hornets_report.html           Charlotte Hornets breakdown
  lakers_report.html            Los Angeles Lakers breakdown
  player_valuations.csv         Raw valuation numbers for all players
  pairwise_compatibility.csv    Two-player lineup synergy scores
```

---

## Code Structure

```
data/           Data fetching and feature engineering
  ingest.py       Pulls from NBA Stats API and public rating feeds (DARKO, RAPTOR)
  date_split.py   Splits season data by date for rolling estimates
  playmaking.py   Constructs creation and playmaking metrics
  stint_matrix.py Builds the stint matrix used for RAPM estimation

models/         Statistical models
  rapm.py         Ridge regression RAPM
  lineup_synergy.py  Lineup interaction model and pairwise compatibility scorer
  validator.py    Cross-validation and sanity checks

output/         Report generation scripts (write results to reports/)
  report.py       Main pipeline — runs all models and assembles output
  html_report.py  HTML templating
  team_report.py  Per-team report generation

run.py          Entry point — fetches data, fits models, writes report
tests/          Development spot-checks (specific player comparisons, edge cases)
```

To regenerate all reports: `python run.py`

---

## How Lineup Synergy Works

**Synergy delta**: `actual_lineup_net_rating − sum(individual_RAPMs)`
Positive = players elevate each other. Negative = overlap or conflict.

**Pairwise compatibility**: `pair_net_rating − (rapm_A + rapm_B)`
The most actionable number for roster construction decisions.

**Lineup optimizer**: scores all 5-player combinations from a roster using
`sum(individual_RAPMs) + 0.5 × sum(pairwise_compatibility)`.

---

## Install

```bash
pip install nba_api pandas numpy scipy scikit-learn pyarrow requests
```
