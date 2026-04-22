# EONET Explorer — Natural Event Frequency Modeling

Statistical analysis of NASA's Earth Observatory Natural Event Tracker (EONET): fitting inhomogeneous Poisson process models to historical event counts to test whether natural disaster frequency is trending up, down, or staying flat.

---

## Overview (Start Here)

Open **`reports/eonet_overview.html`** in a browser for a plain-English guide to:
- What EONET is and where its data comes from
- What each event category contains (and what it doesn't)
- How much data exists and at what precision
- Known data quality issues (reporting artifacts, coverage gaps)
- Summary of modeling results

---

## Modeling Results

Three categories had enough clean data to model:

| Category | Verdict | Annual trend | Notes |
|---|---|---|---|
| **Severe Storms** | Stationary | ×0.99/yr | Both MK and GLM agree — no trend |
| **Wildfires** | Inconclusive | ×0.94/yr (GLM) | Non-parametric test disagrees; 13× overdispersion; 2024 data excluded |
| **Volcanoes** | Inconclusive | ×0.94/yr (GLM) | Non-parametric test marginally disagrees; only 7 years of data |
| **Floods** | Not enough data | — | GDACS source added May 2025; only 12 months usable |

Plots are in `plots/` — one per category showing the fitted rate, seasonal profile, and CV skill scores.

---

## Code Structure

```
eonet_client.py      Reusable EONET API v3 client (fetch, filter, flatten)
eonet_poisson.py     Full modeling pipeline:
                       1. Fetch events year-by-year (cached to cache/)
                       2. Aggregate to monthly counts
                       3. Fit GLM Poisson (seasonal + trend)
                       4. Mann-Kendall + GLM stationarity tests
                       5. Expanding-window time series CV
                       6. Generate plots

reports/             Human-readable outputs
  eonet_overview.html  Plain-English data guide and results summary

plots/               Generated model plots (one per category)
cache/               Raw API downloads — gitignored, regenerate with --refresh

EONET_REFERENCE.md       Full API reference (endpoints, parameters, all categories)
EONET_DATA_EXPLAINER.md  Detailed explanation of geometry, magnitudes, sources
```

To run the full pipeline:
```bash
python eonet_poisson.py                        # uses cached data
python eonet_poisson.py --refresh              # re-fetches from API
python eonet_poisson.py --no-plots             # skip plot generation
python eonet_poisson.py --category wildfires   # single category
```
