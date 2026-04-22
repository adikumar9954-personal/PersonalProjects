# Racial Fairness in Disaster-Affected Credit Markets

An empirical study of whether Black homeowners in Florida received systematically fewer mortgage approvals than comparable white homeowners in areas affected by Hurricane Ian (2022), and whether FEMA disaster aid access explains part of that gap.

---

## What This Does (Plain English)

After a major hurricane, homeowners often need a mortgage refinance or home equity loan to repair their property. This project asks: **conditional on creditworthiness and neighborhood, do Black applicants face lower approval rates?** And if so, is part of the gap explained by unequal access to FEMA disaster grants?

The analysis uses three increasingly rigorous methods:

1. **SHAP decomposition** — breaks down the gap in approval rates into the portion explained by observable factors (income, loan size, credit score proxies) vs. the unexplained residual attributable to race.
2. **Double Machine Learning (DML)** — a causal inference method that isolates the direct effect of race on approval decisions after flexibly controlling for all observable confounders.
3. **Difference-in-Differences (DiD) event study** — compares Black vs. white applicants in hurricane-affected vs. unaffected Florida counties before and after Ian's landfall to estimate the causal effect of the disaster on the racial gap.

---

## Reports (Start Here)

Open either of these in a browser:

```
reports/
  writeup.html      Full research writeup with methodology, results, and interpretation
  explainer.html    Non-technical summary of findings
```

Key figures are in `figures/` — the most important ones:

```
figures/
  fig1_gap_decomposition.png    How much of the gap is "explained" vs. residual
  fig2_shap_distributions.png   SHAP value distributions by race
  fig3_direct_vs_proxy.png      Direct race effect vs. proxy variable effect
  fig5_dml_results.png          Causal estimate of the race coefficient
  fig6_event_study.png          DiD event study around Hurricane Ian
  fig8_ian_event_study.png      Ian-specific county-level event study
```

---

## Code Structure

Scripts are numbered in the order they should be run:

```
src/
  01_data_exploration.py    Initial look at HMDA mortgage data and distributions
  02_shap_race_analysis.py  SHAP decomposition of the racial approval gap
  03_data_pull.py           Pulls and merges HMDA + Census + FEMA data
  03b_fema_pull.py          FEMA disaster grant data pull
  03c_fema_fix.py           FEMA data cleaning and tract-level aggregation
  04_regression.py          OLS and logit baseline regressions
  05_dml.py                 Double Machine Learning causal estimation
  06_did_setup.py           DiD panel construction
  07_did_analysis.py        Main DiD estimation
  08_did_clean.py           Callaway-Sant'Anna staggered DiD
  09_ian_did.py             Hurricane Ian specific event study

data/
  fema_fl_tracts.parquet      FEMA grant amounts aggregated to census tracts
  fema_fl_tracts_full.parquet Full FEMA tract-level panel
  firm_panels_fl.csv          Merged HMDA + FEMA + Census panel for Florida
```

**Note:** The raw HMDA microdata (~270 MB each) and TIGER/Line shapefiles are excluded from this repository due to size — see `src/03_data_pull.py` for download instructions.

---

## Data Sources

| Source | What it contains |
|---|---|
| HMDA (Home Mortgage Disclosure Act) | Every mortgage application in the US — applicant race, income, loan amount, decision |
| FEMA Individuals & Households Program | Disaster grant payments by address, aggregated to census tract |
| ACS (American Community Survey) | Neighborhood-level demographics and income |
| TIGER/Line | Census tract shapefiles for Florida |
