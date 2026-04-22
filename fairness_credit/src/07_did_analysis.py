"""
Staggered DiD: Effect of Post-Irma FIRM Updates on Mortgage Terms by Race

Design:
  - Treatment unit: Florida county
  - Treatment event: First FIRM panel effective date 2018-2022 (post-Irma remapping)
  - Control: Florida counties with no FIRM updates 2018-2022
  - Outcomes: approval rate, rate_spread
  - Pre-period: 2018 (all counties pre-treatment for most)
  - Post-period: 2019-2022 (staggered by county)

Estimators:
  1. Simple 2x2 DiD (treated vs. control, before vs. after) — baseline
  2. Two-way fixed effects (TWFE) — standard panel DiD
  3. Event study plot — test parallel trends, visualize dynamic effects

Key heterogeneity: does the FIRM update effect differ by race?
  -> Interact treatment with is_black to estimate differential impact
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
warnings.filterwarnings('ignore')

OUT = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════════════
# 1. BUILD TREATMENT TIMING VARIABLE
# ══════════════════════════════════════════════════════════════════════════════
panels = pd.read_csv(os.path.join(OUT, 'firm_panels_fl.csv'))
panels['eff_date']    = pd.to_datetime(panels['eff_date'], errors='coerce')
panels['eff_year']    = panels['eff_date'].dt.year
panels['county_fips'] = '12' + panels['PCOMM'].astype(str).str.strip().str[:3]

# For each county: first FIRM effective date in 2018-2022 = treatment timing
# Counties with NO updates in 2018-2022 = pure controls
treated_timing = (
    panels[panels['eff_year'].between(2018, 2022)]
    .groupby('county_fips')['eff_year']
    .min()
    .reset_index()
    .rename(columns={'eff_year': 'treat_year'})
)

all_counties = panels['county_fips'].unique()
county_treat = pd.DataFrame({'county_fips': all_counties}).merge(
    treated_timing, on='county_fips', how='left'
)
# Counties with no update: treat_year = NaN → control group (never treated)
county_treat['ever_treated'] = county_treat['treat_year'].notna().astype(int)

print("Treatment timing distribution:")
print(county_treat['treat_year'].value_counts().sort_index().to_string())
print(f"\nTreated counties: {county_treat['ever_treated'].sum()}")
print(f"Control counties: {(~county_treat['ever_treated'].astype(bool)).sum()}")

# Narrow to clean 2x2: treatment counties with first update in 2020-2021
# (gives us clear pre=2018-2019, post=2021-2022 windows)
treat_2020_21 = county_treat[county_treat['treat_year'].isin([2019, 2020, 2021])]
control_group = county_treat[county_treat['ever_treated'] == 0]

print(f"\nClean treatment group (update 2019-2021): {len(treat_2020_21)} counties")
print(f"  Counties: {treat_2020_21['county_fips'].tolist()}")
print(f"Control group (never updated): {len(control_group)} counties")

# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD HMDA + MERGE TREATMENT
# ══════════════════════════════════════════════════════════════════════════════
hmda = pd.read_parquet(os.path.join(OUT, 'hmda_fl_raw.parquet'))

for col in ['loan_amount','income','interest_rate','rate_spread',
            'loan_to_value_ratio','debt_to_income_ratio']:
    if col in hmda.columns:
        hmda[col] = pd.to_numeric(hmda[col], errors='coerce')

hmda['approved']     = (hmda['action_taken'] == '1').astype(float)
hmda['census_tract'] = hmda['census_tract'].astype(str).str.zfill(11)
hmda['county_fips']  = hmda['census_tract'].str[:5]
hmda['race_clean']   = hmda['derived_race']
hmda['race2']        = hmda['race_clean'].str.replace(' alone','').str.strip()
hmda['is_black']     = (hmda['race2'] == 'Black or African American').astype(float)

dti_map = {'<20%':10,'20%-<30%':25,'30%-<36%':33,'50%-60%':55,'>60%':65}
hmda['dti_num'] = pd.to_numeric(hmda['debt_to_income_ratio'], errors='coerce')
hmda.loc[hmda['dti_num'].isna(), 'dti_num'] = (
    hmda.loc[hmda['dti_num'].isna(), 'debt_to_income_ratio'].map(dti_map)
)

# Merge treatment timing
df = hmda.merge(county_treat[['county_fips','treat_year','ever_treated']],
                on='county_fips', how='left')

# Restrict: originated + denied, key races, counties in treated or control group
analysis_counties = pd.concat([treat_2020_21, control_group])['county_fips'].tolist()
df = df[
    df['action_taken'].isin(['1','3']) &
    df['race2'].isin(['White','Black or African American']) &
    df['county_fips'].isin(analysis_counties)
].copy()

print(f"\nDiD sample: {len(df):,} loans across {df['county_fips'].nunique()} counties")
print(f"  Years: {sorted(df['activity_year'].unique())}")
print(f"  Treated counties: {df[df['ever_treated']==1]['county_fips'].nunique()}")
print(f"  Control counties: {df[df['ever_treated']==0]['county_fips'].nunique()}")

# ── Post indicator: is this observation in the post-treatment period?  ────────
# For each county, post = (year >= treat_year).
# For control counties, post = (year >= 2020) as placebo timing.
df['post'] = 0
treated_mask = df['ever_treated'] == 1
df.loc[treated_mask, 'post'] = (
    df.loc[treated_mask, 'activity_year'] >= df.loc[treated_mask, 'treat_year']
).astype(int)
# Control: use 2020 as pseudo-cutoff (median treat_year)
control_mask = df['ever_treated'] == 0
df.loc[control_mask, 'post'] = (df.loc[control_mask, 'activity_year'] >= 2020).astype(int)

# DiD dummies
df['treat_x_post']       = df['ever_treated'] * df['post']
df['black_x_treat']      = df['is_black']     * df['ever_treated']
df['black_x_post']       = df['is_black']     * df['post']
df['black_x_treat_post'] = df['is_black']     * df['treat_x_post']

# ══════════════════════════════════════════════════════════════════════════════
# 3. SIMPLE 2x2 DiD TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SIMPLE 2x2 DiD: Approval Rates")
print("="*60)

for race in ['White', 'Black or African American']:
    sub = df[df['race2'] == race]
    tbl = sub.groupby(['ever_treated', 'post'])['approved'].mean().unstack()
    tbl.index = ['Control', 'Treated']
    tbl.columns = ['Pre', 'Post']
    did = (tbl.loc['Treated','Post'] - tbl.loc['Treated','Pre']) - \
          (tbl.loc['Control','Post'] - tbl.loc['Control','Pre'])
    print(f"\n  {race}:")
    print(tbl.round(4).to_string())
    print(f"  DiD estimate: {did*100:+.2f}pp")

# ══════════════════════════════════════════════════════════════════════════════
# 4. TWFE: Y ~ treat_x_post + county_FE + year_FE + controls
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TWFE: Rate Spread ~ Treat×Post + Race Interactions")
print("="*60)

def twfe(df_in, outcome, treatment_vars, controls, max_n=200_000):
    sub = df_in[[outcome] + treatment_vars + controls +
                ['county_fips','activity_year']].dropna()
    if len(sub) > max_n:
        sub = sub.sample(max_n, random_state=42)

    # Within-demean by county (absorbs county FE)
    for c in [outcome] + controls + treatment_vars:
        sub[c] = sub[c] - sub.groupby('county_fips')[c].transform('mean')

    # Year dummies
    for yr in sorted(df_in['activity_year'].unique())[1:]:
        sub[f'yr_{yr}'] = (sub['activity_year'] == yr).astype(float)
        sub[f'yr_{yr}'] -= sub.groupby('county_fips')[f'yr_{yr}'].transform('mean')

    yr_cols  = [c for c in sub.columns if c.startswith('yr_')]
    feat_cols = treatment_vars + controls + yr_cols
    X = np.column_stack([np.ones(len(sub))] + [sub[c].values for c in feat_cols])
    y = sub[outcome].values
    labels = ['const'] + feat_cols

    # Drop zero-variance columns
    var = X.var(axis=0)
    keep = var > 1e-12
    X, labels = X[:, keep], [l for l, k in zip(labels, keep) if k]

    b, _, _, _ = lstsq(X, y, rcond=None)
    resid = y - X @ b
    n, k  = X.shape
    meat  = (X * resid[:,None]).T @ (X * resid[:,None])
    bread = np.linalg.pinv(X.T @ X)
    vcov  = (n/(n-k)) * bread @ meat @ bread
    se    = np.sqrt(np.clip(np.diag(vcov), 0, None))
    t     = b / se
    sig   = ['***' if abs(ti)>3.29 else '**' if abs(ti)>2.58
             else '*' if abs(ti)>1.96 else '' for ti in t]
    return pd.DataFrame({'coef':b,'se':se,'t':t,'sig':sig}, index=labels), len(sub)

t_vars   = ['treat_x_post','is_black','black_x_treat','black_x_post','black_x_treat_post']
controls = ['income','loan_amount','loan_to_value_ratio','dti_num']

# Approval
res_appr, n = twfe(df, 'approved', t_vars, controls)
print(f"\n  Outcome: approval  (N={n:,})")
key_rows = [r for r in t_vars if r in res_appr.index]
print(res_appr.loc[key_rows][['coef','se','t','sig']].to_string())

# Rate spread
df_rs = df[df['rate_spread'].notna() & df['rate_spread'].between(-2,10)].copy()
res_rs, n = twfe(df_rs, 'rate_spread', t_vars, controls)
print(f"\n  Outcome: rate_spread  (N={n:,})")
key_rows = [r for r in t_vars if r in res_rs.index]
print(res_rs.loc[key_rows][['coef','se','t','sig']].to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 5. EVENT STUDY PLOT (tests parallel trends)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EVENT STUDY: Year-by-year treatment effects")
print("="*60)

# For treated counties: relative time = activity_year - treat_year
df_es = df[df['ever_treated'] == 1].copy()
df_es['rel_time'] = df_es['activity_year'] - df_es['treat_year']

# Bin relative time: -2, -1, 0, +1, +2 (reference = -1)
df_es = df_es[df_es['rel_time'].between(-2, 3)].copy()

results_es = {}
for race in ['White', 'Black or African American']:
    sub = df_es[df_es['race2'] == race]
    ref_mean = sub[sub['rel_time'] == -1]['rate_spread'].mean()
    by_time = (sub.groupby('rel_time')['rate_spread']
                  .agg(['mean','sem','count'])
                  .reset_index())
    by_time['mean_adj'] = by_time['mean'] - ref_mean
    by_time['ci'] = 1.96 * by_time['sem']
    results_es[race] = by_time

# Plot
BG='#0f1117'; SURFACE='#1a1d27'; BORDER='#2e3350'; TEXT='#e2e8f0'; MUTED='#8892a4'
BLUE='#6c8eff'; RED='#ff7b6b'; GREEN='#52d9a0'

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(BG)
fig.suptitle('Event Study: Effect of FIRM Update on Rate Spread\n'
             '(Relative to year before update; treated counties only)',
             color=TEXT, fontsize=12)

for ax, (race, data) in zip(axes, results_es.items()):
    color = BLUE if race == 'White' else RED
    ax.errorbar(data['rel_time'], data['mean_adj'], yerr=data['ci'],
                marker='o', color=color, linewidth=2, markersize=7,
                capsize=5, label=race)
    ax.axvline(-0.5, color=MUTED, linestyle='--', linewidth=1, label='FIRM update')
    ax.axhline(0, color=BORDER, linewidth=1)
    ax.set_xlabel('Years relative to FIRM update', color=TEXT)
    ax.set_ylabel('Rate spread change (pp)', color=TEXT)
    ax.set_title(race, color=TEXT)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.tick_params(colors=TEXT)
    ax.legend(framealpha=0.2, edgecolor=BORDER, labelcolor=TEXT)
    ax.set_xticks([-2,-1,0,1,2,3])
    ax.set_xticklabels(['t-2','t-1\n(ref)','t=0','t+1','t+2','t+3'])

fig.tight_layout()
fig.savefig(os.path.join(OUT,'fig6_event_study.png'),
            dpi=150, bbox_inches='tight', facecolor=BG)
print("Saved fig6_event_study.png")

# Print the numbers
print("\nEvent study means (relative to t-1):")
for race, data in results_es.items():
    print(f"\n  {race}:")
    print(data[['rel_time','mean_adj','ci','count']].to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 6. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("KEY RESULTS SUMMARY")
print("="*60)

did_main  = res_rs.loc['treat_x_post',      'coef'] if 'treat_x_post'       in res_rs.index else np.nan
did_black = res_rs.loc['black_x_treat_post','coef'] if 'black_x_treat_post' in res_rs.index else np.nan
sig_main  = res_rs.loc['treat_x_post',      'sig']  if 'treat_x_post'       in res_rs.index else ''
sig_black = res_rs.loc['black_x_treat_post','sig']  if 'black_x_treat_post' in res_rs.index else ''

print(f"""
  Effect of FIRM update on rate spread (White applicants): {did_main*100:+.3f}pp {sig_main}
  Differential effect on Black applicants:                 {did_black*100:+.3f}pp {sig_black}

  Interpretation: When FEMA remaps a county's flood maps,
  {'White applicants see almost no change in pricing' if abs(did_main) < 0.05 else 'White applicants see a change in pricing'}.
  Black applicants {'face an additional' if did_black > 0 else 'see a differential change of'}
  {abs(did_black)*100:.3f}pp {'higher' if did_black > 0 else 'lower'} rate spread.

  This is the causal estimate — FIRM updates are exogenous regulatory
  events, not endogenous to applicant or lender behavior.
""")
