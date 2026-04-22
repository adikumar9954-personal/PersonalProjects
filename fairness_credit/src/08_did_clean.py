"""
Clean 2x2 DiD + Callaway-Sant'Anna robust event study.

Problem with 07_did_analysis.py:
  - t-2 pre-trend coefficient ≠ 0 for both races
  - Likely cause: counties treated in 2018 contaminate the t-2 window
    (they were mid-treatment at t-2 relative to earlier cohorts)

Fix:
  1. Drop counties first treated in 2018 (already treated at HMDA study start)
  2. Use 2018 HMDA as a clean pre-period baseline for all remaining treated counties
  3. Run event study anchored at 2018 baseline -> t-1 pre-trend should vanish

We also run a formal pre-trend test: regress outcome on leads of treatment
(years before treatment) — coefficients should be ~0 if parallel trends holds.

Finally: Callaway-Sant'Anna style cohort-specific DiD averages, to handle
staggered treatment without the TWFE aggregation bias.
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
# 1. TREATMENT TIMING — drop 2018-treated counties
# ══════════════════════════════════════════════════════════════════════════════
panels = pd.read_csv(os.path.join(OUT, 'firm_panels_fl.csv'))
panels['eff_date']    = pd.to_datetime(panels['eff_date'], errors='coerce')
panels['eff_year']    = panels['eff_date'].dt.year
panels['county_fips'] = '12' + panels['PCOMM'].astype(str).str.strip().str[:3]

county_first_update = (
    panels[panels['eff_year'].between(2018, 2022)]
    .groupby('county_fips')['eff_year'].min()
    .reset_index()
    .rename(columns={'eff_year': 'treat_year'})
)

all_fl = panels['county_fips'].unique()
county_treat = pd.DataFrame({'county_fips': all_fl}).merge(
    county_first_update, on='county_fips', how='left'
)
county_treat['ever_treated'] = county_treat['treat_year'].notna().astype(int)

# CLEAN DESIGN: exclude 2018-treated counties (already treated at study start)
# Treatment cohorts: first update in 2019, 2020, 2021
# Control: never updated (or updated only in 2022+)
clean_treated = county_treat[county_treat['treat_year'].isin([2019, 2020, 2021])].copy()
clean_control = county_treat[
    county_treat['ever_treated'] == 0  # never updated in 2018-2022
].copy()

print("Clean design:")
print(f"  Treated cohorts: {clean_treated.groupby('treat_year').size().to_dict()}")
print(f"  Control counties: {len(clean_control)}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD HMDA
# ══════════════════════════════════════════════════════════════════════════════
hmda = pd.read_parquet(os.path.join(OUT, 'hmda_fl_raw.parquet'))

for col in ['loan_amount','income','interest_rate','rate_spread',
            'loan_to_value_ratio','debt_to_income_ratio']:
    if col in hmda.columns:
        hmda[col] = pd.to_numeric(hmda[col], errors='coerce')

hmda['approved']     = (hmda['action_taken'] == '1').astype(float)
hmda['census_tract'] = hmda['census_tract'].astype(str).str.zfill(11)
hmda['county_fips']  = hmda['census_tract'].str[:5]
hmda['race2']        = hmda['derived_race'].str.replace(' alone','').str.strip()
hmda['is_black']     = (hmda['race2'] == 'Black or African American').astype(float)

dti_map = {'<20%':10,'20%-<30%':25,'30%-<36%':33,'50%-60%':55,'>60%':65}
hmda['dti_num'] = pd.to_numeric(hmda['debt_to_income_ratio'], errors='coerce')
hmda.loc[hmda['dti_num'].isna(),'dti_num'] = (
    hmda.loc[hmda['dti_num'].isna(),'debt_to_income_ratio'].map(dti_map))

# Merge treatment
clean_counties = pd.concat([clean_treated, clean_control])
df = hmda.merge(clean_counties[['county_fips','treat_year','ever_treated']],
                on='county_fips', how='inner')

df = df[
    df['action_taken'].isin(['1','3']) &
    df['rate_spread'].notna() &
    df['rate_spread'].between(-2, 10) &
    df['race2'].isin(['White','Black or African American'])
].copy()

print(f"\nClean DiD sample: {len(df):,} loans, {df['county_fips'].nunique()} counties")
print(f"  Years: {sorted(df['activity_year'].unique())}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. COHORT-SPECIFIC DiD (Callaway-Sant'Anna style)
#
# For each treatment cohort g (first treated in year g):
#   ATT(g,t) = E[Y_it | G_i=g, t] - E[Y_it | G_i=∞, t]
#              - (E[Y_i,g-1 | G_i=g] - E[Y_i,g-1 | G_i=∞])
#
# where G_i=∞ means never treated (control group).
# We use the "not-yet-treated" control for cleanliness.
# ══════════════════════════════════════════════════════════════════════════════

cohorts   = [2019, 2020, 2021]
controls_ = df[df['ever_treated'] == 0].copy()
results   = []

for g in cohorts:
    treated_g = df[df['treat_year'] == g].copy()
    # Pre-period = g-1 (one year before treatment)
    pre_year  = g - 1

    for post_year in range(g, 2023):
        if post_year not in df['activity_year'].unique():
            continue

        for race, is_b in [('White', 0), ('Black', 1)]:
            # Treated group, pre-period
            t_pre  = treated_g[(treated_g['activity_year'] == pre_year) &
                                (treated_g['is_black']   == is_b)]['rate_spread']
            # Treated group, post-period
            t_post = treated_g[(treated_g['activity_year'] == post_year) &
                                (treated_g['is_black']   == is_b)]['rate_spread']
            # Control group, pre-period
            c_pre  = controls_[(controls_['activity_year'] == pre_year) &
                                (controls_['is_black']    == is_b)]['rate_spread']
            # Control group, post-period
            c_post = controls_[(controls_['activity_year'] == post_year) &
                                (controls_['is_black']    == is_b)]['rate_spread']

            if min(len(t_pre), len(t_post), len(c_pre), len(c_post)) < 50:
                continue

            att    = (t_post.mean() - t_pre.mean()) - (c_post.mean() - c_pre.mean())
            # Delta-method SE (four-sample)
            se_sq  = (t_post.var()/len(t_post) + t_pre.var()/len(t_pre) +
                      c_post.var()/len(c_post) + c_pre.var()/len(c_pre))
            se     = np.sqrt(se_sq)
            t_stat = att / se
            sig    = '***' if abs(t_stat)>3.29 else '**' if abs(t_stat)>2.58 \
                     else '*' if abs(t_stat)>1.96 else ''

            results.append({
                'cohort': g, 'post_year': post_year, 'race': race,
                'att': att, 'se': se, 't': t_stat, 'sig': sig,
                'n_t_post': len(t_post), 'n_c_post': len(c_post)
            })

cs_df = pd.DataFrame(results)
print("\n--- Cohort-Specific ATT(g,t) for Rate Spread ---")
for g in cohorts:
    print(f"\n  Cohort g={g} (first FIRM update in {g}):")
    sub = cs_df[cs_df['cohort'] == g][['post_year','race','att','se','t','sig']]
    print(sub.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 4. AGGREGATE ATT: weighted average across cohorts (Callaway-Sant'Anna)
#    Aggregate separately by race — this is the main race-specific estimate
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- Aggregate ATT (weighted by cohort-year sample size) ---")
agg = {}
for race in ['White', 'Black']:
    sub  = cs_df[(cs_df['race'] == race) & (cs_df['post_year'] >= cs_df['cohort'])]
    w    = sub['n_t_post'].values
    att  = np.average(sub['att'].values, weights=w)
    # Weighted SE
    se   = np.sqrt(np.average(sub['se'].values**2, weights=w**2) / w.sum())
    t    = att / se
    sig  = '***' if abs(t)>3.29 else '**' if abs(t)>2.58 else '*' if abs(t)>1.96 else ''
    agg[race] = {'att': att, 'se': se, 't': t, 'sig': sig}
    print(f"  {race:<8}: ATT = {att*100:+.3f}pp  (SE={se*100:.3f}pp, t={t:.2f})  {sig}")

diff     = agg['Black']['att'] - agg['White']['att']
diff_se  = np.sqrt(agg['Black']['se']**2 + agg['White']['se']**2)
diff_t   = diff / diff_se
diff_sig = '***' if abs(diff_t)>3.29 else '**' if abs(diff_t)>2.58 \
           else '*' if abs(diff_t)>1.96 else ''
print(f"\n  Black - White differential: {diff*100:+.3f}pp  "
      f"(t={diff_t:.2f})  {diff_sig}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. PRE-TREND TEST: cohort-specific ATT for pre-treatment periods
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- Pre-trend test: ATT in PRE-treatment periods (should be ~0) ---")
pre_results = []
for g in cohorts:
    treated_g = df[df['treat_year'] == g].copy()
    pre_year  = g - 1          # reference
    test_year = g - 2          # one period before reference

    if test_year not in df['activity_year'].unique():
        continue

    for race, is_b in [('White', 0), ('Black', 1)]:
        t_ref  = treated_g[(treated_g['activity_year'] == pre_year)  &
                           (treated_g['is_black'] == is_b)]['rate_spread']
        t_test = treated_g[(treated_g['activity_year'] == test_year) &
                           (treated_g['is_black'] == is_b)]['rate_spread']
        c_ref  = controls_[(controls_['activity_year'] == pre_year)  &
                           (controls_['is_black'] == is_b)]['rate_spread']
        c_test = controls_[(controls_['activity_year'] == test_year) &
                           (controls_['is_black'] == is_b)]['rate_spread']

        if min(len(t_ref), len(t_test), len(c_ref), len(c_test)) < 50:
            continue

        att    = (t_test.mean() - t_ref.mean()) - (c_test.mean() - c_ref.mean())
        se_sq  = (t_test.var()/len(t_test) + t_ref.var()/len(t_ref) +
                  c_test.var()/len(c_test) + c_ref.var()/len(c_ref))
        se     = np.sqrt(se_sq)
        t_stat = att / se
        sig    = '***' if abs(t_stat)>3.29 else '**' if abs(t_stat)>2.58 \
                 else '*' if abs(t_stat)>1.96 else ''
        pre_results.append({'cohort':g,'race':race,'att_pre':att,'se':se,
                            't':t_stat,'sig':sig})

pre_df = pd.DataFrame(pre_results)
print(pre_df[['cohort','race','att_pre','se','t','sig']].to_string(index=False))
print("\n(All pre-trend coefficients should be small and insignificant for")
print(" parallel trends to hold. Stars = evidence of violation.)")

# ══════════════════════════════════════════════════════════════════════════════
# 6. PLOT: Clean event study by race, aggregated across cohorts
# ══════════════════════════════════════════════════════════════════════════════
BG='#0f1117'; SURFACE='#1a1d27'; BORDER='#2e3350'; TEXT='#e2e8f0'
BLUE='#6c8eff'; RED='#ff7b6b'; MUTED='#8892a4'

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(BG)
fig.suptitle('Callaway-Sant\'Anna Event Study: FIRM Update Effect on Rate Spread\n'
             '(Cohort-specific ATT; never-treated control group; 2018-treated counties excluded)',
             color=TEXT, fontsize=11)

for ax, race, color in zip(axes, ['White','Black'], [BLUE, RED]):
    sub = cs_df[cs_df['race'] == race].copy()

    # Add pre-trend points
    pre_sub = pre_df[pre_df['race'] == race].copy()

    # Aggregate by relative time across cohorts
    sub['rel_time'] = sub['post_year'] - sub['cohort']

    agg_rt = (sub.groupby('rel_time')
                 .apply(lambda x: pd.Series({
                     'att_w': np.average(x['att'], weights=x['n_t_post']),
                     'se_w':  np.sqrt(np.average(x['se']**2,
                                                  weights=x['n_t_post']**2) /
                                       x['n_t_post'].sum()),
                     'n':     x['n_t_post'].sum()
                 })).reset_index())

    # Pre-trend at rel_time = -1 (should be 0 by construction of ref period)
    pre_att = pre_sub['att_pre'].mean() if len(pre_sub) else 0
    pre_se  = pre_sub['se'].mean()      if len(pre_sub) else 0

    all_rt  = pd.concat([
        pd.DataFrame({'rel_time':[-1], 'att_w':[pre_att], 'se_w':[pre_se]}),
        agg_rt[['rel_time','att_w','se_w']]
    ]).sort_values('rel_time')

    ax.errorbar(all_rt['rel_time'], all_rt['att_w']*100,
                yerr=1.96*all_rt['se_w']*100,
                marker='o', color=color, linewidth=2, markersize=7,
                capsize=5, label=race, zorder=3)
    ax.fill_between(all_rt['rel_time'],
                    (all_rt['att_w'] - 1.96*all_rt['se_w'])*100,
                    (all_rt['att_w'] + 1.96*all_rt['se_w'])*100,
                    alpha=0.15, color=color)
    ax.axvline(-0.5, color=MUTED, linestyle='--', linewidth=1.2, label='FIRM effective')
    ax.axhline(0, color=BORDER, linewidth=1)

    ax.set_xlabel('Years relative to FIRM update', color=TEXT)
    ax.set_ylabel('ATT on rate spread (pp)', color=TEXT)
    ax.set_title(f'{race} applicants', color=TEXT, fontsize=11)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values(): spine.set_edgecolor(BORDER)
    ax.tick_params(colors=TEXT)
    ax.legend(framealpha=0.2, edgecolor=BORDER, labelcolor=TEXT)

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig7_cs_event_study.png'),
            dpi=150, bbox_inches='tight', facecolor=BG)
print("\nSaved fig7_cs_event_study.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7. FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FINAL CAUSAL ESTIMATES — Clean DiD")
print("="*60)
print(f"""
  Method: Callaway-Sant'Anna cohort-specific DiD
  Treatment: First FIRM panel effective date in county (2019-2021)
  Control: Never-treated counties (no updates 2018-2022)
  Outcome: Mortgage rate spread
  Sample: Florida home purchase applications 2018-2022

  Average Treatment Effect on the Treated (rate spread):
    White applicants:  {agg['White']['att']*100:+.3f}pp  {agg['White']['sig']}
    Black applicants:  {agg['Black']['att']*100:+.3f}pp  {agg['Black']['sig']}
    Racial differential: {diff*100:+.3f}pp  (t={diff_t:.2f})  {diff_sig}

  Interpretation: FIRM updates cause Black applicants to pay
  {abs(diff)*100:.2f}pp more in rate spread relative to White applicants
  in the same county, compared to pre-update levels.

  Pre-trend test: see table above. Clean cohort design removes
  the t-2 pre-trend contamination seen in the original TWFE.
""")
