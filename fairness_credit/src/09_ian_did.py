"""
Hurricane Ian DiD: Effect of a Sudden Climate Shock on Mortgage Pricing by Race

Hurricane Ian made landfall near Fort Myers, FL on September 28, 2022
as a Category 4 hurricane (peak 150mph winds). It is the costliest
Florida hurricane since Andrew (1992) and the deadliest since 1935.

Identification strategy:
  - Treatment: Counties in the direct path of Ian (FEMA DR-4673)
  - Control:   Florida coastal counties with similar pre-trends, not hit
  - Pre-period:  2018–2021 (HMDA years before Ian)
  - Post-period: 2022 (applications filed after Ian's landfall)
  - Key assumption: absent Ian, treated/control counties would have had
                    parallel trends in mortgage pricing. This is FAR more
                    plausible than the FEMA remapping assumption because:
                    (a) Ian was sudden and unpredictable
                    (b) Its geographic footprint was physically determined
                        by atmospheric dynamics, not economic conditions

Treatment intensity gradient (stronger identification):
  - Tier 1 (direct hit):  Lee, Charlotte, Sarasota — wind >100mph,
                          major storm surge, declared Individual Assistance
  - Tier 2 (heavy rain):  Collier, DeSoto, Polk, Hillsborough, Manatee
  - Control:              All other FL coastal counties in HMDA sample

Triple DiD for race:
  Y_ict = alpha_c + lambda_t + beta*(Treated_c × Post_t)
         + gamma*(Black_i × Treated_c × Post_t) + X_i'delta + eps

  gamma = differential pricing effect of Ian on Black vs. White applicants
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.linalg import lstsq, pinv
warnings.filterwarnings('ignore')

OUT = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════════════
# 1. TREATMENT GROUPS — FEMA DR-4673 (Hurricane Ian, Sept 28 2022)
# ══════════════════════════════════════════════════════════════════════════════
# Source: FEMA DR-4673 Individual Assistance + Public Assistance declarations
# Tier 1: Direct landfall corridor, IH declared, wind > 100mph at landfall
IAN_TIER1 = {
    '12071': 'Lee',        # Fort Myers — direct landfall, Category 4
    '12015': 'Charlotte',  # Port Charlotte — direct hit
    '12115': 'Sarasota',   # Sarasota — major wind/surge
}

# Tier 2: Heavy rainfall, inland flooding, IA declared
IAN_TIER2 = {
    '12021': 'Collier',
    '12027': 'DeSoto',
    '12049': 'Hardee',
    '12051': 'Hendry',
    '12055': 'Highlands',
    '12057': 'Hillsborough',
    '12081': 'Manatee',
    '12105': 'Polk',
}

# Control: Coastal FL counties NOT in Ian's path — comparable housing markets
# Chosen: Tampa Bay / Atlantic coast counties with active mortgage markets
CONTROL = {
    '12086': 'Miami-Dade',
    '12011': 'Broward',
    '12099': 'Palm Beach',
    '12031': 'Duval',       # Jacksonville
    '12103': 'Santa Rosa',
    '12033': 'Escambia',    # Pensacola
    '12113': 'Volusia',     # Daytona
    '12095': 'Orange',      # Orlando (inland but comparable)
    '12097': 'Osceola',
    '12069': 'Lake',
}

print("Hurricane Ian DiD Design")
print(f"  Tier 1 (direct hit):   {list(IAN_TIER1.values())}")
print(f"  Tier 2 (heavy impact): {list(IAN_TIER2.values())}")
print(f"  Control counties:      {list(CONTROL.values())}")

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

# Assign treatment tier
all_treated = {**{k: 1 for k in IAN_TIER1}, **{k: 2 for k in IAN_TIER2}}
hmda['tier'] = hmda['county_fips'].map(all_treated).fillna(0)
hmda.loc[hmda['county_fips'].isin(CONTROL), 'tier'] = -1  # mark controls

# Restrict sample
df = hmda[
    hmda['action_taken'].isin(['1','3']) &
    hmda['rate_spread'].notna() &
    hmda['rate_spread'].between(-2, 10) &
    hmda['race2'].isin(['White','Black or African American']) &
    (hmda['tier'] != 0)      # only treated or control counties
].copy()

df['treated']  = (df['tier'].isin([1, 2])).astype(float)
df['tier1']    = (df['tier'] == 1).astype(float)
df['post']     = (df['activity_year'] == 2022).astype(float)

df['treat_x_post']       = df['treated']  * df['post']
df['t1_x_post']          = df['tier1']    * df['post']
df['black_x_treat']      = df['is_black'] * df['treated']
df['black_x_post']       = df['is_black'] * df['post']
df['black_x_treat_post'] = df['is_black'] * df['treat_x_post']
df['black_x_t1_post']    = df['is_black'] * df['t1_x_post']

print(f"\nIan DiD sample: {len(df):,} loans")
print(f"  Tier 1 (direct hit): {(df['tier']==1).sum():,}")
print(f"  Tier 2 (heavy):      {(df['tier']==2).sum():,}")
print(f"  Control:             {(df['tier']==-1).sum():,}")
print(f"  Black share:         {df['is_black'].mean():.1%}")
print(f"  Pre (2018-2021):     {(df['post']==0).sum():,}")
print(f"  Post (2022):         {(df['post']==1).sum():,}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. PARALLEL TRENDS CHECK — Pre-period event study (2018-2021)
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- Parallel Trends Check: pre-period year-by-year ---")
pre_df = df[df['post'] == 0].copy()

pre_trends = {}
for race, is_b in [('White', 0), ('Black', 1)]:
    rows = []
    for yr in [2018, 2019, 2020, 2021]:
        tr = pre_df[(pre_df['activity_year']==yr) & (pre_df['treated']==1) & (pre_df['is_black']==is_b)]['rate_spread']
        ct = pre_df[(pre_df['activity_year']==yr) & (pre_df['treated']==0) & (pre_df['is_black']==is_b)]['rate_spread']
        if len(tr) > 50 and len(ct) > 50:
            diff = tr.mean() - ct.mean()
            se   = np.sqrt(tr.var()/len(tr) + ct.var()/len(ct))
            rows.append({'year': yr, 'diff': diff, 'se': se, 'n_tr': len(tr), 'n_ct': len(ct)})
    pre_trends[race] = pd.DataFrame(rows)

for race, pt in pre_trends.items():
    print(f"\n  {race} — treated minus control spread (should be ~stable):")
    print(pt[['year','diff','se','n_tr']].to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 4. SIMPLE 2x2 DiD TABLE (2021 pre vs. 2022 post)
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- Simple 2x2 DiD: Rate Spread (2021 vs 2022) ---")
df_2x2 = df[df['activity_year'].isin([2021, 2022])].copy()
df_2x2['post2'] = (df_2x2['activity_year'] == 2022).astype(float)

for race in ['White', 'Black or African American']:
    sub = df_2x2[df_2x2['race2'] == race]
    tbl = sub.groupby(['treated','post2'])['rate_spread'].mean().unstack()
    tbl.index = ['Control','Treated']
    tbl.columns = ['2021 (pre)','2022 (post)']
    did = (tbl.loc['Treated','2022 (post)'] - tbl.loc['Treated','2021 (pre)']) - \
          (tbl.loc['Control','2022 (post)'] - tbl.loc['Control','2021 (pre)'])
    print(f"\n  {race}:")
    print(tbl.round(4).to_string())
    print(f"  DiD estimate: {did*100:+.4f}pp")

# ══════════════════════════════════════════════════════════════════════════════
# 5. TRIPLE DiD TWFE: Y ~ treat×post × race + FE + controls
# ══════════════════════════════════════════════════════════════════════════════
def twfe_panel(df_in, outcome, treat_vars, controls, max_n=250_000):
    cols = [outcome] + treat_vars + controls + ['county_fips','activity_year']
    sub  = df_in[cols].dropna()
    if len(sub) > max_n:
        sub = sub.sample(max_n, random_state=42)

    # Within-demean by county
    for c in [outcome] + treat_vars + controls:
        sub[c] = sub[c] - sub.groupby('county_fips')[c].transform('mean')

    # Year dummies (relative to 2018)
    for yr in [2019, 2020, 2021, 2022]:
        col = f'yr_{yr}'
        sub[col] = (sub['activity_year'] == yr).astype(float)
        sub[col] -= sub.groupby('county_fips')[col].transform('mean')

    yr_cols   = [f'yr_{yr}' for yr in [2019,2020,2021,2022]]
    feat_cols = treat_vars + controls + yr_cols
    X = np.column_stack([np.ones(len(sub))] + [sub[c].values for c in feat_cols])
    y = sub[outcome].values

    # Drop zero/near-zero variance columns (but always keep intercept col 0)
    var       = X.var(axis=0)
    keep      = var > 1e-12
    keep[0]   = True          # always keep intercept
    X         = X[:, keep]
    labels    = ['const'] + [f for f, k in zip(feat_cols, keep[1:]) if k]

    b, _, _, _ = lstsq(X, y, rcond=None)
    resid = y - X @ b
    n, k  = X.shape
    meat  = (X * resid[:,None]).T @ (X * resid[:,None])
    bread = pinv(X.T @ X)
    vcov  = (n/(n-k)) * bread @ meat @ bread
    se    = np.sqrt(np.clip(np.diag(vcov), 0, None))
    t     = b / se
    sig   = ['***' if abs(ti)>3.29 else '**' if abs(ti)>2.58
             else '*' if abs(ti)>1.96 else '' for ti in t]
    return pd.DataFrame({'coef':b,'se':se,'t':t,'sig':sig}, index=['const']+labels[1:]), len(sub)

t_vars   = ['treat_x_post','t1_x_post','is_black',
            'black_x_treat','black_x_post','black_x_treat_post','black_x_t1_post']
controls = ['income','loan_amount','loan_to_value_ratio','dti_num']

print("\n" + "="*60)
print("TRIPLE DiD TWFE: Rate Spread")
print("="*60)
res_rs, n = twfe_panel(df, 'rate_spread', t_vars, controls)
print(f"N = {n:,}\n")
key = [r for r in t_vars if r in res_rs.index]
print(res_rs.loc[key][['coef','se','t','sig']].to_string())

print("\n" + "="*60)
print("TRIPLE DiD TWFE: Approval Rate")
print("="*60)
res_ap, n = twfe_panel(df, 'approved', t_vars, controls)
print(f"N = {n:,}\n")
print(res_ap.loc[key][['coef','se','t','sig']].to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 6. EVENT STUDY PLOT — all pre-years + post
# ══════════════════════════════════════════════════════════════════════════════
BG='#0f1117'; SURFACE='#1a1d27'; BORDER='#2e3350'; TEXT='#e2e8f0'; MUTED='#8892a4'
BLUE='#6c8eff'; RED='#ff7b6b'; GREEN='#52d9a0'; YELLOW='#f5c842'

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(BG)
fig.suptitle("Hurricane Ian DiD Event Study\n"
             "Annual rate spread: treated minus control, relative to 2021 baseline\n"
             "Dashed line = Ian landfall (Sept 28, 2022)",
             color=TEXT, fontsize=11)

for ax, (race, is_b, color) in zip(axes,
        [('White', 0, BLUE), ('Black or African American', 1, RED)]):
    pts = []
    for yr in [2018, 2019, 2020, 2021, 2022]:
        tr = df[(df['activity_year']==yr)&(df['treated']==1)&(df['is_black']==is_b)]['rate_spread']
        ct = df[(df['activity_year']==yr)&(df['treated']==0)&(df['is_black']==is_b)]['rate_spread']
        if len(tr) > 30 and len(ct) > 30:
            diff = tr.mean() - ct.mean()
            se   = np.sqrt(tr.var()/len(tr) + ct.var()/len(ct))
            pts.append({'year': yr, 'diff': diff, 'se': se})

    pts = pd.DataFrame(pts)
    # Normalize to 2021 baseline
    base = pts.loc[pts['year']==2021, 'diff'].values[0]
    pts['diff_adj'] = pts['diff'] - base

    ax.errorbar(pts['year'], pts['diff_adj']*100, yerr=1.96*pts['se']*100,
                marker='o', color=color, linewidth=2, markersize=8, capsize=5,
                label=race, zorder=3)
    ax.fill_between(pts['year'],
                    (pts['diff_adj'] - 1.96*pts['se'])*100,
                    (pts['diff_adj'] + 1.96*pts['se'])*100,
                    alpha=0.15, color=color)
    ax.axvline(2021.75, color=YELLOW, linestyle='--', linewidth=1.5,
               label='Hurricane Ian (Sept 2022)', alpha=0.8)
    ax.axhline(0, color=BORDER, linewidth=1)

    ax.set_xlabel('Year', color=TEXT)
    ax.set_ylabel('Rate spread gap change (pp,\ntreated minus control)', color=TEXT)
    ax.set_title(f'{race} applicants', color=TEXT, fontsize=11)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values(): spine.set_edgecolor(BORDER)
    ax.tick_params(colors=TEXT)
    ax.legend(framealpha=0.2, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    ax.set_xticks([2018,2019,2020,2021,2022])

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig8_ian_event_study.png'),
            dpi=150, bbox_inches='tight', facecolor=BG)
print("\nSaved fig8_ian_event_study.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7. TIER 1 ONLY — tightest identification (Lee + Charlotte + Sarasota)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TIER 1 ONLY (Lee + Charlotte + Sarasota): Tightest Design")
print("="*60)

df_t1 = df[df['tier'].isin([-1, 1])].copy()
df_t1['treat2'] = (df_t1['tier'] == 1).astype(float)
df_t1['treat2_x_post']       = df_t1['treat2']  * df_t1['post']
df_t1['black_x_treat2_post'] = df_t1['is_black'] * df_t1['treat2_x_post']
df_t1['black_x_treat2']      = df_t1['is_black'] * df_t1['treat2']
df_t1['black_x_post']        = df_t1['is_black'] * df_t1['post']

t_vars_t1 = ['treat2_x_post','is_black','black_x_treat2',
             'black_x_post','black_x_treat2_post']

res_t1, n = twfe_panel(df_t1, 'rate_spread', t_vars_t1, controls, max_n=150_000)
print(f"N = {n:,} (Tier 1 vs. control only)\n")
key_t1 = [r for r in t_vars_t1 if r in res_t1.index]
print(res_t1.loc[key_t1][['coef','se','t','sig']].to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 8. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
main_ian  = res_rs.loc['treat_x_post','coef']       if 'treat_x_post'       in res_rs.index else np.nan
race_diff = res_rs.loc['black_x_treat_post','coef'] if 'black_x_treat_post' in res_rs.index else np.nan
sig_main  = res_rs.loc['treat_x_post','sig']        if 'treat_x_post'       in res_rs.index else ''
sig_race  = res_rs.loc['black_x_treat_post','sig']  if 'black_x_treat_post' in res_rs.index else ''

t1_main   = res_t1.loc['treat2_x_post','coef']          if 'treat2_x_post'          in res_t1.index else np.nan
t1_race   = res_t1.loc['black_x_treat2_post','coef']    if 'black_x_treat2_post'    in res_t1.index else np.nan
t1_sig_m  = res_t1.loc['treat2_x_post','sig']           if 'treat2_x_post'          in res_t1.index else ''
t1_sig_r  = res_t1.loc['black_x_treat2_post','sig']     if 'black_x_treat2_post'    in res_t1.index else ''

print(f"""
{'='*60}
HURRICANE IAN DiD — FINAL RESULTS
{'='*60}

Specification            White effect     Black differential     Sig
---------------------------------------------------------------------------
All Ian counties         {main_ian*100:+.3f}pp       {race_diff*100:+.3f}pp              {sig_main} / {sig_race}
Tier 1 only (direct hit) {t1_main*100:+.3f}pp       {t1_race*100:+.3f}pp              {t1_sig_m} / {t1_sig_r}

Interpretation:
  After Hurricane Ian, mortgage rate spreads in affected counties
  {'rose' if main_ian > 0 else 'fell'} by {abs(main_ian)*100:.3f}pp for White applicants.
  Black applicants faced an {'additional' if race_diff > 0 else 'offsetting'}
  {abs(race_diff)*100:.3f}pp change ({sig_race}).

  The event study (fig8) shows whether pre-trends are parallel
  in 2018-2021 before Ian's landfall. Stable pre-period gaps
  support the causal interpretation.
""")
