"""
Regression Analysis: Does flood zone status affect mortgage outcomes,
and does that effect differ by race?

Three models:
  1. OLS approval (linear probability model) — easy to interpret
  2. Rate spread regression — continuous outcome
  3. Interaction model — flood_zone x race

Controls: income, loan_amount, loan_to_value_ratio, debt_to_income_ratio,
          activity_year FE, county FE
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Try full coverage file, fall back to partial ──────────────────────────────
full_path = os.path.join(OUT, 'fema_fl_tracts_full.parquet')
part_path = os.path.join(OUT, 'fema_fl_tracts.parquet')

fema_path = full_path if os.path.exists(full_path) else part_path
print(f"Using FEMA data: {os.path.basename(fema_path)}")

# ── Load and merge ────────────────────────────────────────────────────────────
hmda = pd.read_parquet(os.path.join(OUT, 'hmda_fl_raw.parquet'))
fema = pd.read_parquet(fema_path)

for col in ['loan_amount','income','interest_rate','rate_spread',
            'property_value','loan_to_value_ratio','debt_to_income_ratio']:
    if col in hmda.columns:
        hmda[col] = pd.to_numeric(hmda[col], errors='coerce')

hmda['approved']      = (hmda['action_taken'] == '1').astype(float)
hmda['census_tract']  = hmda['census_tract'].astype(str).str.zfill(11)
hmda['race_clean']    = hmda['derived_race']
hmda['county_code']   = hmda['census_tract'].str[:5]   # state+county FIPS

# Normalize tract key in fema
tract_col = 'census_tract' if 'census_tract' in fema.columns else 'census_tract_geo'
fema = fema.rename(columns={tract_col: 'census_tract'})
fema['census_tract'] = fema['census_tract'].astype(str).str.zfill(11)

df = hmda.merge(fema[['census_tract','pct_in_sfha']], on='census_tract', how='left')
df['in_sfha'] = (df['pct_in_sfha'] > 5).astype(float)
df.loc[df['pct_in_sfha'].isna(), 'in_sfha'] = np.nan

# Analysis sample: originated + denied, flood data present, White + Black
df = df[
    df['action_taken'].isin(['1','3']) &
    df['in_sfha'].notna() &
    df['race_clean'].isin(['White alone','Black or African American alone',
                           'White','Black or African American'])
].copy()

df['race2']    = df['race_clean'].str.replace(' alone','').str.strip()
df['is_black'] = (df['race2'] == 'Black or African American').astype(float)

# DTI: CFPB encodes as string ranges, convert to midpoint
dti_map = {'<20%':10,'20%-<30%':25,'30%-<36%':33,'36':36,'37':37,'38':38,'39':39,
           '40%':40,'41':41,'42':42,'43':43,'44':44,'45':45,'46':46,'47':47,
           '48':48,'49':49,'50%-60%':55,'>60%':65,'Exempt':np.nan,'NA':np.nan}
if 'debt_to_income_ratio' in df.columns:
    df['dti_num'] = (pd.to_numeric(df['debt_to_income_ratio'], errors='coerce')
                     .fillna(df['debt_to_income_ratio'].map(dti_map)))
else:
    df['dti_num'] = np.nan

print(f"Analysis sample: {len(df):,} loans")
print(f"  Black: {df['is_black'].sum():,.0f} ({df['is_black'].mean():.1%})")
print(f"  In SFHA: {df['in_sfha'].sum():,.0f} ({df['in_sfha'].mean():.1%})")

# ══════════════════════════════════════════════════════════════════════════════
# SIMPLE OLS WITH MANUAL CONTROLS (no extra packages needed)
# We demean continuous controls and add year+county dummies manually.
# ══════════════════════════════════════════════════════════════════════════════
from numpy.linalg import lstsq

def run_ols(y, X, labels):
    """Basic OLS with heteroskedasticity-robust (HC1) standard errors."""
    n, k = X.shape
    b, _, _, _ = lstsq(X, y, rcond=None)
    resid = y - X @ b
    # HC1 sandwich
    meat  = (X * resid[:, None]).T @ (X * resid[:, None])
    bread = np.linalg.inv(X.T @ X)
    vcov  = (n / (n - k)) * bread @ meat @ bread
    se    = np.sqrt(np.diag(vcov))
    t     = b / se
    return pd.DataFrame({'coef': b, 'se': se, 't': t,
                         'sig': ['***' if abs(ti)>3.3 else '**' if abs(ti)>2.6
                                 else '*' if abs(ti)>1.96 else '' for ti in t]},
                        index=labels)

def prep_data(df, outcome, controls, max_n=200_000):
    """Build design matrix with year + county FE via within-group demeaning."""
    cols = [outcome] + controls + ['activity_year','county_code']
    sub  = df[cols].dropna().copy()
    if len(sub) > max_n:
        sub = sub.sample(max_n, random_state=42)

    # Within-demean by county (absorbs county FE cheaply)
    for c in [outcome] + controls:
        sub[c] -= sub.groupby('county_code')[c].transform('mean')

    # Year dummies (relative to 2018)
    for yr in [2019, 2020, 2021, 2022]:
        sub[f'yr_{yr}'] = (sub['activity_year'] == yr).astype(float)

    feat_cols = controls + [f'yr_{yr}' for yr in [2019,2020,2021,2022]]
    X = np.column_stack([np.ones(len(sub))] +
                        [sub[c].values for c in feat_cols])
    y = sub[outcome].values
    labels = ['const'] + feat_cols
    return y, X, labels, len(sub)

# ── Model 1: Approval ~ flood_zone + race + controls ─────────────────────────
print("\n" + "="*60)
print("MODEL 1: Linear Probability — Approval Rate")
print("="*60)

controls_1 = ['in_sfha','is_black','income','loan_amount',
              'loan_to_value_ratio','dti_num']
y1, X1, lab1, n1 = prep_data(df, 'approved', controls_1)
res1 = run_ols(y1, X1, lab1)
print(f"  N = {n1:,}")
print(res1[['coef','se','t','sig']].to_string())

# ── Model 2: Approval ~ flood x race interaction ──────────────────────────────
print("\n" + "="*60)
print("MODEL 2: Flood Zone x Race Interaction (Approval)")
print("="*60)

df['flood_x_black'] = df['in_sfha'] * df['is_black']
controls_2 = ['in_sfha','is_black','flood_x_black','income',
              'loan_amount','loan_to_value_ratio','dti_num']
y2, X2, lab2, n2 = prep_data(df, 'approved', controls_2)
res2 = run_ols(y2, X2, lab2)
print(f"  N = {n2:,}")
print(res2[['coef','se','t','sig']].to_string())

flood_main   = res2.loc['in_sfha',   'coef']
black_main   = res2.loc['is_black',  'coef']
interaction  = res2.loc['flood_x_black', 'coef']
flood_se     = res2.loc['in_sfha',   'se']
int_se       = res2.loc['flood_x_black','se']

print(f"\n  --- Key interpretation ---")
print(f"  White applicant, flood zone penalty:  {flood_main*100:+.2f}pp")
print(f"  Black applicant, flood zone penalty:  {(flood_main+interaction)*100:+.2f}pp")
print(f"  Differential penalty (interaction):   {interaction*100:+.2f}pp  {res2.loc['flood_x_black','sig']}")
print(f"  (Controlling for income, LTV, DTI, county FE, year FE)")

# ── Model 3: Rate spread ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("MODEL 3: Rate Spread ~ Flood Zone x Race")
print("="*60)

df_rs = df[df['rate_spread'].notna() & df['rate_spread'].between(-2, 10)].copy()
df_rs['flood_x_black'] = df_rs['in_sfha'] * df_rs['is_black']
controls_3 = ['in_sfha','is_black','flood_x_black','income',
              'loan_amount','loan_to_value_ratio','dti_num']
y3, X3, lab3, n3 = prep_data(df_rs, 'rate_spread', controls_3)
res3 = run_ols(y3, X3, lab3)
print(f"  N = {n3:,}")
print(res3[['coef','se','t','sig']].to_string())

rs_flood   = res3.loc['in_sfha',      'coef']
rs_black   = res3.loc['is_black',     'coef']
rs_int     = res3.loc['flood_x_black','coef']

print(f"\n  --- Key interpretation ---")
print(f"  Flood zone premium (White applicants):  {rs_flood*100:+.3f}pp spread  {res3.loc['in_sfha','sig']}")
print(f"  Black applicant premium (baseline):     {rs_black*100:+.3f}pp spread  {res3.loc['is_black','sig']}")
print(f"  Additional flood premium for Black:     {rs_int*100:+.3f}pp spread  {res3.loc['flood_x_black','sig']}")
print(f"  Total flood premium for Black:          {(rs_flood+rs_int)*100:+.3f}pp spread")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT: Coefficient plot for the interaction models
# ══════════════════════════════════════════════════════════════════════════════
BG      = '#0f1117'; SURFACE = '#1a1d27'; BORDER = '#2e3350'
TEXT    = '#e2e8f0'; MUTED   = '#8892a4'
BLUE    = '#6c8eff'; RED     = '#ff7b6b'; GREEN   = '#52d9a0'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(BG)
fig.suptitle('Flood Zone × Race Effects on Mortgage Outcomes\n'
             '(Controlling for income, LTV, DTI, county FE, year FE)',
             color=TEXT, fontsize=13)

def plot_coef(ax, result, rows, title, xlabel, color_pos=BLUE, color_neg=RED):
    sub = result.loc[rows]
    y   = np.arange(len(rows))
    ci  = 1.96 * sub['se'].values
    ax.barh(y, sub['coef'].values,
            color=[color_pos if c > 0 else color_neg for c in sub['coef']],
            height=0.5, edgecolor='none', alpha=0.85)
    ax.errorbar(sub['coef'].values, y, xerr=ci,
                fmt='none', color=TEXT, capsize=4, linewidth=1.2)
    ax.axvline(0, color=BORDER, linewidth=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels([r.replace('_',' ') for r in rows], color=TEXT, fontsize=10)
    ax.set_xlabel(xlabel, color=TEXT, fontsize=10)
    ax.set_title(title, color=TEXT, fontsize=11, pad=8)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    # Significance stars
    for i, (r, row) in enumerate(sub.iterrows()):
        if row['sig']:
            ax.text(row['coef'] + (ci[i] + 0.002), i, row['sig'],
                    va='center', color=TEXT, fontsize=9)

plot_coef(ax1, res2,
          ['in_sfha','is_black','flood_x_black'],
          'Approval Rate (pp)',
          'Coefficient (percentage points)',)

plot_coef(ax2, res3,
          ['in_sfha','is_black','flood_x_black'],
          'Rate Spread (pp)',
          'Coefficient (percentage points)')

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig4_regression_coefs.png'),
            dpi=150, bbox_inches='tight', facecolor=BG)
print(f"\nSaved fig4_regression_coefs.png")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY: The Flood Zone Racial Penalty")
print("="*60)
print(f"""
                        White applicant    Black applicant    Difference
Flood zone -> approval:   {flood_main*100:+.2f}pp          {(flood_main+interaction)*100:+.2f}pp          {interaction*100:+.2f}pp {res2.loc['flood_x_black','sig']}
Flood zone -> spread:     {rs_flood*100:+.3f}pp          {(rs_flood+rs_int)*100:+.3f}pp          {rs_int*100:+.3f}pp {res3.loc['flood_x_black','sig']}

These estimates control for income, loan amount, LTV, DTI,
county fixed effects, and year fixed effects.
""")
