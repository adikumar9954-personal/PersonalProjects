"""
Double/Debiased Machine Learning (DML) for Racial Discrimination in Mortgage Pricing

The question: what is the causal effect of race on rate spread, after
flexibly partialing out ALL observable creditworthiness factors using XGBoost?

This is the Chernozhukov et al. (2018) partially linear model:

    Y = theta * D + g(W) + epsilon      (outcome equation)
    D = m(W) + v                        (treatment equation)

where:
    Y = rate_spread
    D = is_black (treatment)
    W = {income, loan_amount, LTV, DTI, county, year, ...} (controls)
    theta = causal effect of race on spread (what we want)

The DML estimator:
    1. Fit XGBoost to predict Y from W  → residuals Y_tilde = Y - E[Y|W]
    2. Fit XGBoost to predict D from W  → residuals D_tilde = D - E[D|W]
    3. theta_DML = Cov(Y_tilde, D_tilde) / Var(D_tilde)
               = OLS of Y_tilde on D_tilde

Cross-fitting (K-fold) ensures the ML regularization bias doesn't contaminate theta.

We then split the sample by flood zone status to estimate:
    theta_outside_flood  (racial premium outside flood zone)
    theta_inside_flood   (racial premium inside flood zone)
    delta = theta_inside - theta_outside  (the climate-race interaction)
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')

OUT = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
hmda = pd.read_parquet(os.path.join(OUT, 'hmda_fl_raw.parquet'))

fema_path = (os.path.join(OUT, 'fema_fl_tracts_full.parquet')
             if os.path.exists(os.path.join(OUT, 'fema_fl_tracts_full.parquet'))
             else os.path.join(OUT, 'fema_fl_tracts.parquet'))
fema = pd.read_parquet(fema_path)

print(f"Using FEMA file: {os.path.basename(fema_path)}")

# ── Numeric conversions ───────────────────────────────────────────────────────
for col in ['loan_amount','income','interest_rate','rate_spread',
            'property_value','loan_to_value_ratio','debt_to_income_ratio']:
    if col in hmda.columns:
        hmda[col] = pd.to_numeric(hmda[col], errors='coerce')

hmda['approved']     = (hmda['action_taken'] == '1').astype(float)
hmda['census_tract'] = hmda['census_tract'].astype(str).str.zfill(11)
hmda['race_clean']   = hmda['derived_race']
hmda['county_code']  = hmda['census_tract'].str[:5]

tract_col = 'census_tract' if 'census_tract' in fema.columns else 'census_tract_geo'
fema = fema.rename(columns={tract_col: 'census_tract'})
fema['census_tract'] = fema['census_tract'].astype(str).str.zfill(11)

df = hmda.merge(fema[['census_tract','pct_in_sfha']], on='census_tract', how='left')
df['in_sfha']      = (df['pct_in_sfha'] > 5).astype(float)
df.loc[df['pct_in_sfha'].isna(), 'in_sfha'] = np.nan

# DTI numeric
dti_map = {'<20%':10,'20%-<30%':25,'30%-<36%':33,'50%-60%':55,'>60%':65}
df['dti_num'] = pd.to_numeric(df['debt_to_income_ratio'], errors='coerce')
mask = df['dti_num'].isna()
df.loc[mask, 'dti_num'] = df.loc[mask, 'debt_to_income_ratio'].map(dti_map)

# Restrict sample
df = df[
    df['action_taken'].isin(['1','3']) &
    df['in_sfha'].notna() &
    df['rate_spread'].notna() &
    df['rate_spread'].between(-2, 10) &
    df['race_clean'].isin(['White alone','Black or African American alone',
                           'White','Black or African American'])
].copy()
df['race2']    = df['race_clean'].str.replace(' alone','').str.strip()
df['is_black'] = (df['race2'] == 'Black or African American').astype(float)

print(f"DML sample: {len(df):,} loans")
print(f"  Black: {df['is_black'].sum():,.0f} ({df['is_black'].mean():.1%})")
print(f"  In SFHA: {df['in_sfha'].sum():,.0f} ({df['in_sfha'].mean():.1%})")
print(f"  Mean rate_spread: {df['rate_spread'].mean():.4f}")

# ── County dummies (top 20 counties by volume, rest = 'other') ───────────────
top_counties = df['county_code'].value_counts().head(20).index.tolist()
df['county_grp'] = df['county_code'].where(df['county_code'].isin(top_counties), 'other')
county_dummies = pd.get_dummies(df['county_grp'], prefix='cty', drop_first=True)
year_dummies   = pd.get_dummies(df['activity_year'], prefix='yr',  drop_first=True)

# ── Feature matrix W (controls — everything except race) ─────────────────────
W_cont = df[['income','loan_amount','loan_to_value_ratio','dti_num']].fillna(0)
W = pd.concat([W_cont, county_dummies, year_dummies], axis=1).astype(float).values

Y = df['rate_spread'].values
D = df['is_black'].values

print(f"\nControl matrix W shape: {W.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. DML ESTIMATOR (5-fold cross-fitting)
# ══════════════════════════════════════════════════════════════════════════════

def dml_estimate(Y, D, W, n_splits=5, subsample=150_000, label=''):
    """
    DML partially linear model with XGBoost nuisance functions.
    Returns theta (causal effect), SE, and diagnostics.
    """
    n = len(Y)
    if n > subsample:
        idx = np.random.default_rng(42).choice(n, subsample, replace=False)
        Y, D, W = Y[idx], D[idx], W[idx]
        n = subsample

    print(f"\n  [{label}] N={n:,}, running {n_splits}-fold DML...")

    Y_res = np.zeros(n)   # Y - E[Y|W]
    D_res = np.zeros(n)   # D - E[D|W]

    xgb_params = dict(n_estimators=200, max_depth=5, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8,
                      eval_metric='rmse', random_state=42, verbosity=0)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_r2_list, d_r2_list = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(W)):
        W_tr, W_te = W[train_idx], W[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]
        D_tr, D_te = D[train_idx], D[test_idx]

        # --- Outcome model: E[Y|W] ---
        m_y = xgb.XGBRegressor(**xgb_params)
        m_y.fit(W_tr, Y_tr)
        y_hat      = m_y.predict(W_te)
        Y_res[test_idx] = Y_te - y_hat
        y_r2_list.append(r2_score(Y_te, y_hat))

        # --- Treatment model: E[D|W] ---
        m_d = xgb.XGBClassifier(**xgb_params)
        m_d.fit(W_tr, D_tr)
        d_hat      = m_d.predict_proba(W_te)[:, 1]
        D_res[test_idx] = D_te - d_hat
        d_r2_list.append(r2_score(D_te, d_hat))

        print(f"    Fold {fold+1}: Y-R²={y_r2_list[-1]:.3f}, D-R²={d_r2_list[-1]:.3f}")

    # --- Final OLS of Y_res on D_res ---
    # theta = Cov(Y_res, D_res) / Var(D_res)
    theta = np.cov(Y_res, D_res)[0, 1] / np.var(D_res)

    # HC1 standard error
    n_obs = len(Y_res)
    resid = Y_res - theta * D_res
    score = D_res * resid
    V     = np.var(score) / (np.var(D_res) ** 2)
    se    = np.sqrt(V / n_obs)
    t     = theta / se

    print(f"\n  [{label}]")
    print(f"  Y nuisance R²:  {np.mean(y_r2_list):.4f}  (how well XGBoost predicts spread from controls)")
    print(f"  D nuisance R²:  {np.mean(d_r2_list):.4f}  (how well XGBoost predicts race from controls)")
    print(f"  theta (DML):    {theta*100:+.4f}pp  (SE={se*100:.4f}pp, t={t:.2f})")

    sig = '***' if abs(t) > 3.29 else '**' if abs(t) > 2.58 else '*' if abs(t) > 1.96 else ''
    print(f"  Significance:   {sig}")

    return {'theta': theta, 'se': se, 't': t, 'sig': sig,
            'y_r2': np.mean(y_r2_list), 'd_r2': np.mean(d_r2_list), 'n': n_obs}

# ══════════════════════════════════════════════════════════════════════════════
# 3. NAIVE OLS BASELINE (for comparison)
# ══════════════════════════════════════════════════════════════════════════════
from numpy.linalg import lstsq

def naive_ols_race(Y, D, W, label=''):
    """OLS of Y on [D, W] — biased if W incompletely captures credit quality."""
    idx = np.random.default_rng(42).choice(len(Y), min(150_000, len(Y)), replace=False)
    Ys, Ds, Ws = Y[idx], D[idx], W[idx]
    # Drop zero-variance columns (e.g. county dummies absent in small subsets)
    col_var = Ws.var(axis=0)
    Ws = Ws[:, col_var > 0]
    X = np.column_stack([Ds, Ws, np.ones(len(Ys))])
    b, _, _, _ = lstsq(X, Ys, rcond=None)
    resid = Ys - X @ b
    n, k  = X.shape
    meat  = (X * resid[:, None]).T @ (X * resid[:, None])
    bread = np.linalg.pinv(X.T @ X)   # pinv handles near-singular cases
    vcov  = (n/(n-k)) * bread @ meat @ bread
    se    = np.sqrt(vcov[0,0])
    theta = b[0]
    t     = theta / se
    sig   = '***' if abs(t)>3.29 else '**' if abs(t)>2.58 else '*' if abs(t)>1.96 else ''
    print(f"  [{label}] Naive OLS: theta={theta*100:+.4f}pp  SE={se*100:.4f}pp  t={t:.2f}  {sig}")
    return {'theta': theta, 'se': se, 't': t, 'sig': sig}

print("\n" + "="*60)
print("NAIVE OLS BASELINES (for comparison with DML)")
print("="*60)

# Full sample
naive_full    = naive_ols_race(Y, D, W, 'Full sample')
# Outside flood zone
mask_out      = df['in_sfha'].values == 0
naive_outside = naive_ols_race(Y[mask_out], D[mask_out], W[mask_out], 'Outside SFHA')
# Inside flood zone
mask_in       = df['in_sfha'].values == 1
naive_inside  = naive_ols_race(Y[mask_in],  D[mask_in],  W[mask_in],  'Inside SFHA')

# ══════════════════════════════════════════════════════════════════════════════
# 4. DML ESTIMATES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("DML ESTIMATES (XGBoost nuisance, 5-fold cross-fitting)")
print("="*60)

dml_full    = dml_estimate(Y, D, W, label='Full sample')

dml_outside = dml_estimate(Y[mask_out], D[mask_out], W[mask_out], label='Outside SFHA')

dml_inside  = dml_estimate(Y[mask_in],  D[mask_in],  W[mask_in],  label='Inside SFHA')

# Differential effect (inside - outside)
delta        = dml_inside['theta'] - dml_outside['theta']
delta_se     = np.sqrt(dml_inside['se']**2 + dml_outside['se']**2)
delta_t      = delta / delta_se
delta_sig    = '***' if abs(delta_t)>3.29 else '**' if abs(delta_t)>2.58 else '*' if abs(delta_t)>1.96 else ''

print("\n" + "="*60)
print("CLIMATE x RACE INTERACTION (DML)")
print("="*60)
print(f"  Racial spread premium outside flood zone: {dml_outside['theta']*100:+.4f}pp  {dml_outside['sig']}")
print(f"  Racial spread premium inside  flood zone: {dml_inside['theta']*100:+.4f}pp  {dml_inside['sig']}")
print(f"  Differential (inside - outside):          {delta*100:+.4f}pp  (t={delta_t:.2f})  {delta_sig}")
print()
print(f"  For comparison — naive OLS:")
print(f"    Outside flood: {naive_outside['theta']*100:+.4f}pp  {naive_outside['sig']}")
print(f"    Inside flood:  {naive_inside['theta']*100:+.4f}pp  {naive_inside['sig']}")
print(f"    Difference:    {(naive_inside['theta']-naive_outside['theta'])*100:+.4f}pp")

# ══════════════════════════════════════════════════════════════════════════════
# 5. VISUALIZE
# ══════════════════════════════════════════════════════════════════════════════
BG      = '#0f1117'; SURFACE = '#1a1d27'; BORDER = '#2e3350'
TEXT    = '#e2e8f0'; MUTED   = '#8892a4'
BLUE    = '#6c8eff'; RED     = '#ff7b6b'; GREEN   = '#52d9a0'; YELLOW  = '#f5c842'

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.patch.set_facecolor(BG)
fig.suptitle('Racial Spread Premium: Naive OLS vs. DML\n'
             '(Positive = Black applicants pay more; controlling for income, LTV, DTI, county, year)',
             color=TEXT, fontsize=12)

specs = [
    ('Full sample',      naive_full,    dml_full),
    ('Outside flood zone', naive_outside, dml_outside),
    ('Inside flood zone',  naive_inside,  dml_inside),
]

for ax, (title, naive, dml_r) in zip(axes, specs):
    labels   = ['Naive OLS', 'DML\n(XGBoost)']
    thetas   = [naive['theta']*100, dml_r['theta']*100]
    errors   = [naive['se']*100*1.96, dml_r['se']*100*1.96]
    colors   = [YELLOW, BLUE]
    sigs     = [naive['sig'], dml_r['sig']]

    bars = ax.bar(labels, thetas, color=colors, width=0.5, edgecolor='none', alpha=0.9)
    ax.errorbar(labels, thetas, yerr=errors, fmt='none',
                color=TEXT, capsize=6, linewidth=1.5)
    ax.axhline(0, color=BORDER, linewidth=1)

    for bar, val, sig in zip(bars, thetas, sigs):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + (errors[0 if bar == bars[0] else 1] + 0.3),
                f'{val:+.2f}pp\n{sig}',
                ha='center', va='bottom', color=TEXT, fontsize=9, fontweight='bold')

    ax.set_title(title, color=TEXT, fontsize=11)
    ax.set_ylabel('Racial spread premium (pp)', color=TEXT, fontsize=9)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.tick_params(colors=TEXT)

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig5_dml_results.png'),
            dpi=150, bbox_inches='tight', facecolor=BG)
print(f"\nSaved fig5_dml_results.png")
