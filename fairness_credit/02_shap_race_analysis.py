"""
Proxy Discrimination Analysis  (5-Fold CV edition)
-------------------------------
Which features carry race signal into the model's predictions?

Two complementary approaches:
  1. SHAP values by group — does the model use a feature differently for
     White vs. Black individuals?
  2. Decomposition of the prediction gap — how much of the White–Black
     approval gap can each feature "explain"?

Key change: all fairness metrics are estimated via 5-fold stratified
cross-validation. We report mean ± std across folds so that findings
are not sensitive to a single random split.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import xgboost as xgb
from folktables import ACSDataSource, ACSIncome
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os, warnings
warnings.filterwarnings('ignore')

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Palette ───────────────────────────────────────────────────────────────────
WHITE_COLOR = '#6c8eff'
BLACK_COLOR = '#ff7b6b'
NEUTRAL     = '#52d9a0'
BG          = '#0f1117'
SURFACE     = '#1a1d27'
SURFACE2    = '#21263a'
BORDER      = '#2e3350'
TEXT        = '#e2e8f0'
MUTED       = '#8892a4'

plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    SURFACE,
    'axes.edgecolor':    BORDER,
    'axes.labelcolor':   TEXT,
    'xtick.color':       MUTED,
    'ytick.color':       MUTED,
    'text.color':        TEXT,
    'grid.color':        BORDER,
    'grid.linestyle':    '--',
    'grid.alpha':        0.5,
    'font.family':       'sans-serif',
})

FEATURE_LABELS = {
    'AGEP': 'Age',
    'COW':  'Class of worker',
    'SCHL': 'Education',
    'MAR':  'Marital status',
    'OCCP': 'Occupation',
    'POBP': 'Place of birth',
    'RELP': 'Relationship',
    'WKHP': 'Hours/week',
    'SEX':  'Sex',
    'RAC1P':'Race',
}

N_FOLDS = 5

# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data...")
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data    = data_source.get_data(states=["CA"], download=True)
features, labels, group = ACSIncome.df_to_numpy(acs_data)

feature_names = ACSIncome.features
df = pd.DataFrame(features, columns=feature_names)
df['income_over_50k'] = labels.astype(int)
df['race'] = group

mask_wb = df['race'].isin([1, 2])

print(f"  Total records:  {len(df):,}")
print(f"  White records:  {(df['race']==1).sum():,}")
print(f"  Black records:  {(df['race']==2).sum():,}")
print(f"  White base rate: {df[df['race']==1]['income_over_50k'].mean():.3f}")
print(f"  Black base rate: {df[df['race']==2]['income_over_50k'].mean():.3f}")

X_all = df[feature_names].values
y_all = df['income_over_50k'].values

# Class imbalance: weight the minority class so positive:negative = 1:1 effective
neg, pos = np.bincount(y_all)
scale_pos_weight = neg / pos
print(f"\n  Class balance — negative: {neg:,}  positive: {pos:,}  "
      f"scale_pos_weight: {scale_pos_weight:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  5-FOLD STRATIFIED CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nRunning {N_FOLDS}-fold stratified CV (SHAP computed per fold)...")

skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
skf_val = StratifiedKFold(n_splits=10,      shuffle=True, random_state=0)

# Storage: one entry per fold
fold_aucs        = []
fold_n_trees     = []   # actual trees used after early stopping
fold_total_gaps  = []
fold_proxy_pcts  = []
fold_gap_dfs     = []   # per-fold gap_df (feature-level decomposition)
fold_shap_white  = []   # list of DataFrames (one per fold)
fold_shap_black  = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all), 1):
    print(f"\n  Fold {fold}/{N_FOLDS}")

    X_train_full, X_test = X_all[train_idx], X_all[test_idx]
    y_train_full, y_test = y_all[train_idx], y_all[test_idx]

    # Inner split: 10% of training fold used ONLY for early-stopping signal.
    # This keeps the test fold completely untouched during training.
    inner_train_idx, inner_val_idx = next(
        skf_val.split(X_train_full, y_train_full)
    )
    X_train = X_train_full[inner_train_idx]
    y_train = y_train_full[inner_train_idx]
    X_val   = X_train_full[inner_val_idx]
    y_val   = y_train_full[inner_val_idx]

    model = xgb.XGBClassifier(
        n_estimators=1000,        # ceiling; early stopping will find the right n
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,   # handles class imbalance
        eval_metric='logloss',
        early_stopping_rounds=20,            # stop if no improvement for 20 rounds
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],           # early stopping on inner val only
        verbose=False,
    )
    n_trees = model.best_iteration + 1
    fold_n_trees.append(n_trees)
    print(f"    Early stopping at {n_trees} trees")

    test_probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_probs)
    fold_aucs.append(auc)
    print(f"    AUC: {auc:.4f}")

    # Restrict test fold to White + Black
    test_df = df.iloc[test_idx].copy()
    test_df['prob'] = test_probs
    wb_test = test_df[test_df['race'].isin([1, 2])].copy()

    if wb_test['race'].nunique() < 2:
        print("    Skipping fold — only one race group in test set")
        continue

    X_wb_test = wb_test[feature_names].values

    # SHAP
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_wb_test)

    shap_df = pd.DataFrame(shap_values, columns=feature_names, index=wb_test.index)
    shap_df['race'] = wb_test['race'].values

    w_shap = shap_df[shap_df['race'] == 1][feature_names]
    b_shap = shap_df[shap_df['race'] == 2][feature_names]

    fold_shap_white.append(w_shap)
    fold_shap_black.append(b_shap)

    # Gap decomposition
    mean_shap_white = w_shap.mean()
    mean_shap_black = b_shap.mean()
    shap_gap        = mean_shap_white - mean_shap_black

    total_gap = (wb_test[wb_test['race']==1]['prob'].mean()
               - wb_test[wb_test['race']==2]['prob'].mean())
    fold_total_gaps.append(total_gap)

    gap_df = pd.DataFrame({
        'feature':      feature_names,
        'label':        [FEATURE_LABELS[f] for f in feature_names],
        'gap':          shap_gap.values,
        'pct_of_total': shap_gap.values / total_gap * 100,
    })
    fold_gap_dfs.append(gap_df)

    race_direct_gap = shap_gap['RAC1P']
    proxy_gap       = shap_gap.drop('RAC1P').sum()
    proxy_pct       = proxy_gap / total_gap * 100
    fold_proxy_pcts.append(proxy_pct)

    print(f"    Total gap: {total_gap:.4f}  |  Proxy gap: {proxy_pct:.1f}%  |  "
          f"Direct race: {race_direct_gap/total_gap*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  AGGREGATE ACROSS FOLDS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("CROSS-VALIDATION SUMMARY")
print("="*60)

auc_mean, auc_std           = np.mean(fold_aucs), np.std(fold_aucs)
gap_mean, gap_std           = np.mean(fold_total_gaps), np.std(fold_total_gaps)
proxy_mean, proxy_std       = np.mean(fold_proxy_pcts), np.std(fold_proxy_pcts)
trees_mean, trees_std       = np.mean(fold_n_trees), np.std(fold_n_trees)

print(f"  Test AUC:          {auc_mean:.4f} +/- {auc_std:.4f}")
print(f"  Trees (ES):        {trees_mean:.0f} +/- {trees_std:.0f}")
print(f"  White-Black gap:   {gap_mean:.4f} +/- {gap_std:.4f}")
print(f"  Proxy gap %%:       {proxy_mean:.1f}%% +/- {proxy_std:.1f}%%")

# Per-feature: mean and std of gap and pct_of_total across folds
all_gaps_df = pd.concat(fold_gap_dfs)
feat_stats = (all_gaps_df
    .groupby(['feature', 'label'])
    .agg(
        gap_mean    = ('gap',          'mean'),
        gap_std     = ('gap',          'std'),
        pct_mean    = ('pct_of_total', 'mean'),
        pct_std     = ('pct_of_total', 'std'),
    )
    .reset_index()
    .sort_values('gap_mean', ascending=False)
)

print(f"\n  {'Feature':<22} {'Mean gap':>10} {'Std':>8} {'Mean %':>10} {'Std %':>8}")
print("  " + "-"*62)
for _, row in feat_stats.iterrows():
    direction = "-> White" if row['gap_mean'] > 0 else "-> Black"
    print(f"  {row['label']:<22} {row['gap_mean']:>10.4f} {row['gap_std']:>8.4f} "
          f"{row['pct_mean']:>9.1f}% {row['pct_std']:>7.1f}%  {direction}")

# Pooled SHAP for distribution plots (all test-fold observations concatenated)
pooled_white = pd.concat(fold_shap_white)
pooled_black = pd.concat(fold_shap_black)

# ══════════════════════════════════════════════════════════════════════════════
# 4.  VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating plots...")

# ── Fig 1: Gap decomposition with error bars ──────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(11, 6))
fig1.patch.set_facecolor(BG)

feat_plot = feat_stats.sort_values('gap_mean', ascending=True)
colors = [WHITE_COLOR if g > 0 else BLACK_COLOR for g in feat_plot['gap_mean']]

bars = ax1.barh(feat_plot['label'], feat_plot['gap_mean'],
                xerr=feat_plot['gap_std'],
                color=colors, edgecolor='none', height=0.6,
                error_kw=dict(ecolor=MUTED, capsize=4, capthick=1.5, elinewidth=1.5))

for bar, val, err in zip(bars, feat_plot['gap_mean'], feat_plot['gap_std']):
    x = bar.get_width()
    offset = err + 0.0005 if x >= 0 else -(err + 0.0005)
    ax1.text(x + offset,
             bar.get_y() + bar.get_height()/2,
             f'{val:+.4f}',
             va='center', ha='left' if x >= 0 else 'right',
             color=TEXT, fontsize=9)

ax1.axvline(0, color=BORDER, linewidth=1.5)
ax1.set_xlabel('Mean SHAP difference (White - Black)  |  error bars = 1 SD across 5 folds',
               color=TEXT, fontsize=10)
ax1.set_title(f'Which features favor White applicants? ({N_FOLDS}-fold CV)',
              color=TEXT, fontsize=13, pad=15)

white_patch = mpatches.Patch(color=WHITE_COLOR, label='Favors White applicants')
black_patch = mpatches.Patch(color=BLACK_COLOR, label='Favors Black applicants')
ax1.legend(handles=[white_patch, black_patch], loc='lower right',
           framealpha=0.2, edgecolor=BORDER)
ax1.set_facecolor(SURFACE)
fig1.tight_layout()
fig1.savefig(os.path.join(OUT, 'fig1_gap_decomposition.png'), dpi=150,
             bbox_inches='tight', facecolor=BG)
print("  Saved fig1_gap_decomposition.png")

# ── Fig 2: SHAP distributions by race (pooled across folds) ──────────────────
top_features = (feat_stats.assign(abs_gap=feat_stats['gap_mean'].abs())
                           .nlargest(4, 'abs_gap')['feature'].tolist())

fig2, axes = plt.subplots(2, 2, figsize=(12, 8))
fig2.patch.set_facecolor(BG)
fig2.suptitle(f'SHAP distributions by race — top 4 proxy features (pooled {N_FOLDS}-fold)',
              color=TEXT, fontsize=13, y=1.01)

for ax, feat in zip(axes.flat, top_features):
    w_vals = pooled_white[feat].values
    b_vals = pooled_black[feat].values

    bins = np.linspace(min(w_vals.min(), b_vals.min()),
                       max(w_vals.max(), b_vals.max()), 50)

    ax.hist(w_vals, bins=bins, alpha=0.65, color=WHITE_COLOR,
            label=f'White (mu={w_vals.mean():.3f})', density=True)
    ax.hist(b_vals, bins=bins, alpha=0.65, color=BLACK_COLOR,
            label=f'Black (mu={b_vals.mean():.3f})', density=True)

    ax.axvline(w_vals.mean(), color=WHITE_COLOR, linestyle='--', linewidth=1.5)
    ax.axvline(b_vals.mean(), color=BLACK_COLOR, linestyle='--', linewidth=1.5)

    ax.set_title(FEATURE_LABELS[feat], color=TEXT, fontsize=11)
    ax.set_xlabel('SHAP value (impact on log-odds of income > $50k)', color=MUTED, fontsize=8)
    ax.legend(fontsize=8, framealpha=0.2, edgecolor=BORDER)
    ax.set_facecolor(SURFACE)

fig2.tight_layout()
fig2.savefig(os.path.join(OUT, 'fig2_shap_distributions.png'), dpi=150,
             bbox_inches='tight', facecolor=BG)
print("  Saved fig2_shap_distributions.png")

# ── Fig 3: Direct vs. proxy gap summary with CV uncertainty ──────────────────
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
fig3.patch.set_facecolor(BG)

# Left: pie of mean gaps (positive contributors only)
pos_feats    = feat_stats[feat_stats['gap_mean'] > 0]
direct_row   = feat_stats[feat_stats['feature'] == 'RAC1P'].iloc[0]
proxy_feats  = pos_feats[pos_feats['feature'] != 'RAC1P']

labels_pie = ['Race (direct)'] + proxy_feats['label'].tolist()
sizes_pie  = [abs(direct_row['gap_mean'])] + proxy_feats['gap_mean'].tolist()
wedge_colors = [WHITE_COLOR, NEUTRAL, '#f5c842', '#a78bfa', '#ff7b6b']

wedges, texts, autotexts = ax3a.pie(
    sizes_pie, labels=labels_pie, autopct='%1.1f%%',
    colors=wedge_colors[:len(sizes_pie)],
    startangle=90, textprops={'color': TEXT, 'fontsize': 9}
)
for at in autotexts:
    at.set_color(BG)
    at.set_fontweight('bold')
ax3a.set_title('What drives the prediction gap?\n(mean across folds)', color=TEXT, fontsize=12, pad=10)
ax3a.set_facecolor(BG)

# Right: full model vs. no-race model gap, with CV error bars
# Compute per-fold no-race gap from pooled shap is tricky; compute per-fold
fold_no_race_gaps = []
for gdf, total in zip(fold_gap_dfs, fold_total_gaps):
    proxy_sum = gdf[gdf['feature'] != 'RAC1P']['gap'].sum()
    # no-race gap = proxy_sum alone (race SHAP removed)
    # approximate: gap_no_race ~ total_gap - direct_race_gap = proxy_sum
    fold_no_race_gaps.append(proxy_sum)

nr_mean = np.mean(fold_no_race_gaps)
nr_std  = np.std(fold_no_race_gaps)

categories = ['Full model\n(race included)', 'Model without\ndirect race signal']
values     = [gap_mean, nr_mean]
errs       = [gap_std,  nr_std]
bar_colors = [WHITE_COLOR, NEUTRAL]

bars3 = ax3b.bar(categories, values, color=bar_colors, width=0.4, edgecolor='none',
                 yerr=errs, capsize=6,
                 error_kw=dict(ecolor=MUTED, capthick=1.5, elinewidth=1.5))
for bar, val in zip(bars3, values):
    ax3b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(errs)*1.2 + 0.001,
              f'{val:.4f}', ha='center', va='bottom', color=TEXT, fontsize=11, fontweight='bold')

ax3b.set_ylabel('White - Black mean predicted probability', color=TEXT)
ax3b.set_title(f'Removing race: gap remaining\n(mean +/- SD, {N_FOLDS}-fold CV)', color=TEXT, fontsize=12)
ax3b.set_facecolor(SURFACE)
ax3b.set_ylim(0, max(values) * 1.35)

pct_remaining = nr_mean / gap_mean * 100
ax3b.annotate(f'{pct_remaining:.0f}% of gap\nremains via proxies',
              xy=(1, nr_mean), xytext=(0.55, nr_mean + 0.018),
              color=NEUTRAL, fontsize=10,
              arrowprops=dict(arrowstyle='->', color=NEUTRAL))

fig3.tight_layout()
fig3.savefig(os.path.join(OUT, 'fig3_direct_vs_proxy.png'), dpi=150,
             bbox_inches='tight', facecolor=BG)
print("  Saved fig3_direct_vs_proxy.png")

print("\nDone.")
print(f"\nKey finding ({N_FOLDS}-fold CV):")
print(f"  White-Black prediction gap:  {gap_mean:.4f} +/- {gap_std:.4f}")
print(f"  Proxy gap (no direct race):  {proxy_mean:.1f}% +/- {proxy_std:.1f}% of total gap persists")
print(f"  Top proxies: {', '.join([FEATURE_LABELS[f] for f in top_features])}")
