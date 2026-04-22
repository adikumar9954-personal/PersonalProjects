import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from folktables import ACSDataSource, ACSIncome
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ── 1. Pull data ───────────────────────────────────────────────────────────────
print("Downloading ACS 2018 California data...")
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["CA"], download=True)

features, labels, group = ACSIncome.df_to_numpy(acs_data)
feature_names = ACSIncome.features

df = pd.DataFrame(features, columns=feature_names)
df['income_over_50k'] = labels.astype(int)
df['race'] = group

print(f"\nDataset shape: {df.shape}")
print(f"\nClass balance:\n{df['income_over_50k'].value_counts(normalize=True).round(3)}")

# ── 2. Base rates by race ──────────────────────────────────────────────────────
race_labels = {1: 'White', 2: 'Black', 3: 'Native American',
               6: 'Asian', 8: 'Other', 9: 'Two or more'}
df['race_label'] = df['race'].map(race_labels).fillna('Other')

base_rates = (df.groupby('race_label')['income_over_50k']
                .agg(['mean', 'count'])
                .rename(columns={'mean': 'base_rate', 'count': 'n'})
                .sort_values('base_rate', ascending=False))

print("\n── Base rates by race (ground truth) ──")
print(base_rates.round(3))

fig, ax = plt.subplots(figsize=(9, 4))
base_rates['base_rate'].plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
ax.axhline(df['income_over_50k'].mean(), color='red', linestyle='--', label='Overall mean')
ax.set_ylabel('P(income > $50k)')
ax.set_title('Base rates by race — ACS 2022 California')
ax.set_xlabel('')
plt.xticks(rotation=30, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('base_rates.png', dpi=150)
print("\nSaved base_rates.png")

# ── 3. Train/test split ────────────────────────────────────────────────────────
X = df[feature_names].values
y = df['income_over_50k'].values
race = df['race'].values

X_train, X_test, y_train, y_test, race_train, race_test = train_test_split(
    X, y, race, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

# ── 4. Logistic Regression ─────────────────────────────────────────────────────
print("\nTraining Logistic Regression...")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)
lr_probs = lr.predict_proba(X_test_sc)[:, 1]
lr_preds = (lr_probs >= 0.5).astype(int)

# ── 5. XGBoost ─────────────────────────────────────────────────────────────────
print("Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
xgb_preds = xgb_model.predict(X_test)

# ── 6. Overall performance ─────────────────────────────────────────────────────
print(f"\n── Model Performance ──")
print(f"{'Model':<22} {'AUC':>8} {'Accuracy':>10}")
print("-" * 42)
print(f"{'Logistic Regression':<22} {roc_auc_score(y_test, lr_probs):>8.4f} {accuracy_score(y_test, lr_preds):>10.4f}")
print(f"{'XGBoost':<22} {roc_auc_score(y_test, xgb_probs):>8.4f} {accuracy_score(y_test, xgb_preds):>10.4f}")

# ── 7. Fairness metrics by hand ────────────────────────────────────────────────
def fairness_report(y_true, y_pred, race_arr, group_a=1, group_b=2,
                    label_a='White', label_b='Black', model_name='Model'):
    mask_a = race_arr == group_a
    mask_b = race_arr == group_b

    # Demographic parity: P(Yhat=1 | A=a)
    dp_a = y_pred[mask_a].mean()
    dp_b = y_pred[mask_b].mean()
    dp_diff = dp_a - dp_b

    # TPR: P(Yhat=1 | Y=1, A=a)
    tpr_a = y_pred[mask_a & (y_true == 1)].mean()
    tpr_b = y_pred[mask_b & (y_true == 1)].mean()

    # FPR: P(Yhat=1 | Y=0, A=a)
    fpr_a = y_pred[mask_a & (y_true == 0)].mean()
    fpr_b = y_pred[mask_b & (y_true == 0)].mean()

    eo_diff = max(abs(tpr_a - tpr_b), abs(fpr_a - fpr_b))

    # Calibration: P(Y=1 | score bucket, A=a) — check if scores mean same thing per group
    bins = np.linspace(0, 1, 6)
    cal_diffs = []
    for i in range(len(bins)-1):
        in_bin_a = mask_a & (y_pred >= bins[i]) & (y_pred < bins[i+1])
        in_bin_b = mask_b & (y_pred >= bins[i]) & (y_pred < bins[i+1])
        if in_bin_a.sum() > 10 and in_bin_b.sum() > 10:
            cal_diffs.append(abs(y_true[in_bin_a].mean() - y_true[in_bin_b].mean()))
    cal_diff = np.mean(cal_diffs) if cal_diffs else np.nan

    print(f"\n── {model_name}: {label_a} vs {label_b} ──")
    print(f"  n({label_a})={mask_a.sum():,}  n({label_b})={mask_b.sum():,}")
    print(f"\n  {'Metric':<40} {label_a:>8} {label_b:>8} {'Gap':>8}")
    print(f"  {'-'*66}")
    print(f"  {'Demographic Parity  P(Yhat=1)':<40} {dp_a:>8.4f} {dp_b:>8.4f} {dp_diff:>+8.4f}")
    print(f"  {'Equalized Odds — TPR  P(Yhat=1|Y=1)':<40} {tpr_a:>8.4f} {tpr_b:>8.4f} {tpr_a-tpr_b:>+8.4f}")
    print(f"  {'Equalized Odds — FPR  P(Yhat=1|Y=0)':<40} {fpr_a:>8.4f} {fpr_b:>8.4f} {fpr_a-fpr_b:>+8.4f}")
    print(f"  {'Equalized Odds Diff (max of above)':<40} {'':>8} {'':>8} {eo_diff:>8.4f}")
    print(f"  {'Calibration diff (avg across bins)':<40} {'':>8} {'':>8} {cal_diff:>8.4f}")

    return {'dp_diff': dp_diff, 'tpr_diff': tpr_a-tpr_b,
            'fpr_diff': fpr_a-fpr_b, 'eo_diff': eo_diff, 'cal_diff': cal_diff}

lr_metrics  = fairness_report(y_test, lr_preds,  race_test, model_name='Logistic Regression')
xgb_metrics = fairness_report(y_test, xgb_preds, race_test, model_name='XGBoost')

# ── 8. Side-by-side comparison plot ───────────────────────────────────────────
metrics = ['dp_diff', 'tpr_diff', 'fpr_diff', 'eo_diff']
labels  = ['Demographic\nParity Gap', 'TPR Gap\n(Equalized Odds)',
           'FPR Gap\n(Equalized Odds)', 'Equalized Odds\nDiff (max)']

x = np.arange(len(metrics))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, [lr_metrics[m]  for m in metrics], width, label='Logistic Regression', color='steelblue')
ax.bar(x + width/2, [xgb_metrics[m] for m in metrics], width, label='XGBoost', color='coral')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('White − Black gap (positive = favors White)')
ax.set_title('Fairness metrics by model — White vs Black, ACS 2022 CA')
ax.legend()
plt.tight_layout()
plt.savefig('fairness_comparison.png', dpi=150)
print("\n\nSaved fairness_comparison.png")
print("\nDone.")
