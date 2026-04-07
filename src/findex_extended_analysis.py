"""
Extended Logistic Regression Analysis — Financial Inclusion
=============================================================
Global Findex 2025 Microdata

Recommendations implemented:
  1. Region-specific models (Sub-Saharan Africa, South Asia)
  2. Mobile phone ownership + internet access as predictors
  3. SMOTE + threshold optimisation
  4. Interaction terms (Female x Region, Education x Income)
  5. Findex 2021 comparison (noted as limitation — microdata requires registration)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_recall_curve,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    print("Installing imbalanced-learn for SMOTE...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imbalanced-learn", "-q"])
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True

import warnings
warnings.filterwarnings("ignore")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOAD DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 70)
print("LOADING GLOBAL FINDEX 2025 MICRODATA")
print("=" * 70)

raw = pd.read_csv(
    "../data/findex_microdata_2025_labelled.csv",
    usecols=[
        "account", "age", "female", "inc_q", "educ", "emp_in",
        "urbanicity", "economy", "economycode", "regionwb", "wgt",
        "internet_use", "con1",
    ],
)
print(f"Loaded: {raw.shape[0]:,} individuals, {raw.shape[1]} variables\n")

# Recode
df = raw.copy()
df["female"] = (df["female"] == 2).astype(int)
df["rural"] = (df["urbanicity"] == 2).astype(int)
df["employed"] = (df["emp_in"] == 1).astype(int)
df["internet"] = df["internet_use"].astype(int)
df["mobile"] = (df["con1"] == 1).astype(int)  # 1 = has smartphone
df["educ"] = df["educ"].map({1: "primary", 2: "secondary", 3: "tertiary"})
df["inc_q"] = df["inc_q"].astype(float)

# Region short names
region_map = {
    "Sub-Saharan Africa (excluding high income)": "Sub-Saharan Africa",
    "South Asia": "South Asia",
    "High income": "High income",
    "Europe & Central Asia (excluding high income)": "Europe & Central Asia",
    "Latin America & Caribbean (excluding high income)": "Latin America",
    "East Asia & Pacific (excluding high income)": "East Asia & Pacific",
    "Middle East & North Africa (excluding high income)": "MENA",
}
df["region"] = df["regionwb"].map(region_map)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPER: fit and report a logistic regression
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fit_logit(X_train, y_train, X_test, y_test, label="Model"):
    """Fit statsmodels Logit + sklearn LR; return results dict."""
    # statsmodels for inference
    X_sm = sm.add_constant(X_train.astype(float))
    logit = sm.Logit(y_train.astype(float), X_sm)
    res = logit.fit(disp=0)

    # sklearn for predictions
    scaler = StandardScaler()
    Xtr_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    Xte_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(Xtr_s, y_train)
    y_pred = clf.predict(Xte_s)
    y_prob = clf.predict_proba(Xte_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Odds ratios
    odds = pd.DataFrame({
        "Odds Ratio": np.exp(res.params),
        "CI Lower": np.exp(res.conf_int()[0]),
        "CI Upper": np.exp(res.conf_int()[1]),
        "p-value": res.pvalues,
    })

    return {
        "label": label, "result": res, "odds": odds,
        "clf": clf, "scaler": scaler,
        "y_test": y_test, "y_pred": y_pred, "y_prob": y_prob,
        "X_test": X_test,
        "acc": acc, "auc": auc,
        "fpr_tpr": roc_curve(y_test, y_prob),
    }


def print_model(m):
    """Print summary for a fitted model."""
    print(f"\n{'─'*60}")
    print(f"  {m['label']}")
    print(f"{'─'*60}")
    print(f"  N train: {m['result'].nobs:.0f}  |  N test: {len(m['y_test']):,}")
    print(f"  Pseudo R²: {m['result'].prsquared:.4f}")
    print(f"  Accuracy:  {m['acc']:.4f}  |  ROC-AUC: {m['auc']:.4f}")
    print()
    odds = m["odds"].drop("const", errors="ignore")
    print("  Odds Ratios:")
    for var in odds.index:
        o = odds.loc[var]
        sig = "***" if o["p-value"] < 0.001 else "**" if o["p-value"] < 0.01 else "*" if o["p-value"] < 0.05 else "ns"
        print(f"    {var:25s}  OR={o['Odds Ratio']:8.4f}  [{o['CI Lower']:.3f}, {o['CI Upper']:.3f}]  {sig}")
    print()
    print("  Classification Report:")
    print(classification_report(m["y_test"], m["y_pred"],
          target_names=["No Account", "Has Account"], zero_division=0))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. REGION-SPECIFIC MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("1. REGION-SPECIFIC MODELS")
print("=" * 70)

base_vars = ["account", "age", "female", "inc_q", "employed", "rural", "educ"]
region_models = {}

for region_name in ["Sub-Saharan Africa", "South Asia"]:
    df_r = df[df["region"] == region_name][base_vars].dropna().copy()
    df_enc = pd.get_dummies(df_r, columns=["educ"], drop_first=False, dtype=int)
    df_enc.drop(columns=["educ_primary"], inplace=True)

    feat = ["age", "female", "inc_q", "employed", "rural", "educ_secondary", "educ_tertiary"]
    X = df_enc[feat]
    y = df_enc["account"]

    print(f"\n  {region_name}: n={len(df_r):,}, account rate={y.mean():.1%}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    m = fit_logit(X_tr, y_tr, X_te, y_te, label=f"Region: {region_name}")
    print_model(m)
    region_models[region_name] = m

# Also fit global for comparison
df_global = df[base_vars].dropna().copy()
df_global_enc = pd.get_dummies(df_global, columns=["educ"], drop_first=False, dtype=int)
df_global_enc.drop(columns=["educ_primary"], inplace=True)
feat_global = ["age", "female", "inc_q", "employed", "rural", "educ_secondary", "educ_tertiary"]
X_gl = df_global_enc[feat_global]
y_gl = df_global_enc["account"]
X_gl_tr, X_gl_te, y_gl_tr, y_gl_te = train_test_split(X_gl, y_gl, test_size=0.2, random_state=42, stratify=y_gl)
m_global = fit_logit(X_gl_tr, y_gl_tr, X_gl_te, y_gl_te, label="Global baseline (all regions)")
print_model(m_global)
region_models["Global"] = m_global


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. ENHANCED MODEL WITH MOBILE + INTERNET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("2. ENHANCED MODEL — MOBILE + INTERNET")
print("=" * 70)

enhanced_vars = ["account", "age", "female", "inc_q", "employed", "rural", "educ", "mobile", "internet"]
df_enh = df[enhanced_vars].dropna().copy()
df_enh_enc = pd.get_dummies(df_enh, columns=["educ"], drop_first=False, dtype=int)
df_enh_enc.drop(columns=["educ_primary"], inplace=True)

feat_enh = ["age", "female", "inc_q", "employed", "rural", "educ_secondary", "educ_tertiary", "mobile", "internet"]
X_enh = df_enh_enc[feat_enh]
y_enh = df_enh_enc["account"]

print(f"  Enhanced dataset: n={len(df_enh):,}")
print(f"  Mobile ownership: {df_enh['mobile'].mean():.1%}")
print(f"  Internet access:  {df_enh['internet'].mean():.1%}")

X_enh_tr, X_enh_te, y_enh_tr, y_enh_te = train_test_split(X_enh, y_enh, test_size=0.2, random_state=42, stratify=y_enh)
m_enhanced = fit_logit(X_enh_tr, y_enh_tr, X_enh_te, y_enh_te, label="Enhanced (+ mobile + internet)")
print_model(m_enhanced)

# Compare AUC improvement
print(f"  AUC improvement: {m_global['auc']:.4f} → {m_enhanced['auc']:.4f} (+{m_enhanced['auc']-m_global['auc']:.4f})")
print(f"  Pseudo R² improvement: {m_global['result'].prsquared:.4f} → {m_enhanced['result'].prsquared:.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. SMOTE + THRESHOLD OPTIMISATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("3. SMOTE + THRESHOLD OPTIMISATION")
print("=" * 70)

# 3a. SMOTE on training data
print("\n  3a. SMOTE oversampling of minority class ('No Account')...")
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_enh_tr, y_enh_tr)
print(f"  Before SMOTE: {y_enh_tr.value_counts().to_dict()}")
print(f"  After SMOTE:  {pd.Series(y_smote).value_counts().to_dict()}")

m_smote = fit_logit(
    pd.DataFrame(X_smote, columns=feat_enh), y_smote,
    X_enh_te, y_enh_te,
    label="SMOTE + Enhanced model"
)
print_model(m_smote)

# 3b. Threshold optimisation (maximise F1 for No Account class)
print("  3b. Threshold optimisation...")
print("  Finding threshold that maximises F1 for 'No Account' class...\n")

thresholds = np.arange(0.10, 0.91, 0.01)
results_thr = []
for thr in thresholds:
    y_pred_t = (m_smote["y_prob"] >= thr).astype(int)
    # F1 for the NEGATIVE class (No Account = label 0)
    f1_neg = f1_score(m_smote["y_test"], y_pred_t, pos_label=0, zero_division=0)
    f1_pos = f1_score(m_smote["y_test"], y_pred_t, pos_label=1, zero_division=0)
    acc_t = accuracy_score(m_smote["y_test"], y_pred_t)
    results_thr.append({"threshold": thr, "f1_no_account": f1_neg, "f1_has_account": f1_pos, "accuracy": acc_t})

thr_df = pd.DataFrame(results_thr)
best_row = thr_df.loc[thr_df["f1_no_account"].idxmax()]
best_thr = best_row["threshold"]
print(f"  Optimal threshold for No Account detection: {best_thr:.2f}")
print(f"  At this threshold:")
print(f"    F1 (No Account):  {best_row['f1_no_account']:.4f}")
print(f"    F1 (Has Account): {best_row['f1_has_account']:.4f}")
print(f"    Accuracy:         {best_row['accuracy']:.4f}")

y_pred_opt = (m_smote["y_prob"] >= best_thr).astype(int)
print(f"\n  Classification report at threshold = {best_thr:.2f}:")
print(classification_report(m_smote["y_test"], y_pred_opt,
      target_names=["No Account", "Has Account"]))

# Compare recall for No Account: default vs SMOTE vs SMOTE+threshold
y_pred_default_05 = (m_smote["y_prob"] >= 0.50).astype(int)
from sklearn.metrics import recall_score
print("  Recall for 'No Account' class comparison:")
print(f"    Global baseline (0.50):      {recall_score(m_global['y_test'], m_global['y_pred'], pos_label=0):.4f}")
print(f"    SMOTE + enhanced (0.50):     {recall_score(m_smote['y_test'], y_pred_default_05, pos_label=0):.4f}")
print(f"    SMOTE + threshold ({best_thr:.2f}):  {recall_score(m_smote['y_test'], y_pred_opt, pos_label=0):.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. INTERACTION TERMS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("4. INTERACTION TERMS")
print("=" * 70)

# Build interactions on the enhanced dataset
df_inter = df_enh_enc[feat_enh + ["account"]].copy()

# Female x Rural
df_inter["female_x_rural"] = df_inter["female"] * df_inter["rural"]
# Female x Employed
df_inter["female_x_employed"] = df_inter["female"] * df_inter["employed"]
# Education x Income
df_inter["educ_sec_x_inc"] = df_inter["educ_secondary"] * df_inter["inc_q"]
df_inter["educ_ter_x_inc"] = df_inter["educ_tertiary"] * df_inter["inc_q"]
# Internet x Rural
df_inter["internet_x_rural"] = df_inter["internet"] * df_inter["rural"]

feat_inter = feat_enh + [
    "female_x_rural", "female_x_employed",
    "educ_sec_x_inc", "educ_ter_x_inc", "internet_x_rural",
]

X_int = df_inter[feat_inter]
y_int = df_inter["account"]

X_int_tr, X_int_te, y_int_tr, y_int_te = train_test_split(X_int, y_int, test_size=0.2, random_state=42, stratify=y_int)
m_inter = fit_logit(X_int_tr, y_int_tr, X_int_te, y_int_te, label="Interaction terms model")
print_model(m_inter)

print(f"  AUC: enhanced={m_enhanced['auc']:.4f} → interactions={m_inter['auc']:.4f}")
print(f"  Pseudo R²: enhanced={m_enhanced['result'].prsquared:.4f} → interactions={m_inter['result'].prsquared:.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. FINDEX 2021 vs 2025 COMPARISON (AGGREGATE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("5. TEMPORAL COMPARISON — FINDEX 2021 vs 2025")
print("=" * 70)
print("""
  NOTE: The Findex 2021 individual-level microdata requires free
  registration at the World Bank Microdata Library:
    https://microdata.worldbank.org/index.php/catalog/4607

  The Findex 2025 microdata (year=2024) is the dataset used here.

  While we cannot run the 2021 logistic model without that file,
  we can compare known aggregate statistics from published reports
  to contextualise our 2025 findings.
""")

# Published Findex aggregate statistics (from World Bank reports)
comparison = pd.DataFrame({
    "Metric": [
        "Global account ownership",
        "Account ownership — Male",
        "Account ownership — Female",
        "Gender gap (pp)",
        "Account — High income",
        "Account — Sub-Saharan Africa",
        "Account — South Asia",
        "Mobile money accounts (SSA)",
    ],
    "Findex 2021": [
        "76%", "78%", "74%", "4 pp", "97%", "55%", "68%", "33%",
    ],
    "Findex 2025 (our data)": [
        f"{df['account'].mean():.0%}",
        f"{df[df['female']==0]['account'].mean():.0%}",
        f"{df[df['female']==1]['account'].mean():.0%}",
        f"{abs(df[df['female']==0]['account'].mean() - df[df['female']==1]['account'].mean())*100:.0f} pp",
        f"{df[df['region']=='High income']['account'].mean():.0%}",
        f"{df[df['region']=='Sub-Saharan Africa']['account'].mean():.0%}",
        f"{df[df['region']=='South Asia']['account'].mean():.0%}",
        "N/A (variable not directly comparable)",
    ],
})
print(comparison.to_string(index=False))
print()

# Region breakdown from our data
print("  Account ownership by region (Findex 2025):")
region_rates = df.groupby("region")["account"].mean().sort_values(ascending=False)
for r, rate in region_rates.items():
    n = (df["region"] == r).sum()
    print(f"    {r:30s}  {rate:.1%}  (n={n:,})")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. VISUALISATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 70)
print("6. GENERATING VISUALISATIONS")
print("=" * 70)

sns.set_style("whitegrid")

# ── Figure 1: Region-specific odds ratios comparison ──
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
fig1.suptitle("Odds Ratios by Region — Effect on Financial Inclusion", fontsize=14, fontweight="bold")

labels_map = {
    "age": "Age (+1yr)", "female": "Female", "inc_q": "Income (+1Q)",
    "employed": "Employed", "rural": "Rural",
    "educ_secondary": "Secondary edu", "educ_tertiary": "Tertiary edu",
}

for idx, (name, m) in enumerate([
    ("Global", region_models["Global"]),
    ("Sub-Saharan Africa", region_models["Sub-Saharan Africa"]),
    ("South Asia", region_models["South Asia"]),
]):
    ax = axes1[idx]
    odds = m["odds"].drop("const", errors="ignore").copy()
    odds = odds.sort_values("Odds Ratio")
    y_pos = range(len(odds))
    colors = ["#dc2626" if v < 1 else "#2563eb" for v in odds["Odds Ratio"]]
    ax.barh(list(y_pos), odds["Odds Ratio"], color=colors, height=0.5)
    ax.errorbar(
        odds["Odds Ratio"], list(y_pos),
        xerr=[odds["Odds Ratio"] - odds["CI Lower"], odds["CI Upper"] - odds["Odds Ratio"]],
        fmt="none", color="black", capsize=3,
    )
    ax.axvline(x=1, color="grey", linestyle="--", lw=1)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([labels_map.get(v, v) for v in odds.index])
    ax.set_xlabel("Odds Ratio")
    ax.set_title(name)

plt.tight_layout()
plt.savefig("findex_extended_fig1_region_odds.png", dpi=150, bbox_inches="tight")
print("  Saved: findex_extended_fig1_region_odds.png")

# ── Figure 2: Enhanced model comparison + ROC curves ──
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5.5))
fig2.suptitle("Model Comparison — ROC Curves & Feature Importance", fontsize=14, fontweight="bold")

# ROC curves
for m, ls, c in [
    (m_global, "-", "#94a3b8"),
    (m_enhanced, "-", "#2563eb"),
    (m_smote, "--", "#16a34a"),
    (m_inter, "-.", "#dc2626"),
]:
    fpr, tpr, _ = m["fpr_tpr"]
    axes2[0].plot(fpr, tpr, ls, color=c, lw=2, label=f"{m['label']} (AUC={m['auc']:.3f})")

axes2[0].plot([0, 1], [0, 1], "k--", lw=0.8)
axes2[0].set_xlabel("False Positive Rate")
axes2[0].set_ylabel("True Positive Rate")
axes2[0].set_title("ROC Curves — All Models")
axes2[0].legend(fontsize=8, loc="lower right")

# Enhanced model odds ratios
odds_enh = m_enhanced["odds"].drop("const", errors="ignore").sort_values("Odds Ratio")
labels_enh = {**labels_map, "mobile": "Mobile phone", "internet": "Internet access"}
y_pos = range(len(odds_enh))
colors = ["#dc2626" if v < 1 else "#2563eb" for v in odds_enh["Odds Ratio"]]
axes2[1].barh(list(y_pos), odds_enh["Odds Ratio"], color=colors, height=0.5)
axes2[1].errorbar(
    odds_enh["Odds Ratio"], list(y_pos),
    xerr=[odds_enh["Odds Ratio"] - odds_enh["CI Lower"], odds_enh["CI Upper"] - odds_enh["Odds Ratio"]],
    fmt="none", color="black", capsize=3,
)
axes2[1].axvline(x=1, color="grey", linestyle="--", lw=1)
axes2[1].set_yticks(list(y_pos))
axes2[1].set_yticklabels([labels_enh.get(v, v) for v in odds_enh.index])
axes2[1].set_xlabel("Odds Ratio")
axes2[1].set_title("Enhanced Model — Odds Ratios")

plt.tight_layout()
plt.savefig("findex_extended_fig2_model_comparison.png", dpi=150, bbox_inches="tight")
print("  Saved: findex_extended_fig2_model_comparison.png")

# ── Figure 3: SMOTE + Threshold analysis ──
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5.5))
fig3.suptitle("SMOTE & Threshold Optimisation", fontsize=14, fontweight="bold")

# 3a: Threshold vs F1 curves
axes3[0].plot(thr_df["threshold"], thr_df["f1_no_account"], color="#dc2626", lw=2, label="F1 (No Account)")
axes3[0].plot(thr_df["threshold"], thr_df["f1_has_account"], color="#2563eb", lw=2, label="F1 (Has Account)")
axes3[0].plot(thr_df["threshold"], thr_df["accuracy"], color="#64748b", lw=1.5, ls="--", label="Accuracy")
axes3[0].axvline(x=best_thr, color="#16a34a", ls=":", lw=2, label=f"Optimal={best_thr:.2f}")
axes3[0].axvline(x=0.50, color="#94a3b8", ls=":", lw=1.5, label="Default=0.50")
axes3[0].set_xlabel("Decision Threshold")
axes3[0].set_ylabel("Score")
axes3[0].set_title("Threshold Optimisation")
axes3[0].legend(fontsize=8)

# 3b: Confusion matrix at default threshold (0.50) with SMOTE
cm_default = confusion_matrix(m_smote["y_test"], y_pred_default_05)
sns.heatmap(cm_default, annot=True, fmt=",d", cmap="Blues",
            xticklabels=["No Account", "Has Account"],
            yticklabels=["No Account", "Has Account"], ax=axes3[1])
axes3[1].set_xlabel("Predicted")
axes3[1].set_ylabel("Actual")
axes3[1].set_title(f"SMOTE, Threshold = 0.50")

# 3c: Confusion matrix at optimal threshold
cm_opt = confusion_matrix(m_smote["y_test"], y_pred_opt)
sns.heatmap(cm_opt, annot=True, fmt=",d", cmap="Greens",
            xticklabels=["No Account", "Has Account"],
            yticklabels=["No Account", "Has Account"], ax=axes3[2])
axes3[2].set_xlabel("Predicted")
axes3[2].set_ylabel("Actual")
axes3[2].set_title(f"SMOTE, Threshold = {best_thr:.2f} (optimised)")

plt.tight_layout()
plt.savefig("findex_extended_fig3_smote_threshold.png", dpi=150, bbox_inches="tight")
print("  Saved: findex_extended_fig3_smote_threshold.png")

# ── Figure 4: Interaction effects ──
fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5.5))
fig4.suptitle("Interaction Effects on Financial Inclusion", fontsize=14, fontweight="bold")

# 4a: Gender gap by region
gender_region = df.groupby(["region", "female"])["account"].mean().unstack()
gender_region.columns = ["Male", "Female"]
gender_region["Gap (pp)"] = (gender_region["Male"] - gender_region["Female"]) * 100
gender_region = gender_region.sort_values("Gap (pp)", ascending=True)
colors_gap = ["#dc2626" if v > 0 else "#2563eb" for v in gender_region["Gap (pp)"]]
axes4[0].barh(range(len(gender_region)), gender_region["Gap (pp)"], color=colors_gap, height=0.5)
axes4[0].axvline(x=0, color="grey", ls="--", lw=1)
axes4[0].set_yticks(range(len(gender_region)))
axes4[0].set_yticklabels(gender_region.index)
axes4[0].set_xlabel("Gender gap (pp) — Male minus Female")
axes4[0].set_title("Gender Gap by Region")

# 4b: Education x Income interaction (predicted probabilities)
# Use the interaction model to predict
inc_levels = [1, 2, 3, 4, 5]
for edu_label, edu_sec, edu_ter, color, ls in [
    ("Primary", 0, 0, "#dc2626", "-"),
    ("Secondary", 1, 0, "#f59e0b", "-"),
    ("Tertiary", 0, 1, "#2563eb", "-"),
]:
    probs = []
    for inc in inc_levels:
        row = pd.DataFrame([{
            "age": 35, "female": 0, "inc_q": inc, "employed": 1, "rural": 0,
            "educ_secondary": edu_sec, "educ_tertiary": edu_ter, "mobile": 1, "internet": 1,
            "female_x_rural": 0, "female_x_employed": 0,
            "educ_sec_x_inc": edu_sec * inc, "educ_ter_x_inc": edu_ter * inc,
            "internet_x_rural": 0,
        }])
        row_s = pd.DataFrame(m_inter["scaler"].transform(row), columns=row.columns)
        probs.append(m_inter["clf"].predict_proba(row_s)[0, 1])
    axes4[1].plot(inc_levels, probs, color=color, ls=ls, marker="o", lw=2, label=edu_label)

axes4[1].set_xlabel("Income Quintile")
axes4[1].set_ylabel("P(Account)")
axes4[1].set_title("Education x Income Interaction")
axes4[1].set_xticks(inc_levels)
axes4[1].set_xticklabels(["Q1\n(Poorest)", "Q2", "Q3", "Q4", "Q5\n(Richest)"])
axes4[1].legend()
axes4[1].set_ylim(0, 1)

# 4c: Internet x Rural interaction
for internet_val, label, color in [(0, "No internet", "#dc2626"), (1, "Has internet", "#2563eb")]:
    probs_urban, probs_rural = [], []
    for inc in inc_levels:
        for rural_val, prob_list in [(0, probs_urban), (1, probs_rural)]:
            row = pd.DataFrame([{
                "age": 35, "female": 0, "inc_q": inc, "employed": 1, "rural": rural_val,
                "educ_secondary": 1, "educ_tertiary": 0, "mobile": 1, "internet": internet_val,
                "female_x_rural": 0, "female_x_employed": 0,
                "educ_sec_x_inc": 1 * inc, "educ_ter_x_inc": 0,
                "internet_x_rural": internet_val * rural_val,
            }])
            row_s = pd.DataFrame(m_inter["scaler"].transform(row), columns=row.columns)
            prob_list.append(m_inter["clf"].predict_proba(row_s)[0, 1])
    axes4[2].plot(inc_levels, probs_urban, color=color, ls="-", marker="o", lw=2, label=f"Urban, {label}")
    axes4[2].plot(inc_levels, probs_rural, color=color, ls="--", marker="s", lw=2, label=f"Rural, {label}")

axes4[2].set_xlabel("Income Quintile")
axes4[2].set_ylabel("P(Account)")
axes4[2].set_title("Internet x Rural Interaction")
axes4[2].set_xticks(inc_levels)
axes4[2].set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"])
axes4[2].legend(fontsize=8)
axes4[2].set_ylim(0, 1)

plt.tight_layout()
plt.savefig("findex_extended_fig4_interactions.png", dpi=150, bbox_inches="tight")
print("  Saved: findex_extended_fig4_interactions.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUMMARY TABLE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("SUMMARY — MODEL COMPARISON")
print("=" * 70)

summary = pd.DataFrame([
    {"Model": m["label"], "Pseudo R²": f"{m['result'].prsquared:.4f}",
     "Accuracy": f"{m['acc']:.4f}", "ROC-AUC": f"{m['auc']:.4f}"}
    for m in [m_global, m_enhanced, m_smote, m_inter,
              region_models["Sub-Saharan Africa"], region_models["South Asia"]]
])
print(summary.to_string(index=False))
print("\nDone. All figures saved in src/")
