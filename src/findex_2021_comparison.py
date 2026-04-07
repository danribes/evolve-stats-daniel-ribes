"""
Findex 2021 vs 2025 — Full Coefficient Comparison
====================================================
Fits identical logistic regression models on both rounds
and compares odds ratios, model fit, and regional patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve, accuracy_score,
)
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOAD BOTH DATASETS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 70)
print("LOADING FINDEX 2021 AND 2025 MICRODATA")
print("=" * 70)

# --- 2021 ---
raw21 = pd.read_csv(
    "../data/findex_microdata_2021.csv",
    usecols=["account", "age", "female", "inc_q", "educ", "emp_in",
             "mobileowner", "internetaccess", "urbanicity_f2f", "regionwb"],
    encoding="latin-1",
)
df21 = raw21.copy()
df21["year"] = 2021
df21["female"] = (df21["female"] == 2).astype(int)
df21["employed"] = (df21["emp_in"] == 1).astype(int)
# urbanicity_f2f: 1=urban, 2=rural (47% missing — phone surveys lack this)
df21["rural"] = (df21["urbanicity_f2f"] == 2).astype(float)
df21.loc[df21["urbanicity_f2f"].isna(), "rural"] = np.nan
# mobileowner: 1=yes, 2=no (3,4 are rare DK/refused → treat as missing)
df21["mobile"] = df21["mobileowner"].map({1: 1, 2: 0})
# internetaccess: 1=yes, 2=no
df21["internet"] = df21["internetaccess"].map({1: 1, 2: 0})
# educ: 1=primary, 2=secondary, 3=tertiary (4,5 are DK/refused → drop)
df21["educ"] = df21["educ"].map({1: "primary", 2: "secondary", 3: "tertiary"})
df21["inc_q"] = df21["inc_q"].astype(float)

region_map = {
    "Sub-Saharan Africa (excluding high income)": "Sub-Saharan Africa",
    "South Asia": "South Asia",
    "High income": "High income",
    "Europe & Central Asia (excluding high income)": "Europe & Central Asia",
    "Latin America & Caribbean (excluding high income)": "Latin America",
    "East Asia & Pacific (excluding high income)": "East Asia & Pacific",
    "Middle East & North Africa (excluding high income)": "MENA",
}
df21["region"] = df21["regionwb"].map(region_map)

print(f"Findex 2021: {len(df21):,} individuals")

# --- 2025 ---
raw25 = pd.read_csv(
    "../data/findex_microdata_2025_labelled.csv",
    usecols=["account", "age", "female", "inc_q", "educ", "emp_in",
             "urbanicity", "regionwb", "internet_use", "con1"],
)
df25 = raw25.copy()
df25["year"] = 2025
df25["female"] = (df25["female"] == 2).astype(int)
df25["employed"] = (df25["emp_in"] == 1).astype(int)
df25["rural"] = (df25["urbanicity"] == 2).astype(float)
df25.loc[df25["urbanicity"].isna(), "rural"] = np.nan
df25["mobile"] = (df25["con1"] == 1).astype(float)
df25.loc[df25["con1"].isna(), "mobile"] = np.nan
df25["internet"] = df25["internet_use"].astype(float)
df25["educ"] = df25["educ"].map({1: "primary", 2: "secondary", 3: "tertiary"})
df25["inc_q"] = df25["inc_q"].astype(float)
df25["region"] = df25["regionwb"].map(region_map)

print(f"Findex 2025: {len(df25):,} individuals\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPER: fit logistic regression
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fit_model(df_input, features, label="Model"):
    """Fit logistic regression on given dataframe and features."""
    model_vars = features + ["account"]
    df_clean = df_input[model_vars].dropna().copy()

    # One-hot encode education if present
    if "educ" in features:
        df_clean = pd.get_dummies(df_clean, columns=["educ"], drop_first=False, dtype=int)
        df_clean.drop(columns=["educ_primary"], inplace=True, errors="ignore")
        feat = [f for f in features if f != "educ"] + ["educ_secondary", "educ_tertiary"]
    else:
        feat = features

    X = df_clean[feat]
    y = df_clean["account"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # statsmodels
    X_sm = sm.add_constant(X_tr.astype(float))
    res = sm.Logit(y_tr.astype(float), X_sm).fit(disp=0)

    # sklearn
    scaler = StandardScaler()
    Xtr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
    Xte_s = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(Xtr_s, y_tr)
    y_pred = clf.predict(Xte_s)
    y_prob = clf.predict_proba(Xte_s)[:, 1]

    odds = pd.DataFrame({
        "Odds Ratio": np.exp(res.params),
        "CI Lower": np.exp(res.conf_int()[0]),
        "CI Upper": np.exp(res.conf_int()[1]),
        "p-value": res.pvalues,
    })

    return {
        "label": label,
        "n": len(df_clean),
        "account_rate": y.mean(),
        "result": res, "odds": odds,
        "acc": accuracy_score(y_te, y_pred),
        "auc": roc_auc_score(y_te, y_prob),
        "fpr_tpr": roc_curve(y_te, y_prob),
        "y_test": y_te, "y_pred": y_pred, "y_prob": y_prob,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. GLOBAL MODEL COMPARISON (same 7 features)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 70)
print("1. GLOBAL MODEL COMPARISON (7 base features)")
print("=" * 70)

base_feat = ["age", "female", "inc_q", "employed", "rural", "educ"]

m21_global = fit_model(df21, base_feat, "Findex 2021 — Global")
m25_global = fit_model(df25, base_feat, "Findex 2025 — Global")

for m in [m21_global, m25_global]:
    print(f"\n  {m['label']}")
    print(f"  n = {m['n']:,}, account rate = {m['account_rate']:.1%}")
    print(f"  Pseudo R² = {m['result'].prsquared:.4f}, Accuracy = {m['acc']:.4f}, AUC = {m['auc']:.4f}")
    odds = m["odds"].drop("const", errors="ignore")
    for var in odds.index:
        o = odds.loc[var]
        sig = "***" if o["p-value"] < 0.001 else "ns"
        print(f"    {var:22s}  OR={o['Odds Ratio']:8.4f}  [{o['CI Lower']:.3f}, {o['CI Upper']:.3f}]  {sig}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. ENHANCED MODEL COMPARISON (+ mobile + internet)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n\n" + "=" * 70)
print("2. ENHANCED MODEL COMPARISON (+ mobile + internet)")
print("=" * 70)

enh_feat = ["age", "female", "inc_q", "employed", "rural", "educ", "mobile", "internet"]

m21_enh = fit_model(df21, enh_feat, "Findex 2021 — Enhanced")
m25_enh = fit_model(df25, enh_feat, "Findex 2025 — Enhanced")

for m in [m21_enh, m25_enh]:
    print(f"\n  {m['label']}")
    print(f"  n = {m['n']:,}, account rate = {m['account_rate']:.1%}")
    print(f"  Pseudo R² = {m['result'].prsquared:.4f}, Accuracy = {m['acc']:.4f}, AUC = {m['auc']:.4f}")
    odds = m["odds"].drop("const", errors="ignore")
    for var in odds.index:
        o = odds.loc[var]
        sig = "***" if o["p-value"] < 0.001 else "ns"
        print(f"    {var:22s}  OR={o['Odds Ratio']:8.4f}  [{o['CI Lower']:.3f}, {o['CI Upper']:.3f}]  {sig}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. REGION-SPECIFIC COMPARISON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n\n" + "=" * 70)
print("3. REGION-SPECIFIC COMPARISON (Sub-Saharan Africa, South Asia)")
print("=" * 70)

# For region models, drop rural (high missingness in 2021 phone surveys)
region_feat = ["age", "female", "inc_q", "employed", "educ"]
region_models = {}

for region_name in ["Sub-Saharan Africa", "South Asia"]:
    print(f"\n  ── {region_name} ──")
    for yr, df_yr in [(2021, df21), (2025, df25)]:
        df_r = df_yr[df_yr["region"] == region_name].copy()
        m = fit_model(df_r, region_feat, f"Findex {yr} — {region_name}")
        region_models[(yr, region_name)] = m
        print(f"\n  {m['label']}")
        print(f"  n = {m['n']:,}, account rate = {m['account_rate']:.1%}")
        print(f"  Pseudo R² = {m['result'].prsquared:.4f}, AUC = {m['auc']:.4f}")
        odds = m["odds"].drop("const", errors="ignore")
        for var in odds.index:
            o = odds.loc[var]
            sig = "***" if o["p-value"] < 0.001 else "**" if o["p-value"] < 0.01 else "*" if o["p-value"] < 0.05 else "ns"
            print(f"    {var:22s}  OR={o['Odds Ratio']:8.4f}  [{o['CI Lower']:.3f}, {o['CI Upper']:.3f}]  {sig}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. ACCOUNT OWNERSHIP BY REGION — TEMPORAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n\n" + "=" * 70)
print("4. ACCOUNT OWNERSHIP BY REGION — 2021 vs 2025")
print("=" * 70)

regions_all = ["High income", "Europe & Central Asia", "East Asia & Pacific",
               "Latin America", "South Asia", "Sub-Saharan Africa", "MENA"]

comparison_rows = []
for r in regions_all:
    r21 = df21[df21["region"] == r]["account"].mean()
    r25 = df25[df25["region"] == r]["account"].mean()
    n21 = (df21["region"] == r).sum()
    n25 = (df25["region"] == r).sum()
    comparison_rows.append({
        "Region": r,
        "2021 rate": f"{r21:.1%}",
        "2025 rate": f"{r25:.1%}",
        "Change (pp)": f"{(r25-r21)*100:+.1f}",
        "n (2021)": f"{n21:,}",
        "n (2025)": f"{n25:,}",
    })

comp_df = pd.DataFrame(comparison_rows)
print(comp_df.to_string(index=False))

# Gender gap by region and year
print("\n\n  Gender gap by region (Male - Female, pp):")
print(f"  {'Region':30s}  {'2021':>8s}  {'2025':>8s}  {'Change':>8s}")
print(f"  {'─'*30}  {'─'*8}  {'─'*8}  {'─'*8}")
for r in regions_all:
    for yr, df_yr in [(2021, df21), (2025, df25)]:
        m_rate = df_yr[(df_yr["region"] == r) & (df_yr["female"] == 0)]["account"].mean()
        f_rate = df_yr[(df_yr["region"] == r) & (df_yr["female"] == 1)]["account"].mean()
        if yr == 2021:
            gap21 = (m_rate - f_rate) * 100
        else:
            gap25 = (m_rate - f_rate) * 100
    print(f"  {r:30s}  {gap21:+7.1f}  {gap25:+7.1f}  {gap25-gap21:+7.1f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. VISUALISATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n\n" + "=" * 70)
print("5. GENERATING VISUALISATIONS")
print("=" * 70)

sns.set_style("whitegrid")

labels_map = {
    "age": "Age (+1yr)", "female": "Female", "inc_q": "Income (+1Q)",
    "employed": "Employed", "rural": "Rural",
    "educ_secondary": "Secondary edu", "educ_tertiary": "Tertiary edu",
    "mobile": "Mobile phone", "internet": "Internet access",
}

# ── Figure 1: Side-by-side odds ratio comparison (global, base features) ──
fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
fig1.suptitle("Global Model Comparison — Findex 2021 vs 2025 (Base Features)", fontsize=14, fontweight="bold")

for idx, m in enumerate([m21_global, m25_global]):
    ax = axes1[idx]
    odds = m["odds"].drop("const", errors="ignore").sort_values("Odds Ratio")
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
    ax.set_title(f"{m['label']}\n(n={m['n']:,}, AUC={m['auc']:.3f})")

plt.tight_layout()
plt.savefig("findex_comparison_fig1_global_odds.png", dpi=150, bbox_inches="tight")
print("  Saved: findex_comparison_fig1_global_odds.png")

# ── Figure 2: ROC curves comparison ──
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
fig2.suptitle("ROC Curve Comparison — 2021 vs 2025", fontsize=14, fontweight="bold")

for m, color, ls in [
    (m21_global, "#94a3b8", "-"),
    (m25_global, "#2563eb", "-"),
    (m21_enh, "#f59e0b", "--"),
    (m25_enh, "#dc2626", "--"),
]:
    fpr, tpr, _ = m["fpr_tpr"]
    ax2.plot(fpr, tpr, ls, color=color, lw=2, label=f"{m['label']} (AUC={m['auc']:.3f})")

ax2.plot([0, 1], [0, 1], "k--", lw=0.8)
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend(fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig("findex_comparison_fig2_roc.png", dpi=150, bbox_inches="tight")
print("  Saved: findex_comparison_fig2_roc.png")

# ── Figure 3: Account ownership by region, 2021 vs 2025 ──
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))
fig3.suptitle("Financial Inclusion Trends — 2021 vs 2025", fontsize=14, fontweight="bold")

# 3a: Rates by region
rates_21 = [df21[df21["region"] == r]["account"].mean() for r in regions_all]
rates_25 = [df25[df25["region"] == r]["account"].mean() for r in regions_all]

x = np.arange(len(regions_all))
w = 0.35
axes3[0].barh(x - w/2, rates_21, w, label="Findex 2021", color="#94a3b8")
axes3[0].barh(x + w/2, rates_25, w, label="Findex 2025", color="#2563eb")
axes3[0].set_yticks(x)
axes3[0].set_yticklabels(regions_all)
axes3[0].set_xlabel("Account Ownership Rate")
axes3[0].set_title("Account Ownership by Region")
axes3[0].legend()
axes3[0].set_xlim(0, 1)

# 3b: Gender gap by region, 2021 vs 2025
gaps_21, gaps_25 = [], []
for r in regions_all:
    for yr, df_yr, gap_list in [(2021, df21, gaps_21), (2025, df25, gaps_25)]:
        m_rate = df_yr[(df_yr["region"] == r) & (df_yr["female"] == 0)]["account"].mean()
        f_rate = df_yr[(df_yr["region"] == r) & (df_yr["female"] == 1)]["account"].mean()
        gap_list.append((m_rate - f_rate) * 100)

axes3[1].barh(x - w/2, gaps_21, w, label="Findex 2021", color="#94a3b8")
axes3[1].barh(x + w/2, gaps_25, w, label="Findex 2025", color="#2563eb")
axes3[1].set_yticks(x)
axes3[1].set_yticklabels(regions_all)
axes3[1].set_xlabel("Gender Gap (pp) — Male minus Female")
axes3[1].axvline(x=0, color="grey", ls="--", lw=1)
axes3[1].set_title("Gender Gap by Region")
axes3[1].legend()

plt.tight_layout()
plt.savefig("findex_comparison_fig3_trends.png", dpi=150, bbox_inches="tight")
print("  Saved: findex_comparison_fig3_trends.png")

# ── Figure 4: Coefficient change (2021 → 2025) ──
fig4, axes4 = plt.subplots(1, 2, figsize=(16, 6))
fig4.suptitle("Coefficient Evolution — 2021 vs 2025", fontsize=14, fontweight="bold")

# 4a: Base model — odds ratio comparison
shared_vars = ["age", "female", "inc_q", "employed", "rural", "educ_secondary", "educ_tertiary"]
or21 = m21_global["odds"].drop("const", errors="ignore").loc[shared_vars, "Odds Ratio"]
or25 = m25_global["odds"].drop("const", errors="ignore").loc[shared_vars, "Odds Ratio"]

y_pos = np.arange(len(shared_vars))
axes4[0].barh(y_pos - 0.17, or21.values, 0.34, label="2021", color="#94a3b8")
axes4[0].barh(y_pos + 0.17, or25.values, 0.34, label="2025", color="#2563eb")
axes4[0].axvline(x=1, color="grey", ls="--", lw=1)
axes4[0].set_yticks(y_pos)
axes4[0].set_yticklabels([labels_map.get(v, v) for v in shared_vars])
axes4[0].set_xlabel("Odds Ratio")
axes4[0].set_title("Base Model — Odds Ratios")
axes4[0].legend()

# 4b: Enhanced model — odds ratio comparison
enh_vars = shared_vars + ["mobile", "internet"]
or21e = m21_enh["odds"].drop("const", errors="ignore").loc[enh_vars, "Odds Ratio"]
or25e = m25_enh["odds"].drop("const", errors="ignore").loc[enh_vars, "Odds Ratio"]

y_pos2 = np.arange(len(enh_vars))
axes4[1].barh(y_pos2 - 0.17, or21e.values, 0.34, label="2021", color="#94a3b8")
axes4[1].barh(y_pos2 + 0.17, or25e.values, 0.34, label="2025", color="#2563eb")
axes4[1].axvline(x=1, color="grey", ls="--", lw=1)
axes4[1].set_yticks(y_pos2)
axes4[1].set_yticklabels([labels_map.get(v, v) for v in enh_vars])
axes4[1].set_xlabel("Odds Ratio")
axes4[1].set_title("Enhanced Model — Odds Ratios")
axes4[1].legend()

plt.tight_layout()
plt.savefig("findex_comparison_fig4_coefficients.png", dpi=150, bbox_inches="tight")
print("  Saved: findex_comparison_fig4_coefficients.png")

# ── Figure 5: Region-specific temporal comparison ──
fig5, axes5 = plt.subplots(1, 2, figsize=(16, 6))
fig5.suptitle("Regional Model Comparison — Sub-Saharan Africa & South Asia (2021 vs 2025)", fontsize=14, fontweight="bold")

region_feat_clean = ["age", "female", "inc_q", "employed", "educ_secondary", "educ_tertiary"]

for idx, region_name in enumerate(["Sub-Saharan Africa", "South Asia"]):
    ax = axes5[idx]
    m_21 = region_models[(2021, region_name)]
    m_25 = region_models[(2025, region_name)]

    or_21 = m_21["odds"].drop("const", errors="ignore").loc[region_feat_clean, "Odds Ratio"]
    or_25 = m_25["odds"].drop("const", errors="ignore").loc[region_feat_clean, "Odds Ratio"]

    y_pos = np.arange(len(region_feat_clean))
    ax.barh(y_pos - 0.17, or_21.values, 0.34, label="2021", color="#94a3b8")
    ax.barh(y_pos + 0.17, or_25.values, 0.34, label="2025", color="#2563eb")
    ax.axvline(x=1, color="grey", ls="--", lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([labels_map.get(v, v) for v in region_feat_clean])
    ax.set_xlabel("Odds Ratio")
    ax.set_title(f"{region_name}\n2021: AUC={m_21['auc']:.3f}, 2025: AUC={m_25['auc']:.3f}")
    ax.legend()

plt.tight_layout()
plt.savefig("findex_comparison_fig5_regional.png", dpi=150, bbox_inches="tight")
print("  Saved: findex_comparison_fig5_regional.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUMMARY TABLE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n\n" + "=" * 70)
print("SUMMARY — ALL MODELS")
print("=" * 70)

all_models = [
    m21_global, m25_global, m21_enh, m25_enh,
    region_models[(2021, "Sub-Saharan Africa")],
    region_models[(2025, "Sub-Saharan Africa")],
    region_models[(2021, "South Asia")],
    region_models[(2025, "South Asia")],
]
summary = pd.DataFrame([
    {"Model": m["label"], "n": f"{m['n']:,}", "Account %": f"{m['account_rate']:.1%}",
     "Pseudo R²": f"{m['result'].prsquared:.4f}", "Accuracy": f"{m['acc']:.4f}", "AUC": f"{m['auc']:.4f}"}
    for m in all_models
])
print(summary.to_string(index=False))
print("\nDone. All figures saved in src/")
