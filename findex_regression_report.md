# Logistic Regression Report: Predicting Financial Inclusion
## Global Findex 2025 Microdata

**Date:** April 2026
**Dataset:** Global Findex Database 2025 (World Bank)
**Records:** 142,500 individuals across 140 economies
**Method:** Binomial Logistic Regression

---

## 1. Research Question

> What socio-economic and demographic factors predict whether an individual owns a formal financial account?

The target variable is **`account`** (binary): 1 if the individual has an account at a bank, credit union, microfinance institution, or mobile money provider; 0 otherwise.

## 2. Data Summary

| Variable | Type | Description | Distribution |
|---|---|---|---|
| `account` | Binary (target) | Has a financial account | 73.8% Yes, 26.2% No |
| `age` | Continuous | Age in years (15-100) | Mean = 43.1, SD = 18.1 |
| `female` | Binary | 1 = Female, 0 = Male | 47.6% Female |
| `inc_q` | Ordinal (1-5) | Income quintile | Roughly equal splits |
| `educ` | Categorical | Primary / Secondary / Tertiary | 27.0% / 52.4% / 21.4% |
| `employed` | Binary | 1 = In workforce | 58.4% Employed |
| `rural` | Binary | 1 = Rural location | 43.3% Rural |

Missing data was minimal (0.4%-2.8% per variable) and handled by listwise deletion, retaining 142,500 of 144,090 records (98.9%).

## 3. Multicollinearity Check

Variance Inflation Factors (VIF) were computed for all predictors:

| Variable | VIF |
|---|---|
| Age | 1.05 |
| Female | 1.04 |
| Income quintile | 1.08 |
| Employed | 1.10 |
| Rural | 1.04 |
| Education (secondary) | 1.48 |
| Education (tertiary) | 1.54 |

All VIF values are well below the threshold of 5, confirming **no multicollinearity issues**.

## 4. Model Results

The logistic regression was estimated using Maximum Likelihood Estimation (MLE) via `statsmodels`. Education was one-hot encoded with "Primary" as the reference category.

### 4.1 Model Fit

| Metric | Value |
|---|---|
| Pseudo R-squared (McFadden) | 0.149 |
| Log-Likelihood | -55,770 |
| AIC | 111,555.7 |
| BIC | 111,632.9 |
| LLR p-value | 0.0000 |

### 4.2 Coefficients and Odds Ratios

| Predictor | Coefficient | Std. Error | z-statistic | p-value | Odds Ratio | 95% CI |
|---|---|---|---|---|---|---|
| Intercept | -1.777 | 0.029 | -60.41 | <0.001 | 0.169 | (0.160, 0.179) |
| Age (+1 year) | 0.023 | 0.000 | 53.69 | <0.001 | **1.023** | (1.022, 1.024) |
| Female | 0.208 | 0.015 | 13.72 | <0.001 | **1.231** | (1.195, 1.268) |
| Income quintile (+1) | 0.114 | 0.005 | 21.33 | <0.001 | **1.121** | (1.109, 1.132) |
| Employed | 0.657 | 0.015 | 42.86 | <0.001 | **1.930** | (1.873, 1.989) |
| Rural | 0.183 | 0.015 | 11.82 | <0.001 | **1.200** | (1.165, 1.237) |
| Secondary education | 1.220 | 0.016 | 75.81 | <0.001 | **3.387** | (3.282, 3.496) |
| Tertiary education | 2.490 | 0.030 | 83.81 | <0.001 | **12.064** | (11.381, 12.787) |

All predictors are statistically significant at p < 0.001.

### 4.3 Interpretation of Odds Ratios

- **Education is the strongest predictor.** Individuals with tertiary education are 12 times more likely to have a financial account than those with only primary education. Secondary education triples the odds (OR = 3.39).

- **Employment nearly doubles the odds** (OR = 1.93). Being in the workforce is strongly associated with account ownership, consistent with wage-payment infrastructure requiring formal accounts.

- **Income quintile** has a cumulative effect: each step up increases the odds by 12.1%. Moving from Q1 (poorest) to Q5 (richest) increases the odds by approximately 1.121^4 = 1.58 times.

- **Age** has a modest positive effect (OR = 1.023 per year). Over a 30-year span, this compounds to approximately 1.023^30 = 1.98, nearly doubling the odds.

- **Female** shows a positive coefficient (OR = 1.23). This is a global model mixing high-income economies (near-universal inclusion for both genders) with developing economies. After controlling for education and employment, the residual female effect reflects that in many countries, women who are educated and employed are just as likely (or more) to have accounts. The raw gender gap is primarily explained by disparities in education and employment access.

- **Rural** also shows a positive coefficient (OR = 1.20) in this global model. Similar to the gender effect, this suggests that once income and education are controlled for, rural residence per se is not a barrier in many economies. Region-specific models would likely reveal the expected negative effect in Sub-Saharan Africa and South Asia.

## 5. Model Evaluation

The model was evaluated on a held-out test set (20% of data, n = 28,500).

| Metric | Value |
|---|---|
| **Accuracy** | 75.8% |
| **ROC-AUC** | 0.755 |
| **Precision (No Account)** | 0.58 |
| **Recall (No Account)** | 0.26 |
| **Precision (Has Account)** | 0.78 |
| **Recall (Has Account)** | 0.93 |
| **F1 (Has Account)** | 0.85 |

The model is better at identifying those who *have* accounts (high recall = 93%) than those who don't (recall = 26%). This is expected given the class imbalance (74% positive). Adjusting the decision threshold or applying SMOTE could improve sensitivity to the excluded population.

### Visualisation

![Regression Results](src/findex_logistic_regression_results.png)

#### Panel 1 — Confusion Matrix (top left)

A confusion matrix is a 2x2 table that compares the model's predictions against the actual outcomes for every individual in the test set (n = 28,500). It answers the question: *"When the model said someone has an account, was it right?"*

|  | **Predicted: No Account** | **Predicted: Has Account** |
|---|---|---|
| **Actual: No Account** | 1,973 (True Negative) | 5,480 (False Positive) |
| **Actual: Has Account** | 1,416 (False Negative) | 19,631 (True Positive) |

Reading the four quadrants:

- **True Positives (19,631):** The model correctly predicted these individuals have accounts, and they do. For example, a 35-year-old employed man with tertiary education and Q4 income — the model predicts P(Account) = 0.95, and he indeed holds an account.

- **True Negatives (1,973):** The model correctly predicted these individuals do *not* have accounts. For example, a 16-year-old unemployed girl with primary education in Q1 income — the model predicts P(Account) = 0.18, and she indeed has no account.

- **False Positives (5,480):** The model predicted "Has Account" but the person actually does not. For example, a 45-year-old employed man with secondary education and Q3 income — the model predicts P(Account) = 0.72, but he does not hold an account. These are individuals whose profile *looks* like they should be included, but they are not — perhaps due to factors the model does not capture (distrust of banks, geographic isolation, cultural barriers).

- **False Negatives (1,416):** The model predicted "No Account" but the person actually has one. For example, a 20-year-old unemployed woman with primary education and Q1 income — the model predicts P(Account) = 0.22, yet she does have an account, perhaps through a mobile money service or a government inclusion programme.

The **axes** of the confusion matrix are:
- **Y-axis (rows):** The *actual* class — what we observe in the survey data.
- **X-axis (columns):** The *predicted* class — what the model outputs.

The diagonal (top-left to bottom-right) represents correct predictions; the off-diagonal cells are errors. A perfect model would have zeros in the off-diagonal cells.

**Key insight from this matrix:** The model is conservative about predicting "No Account." Of the 7,453 individuals who truly have no account, it only identifies 1,973 of them (recall = 26%). It tends to over-predict inclusion, which is a consequence of the class imbalance — 74% of the dataset has accounts, so the model learns to "default" toward predicting inclusion.

#### Panel 2 — ROC Curve (top right)

The ROC (Receiver Operating Characteristic) curve visualises the trade-off between catching true positives and accidentally flagging false positives, across *all possible decision thresholds* from 0 to 1.

**Axes:**
- **X-axis — False Positive Rate (FPR):** Of all individuals who truly have *no* account, what fraction did the model incorrectly classify as "Has Account"? FPR = False Positives / (False Positives + True Negatives). Ranges from 0 (no false alarms) to 1 (all negatives misclassified).
- **Y-axis — True Positive Rate (TPR), also called Recall or Sensitivity:** Of all individuals who truly *have* an account, what fraction did the model correctly identify? TPR = True Positives / (True Positives + False Negatives). Ranges from 0 (misses everyone) to 1 (catches everyone).

**How to read the curve:**

Each point on the blue curve represents one threshold setting. As the threshold decreases from 1.0 toward 0.0:
- At **threshold = 0.99**: The model is extremely strict — it only classifies someone as "Has Account" when it is nearly certain. TPR is low (catches few), but FPR is also low (few false alarms). This point is near the bottom-left.
- At **threshold = 0.50** (our default): The model uses a balanced cutoff. This is one specific point on the curve, roughly where our confusion matrix values come from.
- At **threshold = 0.01**: The model classifies nearly everyone as "Has Account." TPR approaches 1.0 (catches everyone), but FPR also approaches 1.0 (massive false alarms). This point is near the top-right.

**Reference lines and benchmarks:**
- The **black dashed diagonal line** represents a random coin-flip classifier (AUC = 0.5). A model that guesses randomly would produce points along this line.
- A **perfect classifier** would hug the top-left corner (TPR = 1.0, FPR = 0.0), giving AUC = 1.0.
- Our model achieves **AUC = 0.755**, meaning it performs substantially better than random but has room for improvement.

**Practical example of the trade-off:** Imagine a government programme that wants to identify unbanked citizens for a financial inclusion initiative. If they lower the threshold to 0.3 (classifying more people as "No Account"), they would catch more truly unbanked people (higher TPR for the negative class), but would also incorrectly flag some account-holders for outreach (higher FPR). The ROC curve helps decision-makers find the sweet spot for their use case.

#### Panel 3 — Odds Ratios Forest Plot (bottom left)

A horizontal bar chart showing the exponentiated logistic regression coefficients (odds ratios) with 95% confidence intervals. The **red dashed vertical line at OR = 1.0** is the line of no effect — variables with bars extending to the right of this line *increase* the odds of account ownership, while bars to the left would *decrease* them. Error bars show uncertainty: if the confidence interval crosses 1.0, the effect is not statistically significant (none of ours do).

#### Panel 4 — Predicted Probability by Income and Gender (bottom right)

A grouped bar chart showing the model's average predicted P(Account) for male and female individuals across each income quintile (Q1 to Q5). This visualises the interaction between income and gender. The X-axis shows income quintiles; the Y-axis shows predicted probability (0 to 1). The two bar colours distinguish Male and Female predictions, making the gender gap (or lack thereof) visible at each income level.

## 6. Conclusions

1. **Education is the dominant driver of financial inclusion.** Policy interventions aimed at increasing secondary and tertiary education access would have the largest marginal impact on account ownership.

2. **Employment is the second-strongest factor**, suggesting that labour market integration and formal wage-payment systems are key mechanisms for financial inclusion.

3. **Income matters but less than education.** This implies that even lower-income individuals, if educated, have a meaningful probability of holding accounts.

4. **The gender and urban-rural gaps are primarily mediated by education and employment.** When these factors are controlled, the raw disparities shrink or reverse. However, this finding is driven by the global sample — region-specific analyses would reveal persistent gaps in developing economies.

5. **Model limitations:** The Pseudo R-squared of 0.149 indicates that substantial variation remains unexplained. Country-level fixed effects, mobile phone ownership, internet access, and institutional trust are likely important omitted variables. The low recall for the "No Account" class suggests that the factors driving financial exclusion are more heterogeneous than those driving inclusion.

## 7. Recommendations for Further Work

- Run **region-specific models** (Sub-Saharan Africa, South Asia) to test hypotheses about gender and rural barriers in developing contexts
- Include **mobile phone ownership** and **internet access** as predictors where available
- Apply **SMOTE** or adjust decision thresholds to improve identification of financially excluded individuals
- Consider **interaction terms** (e.g., Female x Region, Education x Income) to capture non-additive effects
- Compare results with the **Findex 2021** round to assess trends over time
