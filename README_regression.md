# Financial Inclusion — Logistic Regression Project

Binomial logistic regression predicting individual-level financial account ownership using the **Global Findex 2025** microdata from the World Bank.

## Quick Start

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels

# 3. Run the regression analysis
python findex_logistic_regression.py

# 4. Open the interactive simulator
open logistic_regression_simulator.html    # macOS
xdg-open logistic_regression_simulator.html # Linux
```

## Project Files

| File | Description |
|---|---|
| `Predicting Financial Inclusion with Logistic Regression.md` | Project proposal and methodology |
| `findex_logistic_regression.py` | Full regression pipeline (data prep, VIF, model, evaluation, plots) |
| `findex_regression_report.md` | Detailed results report with tables and interpretation |
| `findex_logistic_regression_results.png` | 4-panel visualisation (confusion matrix, ROC, odds ratios, income x gender) |
| `logistic_regression_simulator.html` | Interactive browser-based logistic regression simulator |
| `findex_data/` | Downloaded Findex 2025 microdata (CSV) |

## Dataset

- **Source:** [Global Findex Database 2025](https://www.worldbank.org/en/publication/globalfindex) (World Bank)
- **Records:** 144,090 individuals surveyed across 140 economies
- **Year:** 2024 survey round (published 2025)

### Variables Used

| Variable | Type | Values |
|---|---|---|
| `account` (target) | Binary | 0 = No account, 1 = Has account |
| `age` | Continuous | 15-100 years |
| `female` | Binary | 0 = Male, 1 = Female |
| `inc_q` | Ordinal | 1 (poorest 20%) to 5 (richest 20%) |
| `educ` | Categorical | Primary, Secondary, Tertiary |
| `employed` | Binary | 0 = Out of workforce, 1 = Employed |
| `rural` | Binary | 0 = Urban, 1 = Rural |

## Methodology

### Pipeline Steps (`findex_logistic_regression.py`)

1. **Data loading** — Reads the Findex microdata CSV, selects modelling variables
2. **Recoding** — Converts raw survey codes to interpretable binary/categorical values
3. **Missing data** — Listwise deletion (retains 98.9% of records)
4. **One-hot encoding** — Education dummies with "Primary" as reference category
5. **Multicollinearity check** — Variance Inflation Factor (VIF) for all predictors
6. **Train/test split** — 80/20 stratified split (n_train = 114,000, n_test = 28,500)
7. **Model fitting** — `statsmodels.Logit` for inference (coefficients, p-values, confidence intervals)
8. **Evaluation** — `sklearn.LogisticRegression` for accuracy, ROC-AUC, classification report
9. **Visualisation** — 4-panel figure saved as PNG

### The Logistic Model

The logit link function:

```
ln(p / (1-p)) = B0 + B1*age + B2*female + B3*inc_q + B4*employed + B5*rural + B6*educ_secondary + B7*educ_tertiary
```

Where `p` is the probability of having a financial account. Coefficients are exponentiated to produce **odds ratios** for interpretation.

## Key Results

| Predictor | Odds Ratio | Interpretation |
|---|---|---|
| Tertiary education | 12.06 | 12x more likely to have an account vs. primary |
| Secondary education | 3.39 | 3.4x more likely vs. primary |
| Employed | 1.93 | Nearly double the odds |
| Female | 1.23 | 23% higher odds (global model, after controls) |
| Rural | 1.20 | 20% higher odds (global model, after controls) |
| Income quintile (+1) | 1.12 | 12% increase per quintile step |
| Age (+1 year) | 1.02 | 2.3% increase per year |

**Model performance:** Accuracy = 75.8%, ROC-AUC = 0.755

See `findex_regression_report.md` for the full analysis and interpretation.

## Interactive Simulator

Open `logistic_regression_simulator.html` in any browser. It provides:

- **Coefficient sliders** — Adjust each weight and the intercept to see how they shift the probability curve
- **Sigmoid curve** — Shows P(Account) vs. Age for three income/education profiles
- **Income x Education chart** — Grouped bar chart of predicted probabilities
- **10 representative individuals** — Real-time probability and classification updates
- **Decision threshold** — Slide to explore precision/recall trade-offs
- **Live formula** — The logit equation updates as you adjust coefficients

Default values are pre-loaded from the actual Findex 2025 regression estimates.

## Dependencies

- Python 3.10+
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- statsmodels
