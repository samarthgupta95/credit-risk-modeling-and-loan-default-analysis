# Credit Risk Modeling & Loan Default Analysis

## 📌 Project Overview

This project builds an end-to-end automated credit risk scoring system for a fintech lending startup serving borrowers with limited or no formal credit history. Using 307,511 Home Credit loan applications, the analysis maps which applicant characteristics predict default, quantifies the financial cost of misclassification, and delivers a deployable three-tier risk framework — Low Risk, Medium Risk, and High Risk — with a threshold-tuned XGBoost champion model and full SHAP interpretability. The goal: replace inconsistent manual underwriting with a scoring system that is faster, more consistent, and defensible to regulators and senior leadership.

---

## 🎯 Business Problem

A fintech lender operating in the underserved borrower segment faces a dual failure: high-risk applicants slip through manual underwriting and default, generating direct financial losses; and creditworthy applicants are rejected because their profiles don't fit traditional scoring models, destroying revenue at the same time. With over 300,000 applications on record, the credit team needed to understand which applicant characteristics actually predict default, where risk is being mispriced, and how to make lending decisions that are faster, more consistent, and defensible.

**Questions from stakeholders:**
1. Which applicant characteristics are the strongest predictors of loan default?
2. How accurately can historical application data predict which applicants will default?
3. At what probability threshold should an applicant be classified as high-risk — and what is the financial trade-off of that decision?
4. How should applicants be segmented into approval tiers, and what is the expected default rate within each tier?
5. Which applicants are being incorrectly assessed — and where is risk being mispriced?

---

## 📂 Dataset & Scope

- **Source:** [Home Credit Default Risk — Kaggle](https://www.kaggle.com/c/home-credit-default-risk)
- **Raw dataset:** 307,511 rows × 122 columns (application_train.csv only)
- **Cleaned dataset:** 307,511 rows × 54 columns
- **Target variable:** 0 = Repaid (91.94%) | 1 = Defaulted (8.06%) — ~11:1 class imbalance
- **Domain:** Consumer finance / fintech lending

---

## 🧠 Approach

### 1. Plan Stage — Data Cleaning & Preparation
- Dropped SK_ID_CURR (identifier), 19 FLAG_DOCUMENT columns, and ~45 building/property measurement columns (48–70% missing, no credit signal)
- Replaced 365,243 sentinel values in `days_employed` with NaN prior to conversion
- Converted all negative day columns to positive year-based metrics and renamed for interpretability
- Imputed numerical columns with median, categorical columns with mode — zero nulls confirmed post-imputation
- Renamed all columns to snake_case

### 2. Analyse Stage — Exploratory Data Analysis
- Confirmed ~11:1 class imbalance — established ROC-AUC and Recall as primary metrics over accuracy
- EXT_SOURCE_2 and EXT_SOURCE_3 identified as strongest numerical predictors (−0.16 correlation with target each)
- Low-skill laborers default at 17.15%, academic degree holders at 1.83% — occupation and education carry stronger signal than raw income
- Raw financial amounts (income, credit, annuity) show near-zero correlation with target — signal is in ratios, not amounts
- Winsorization applied at 1st and 99th percentile for four skewed financial columns

### 3. Construct Stage — Feature Engineering & Modeling
- Engineered 6 new features: `credit_income_ratio`, `annuity_income_ratio`, `credit_goods_ratio`, `income_per_person`, `ext_source_mean`, `ext_source_min`
- Label Encoding for binary categoricals, One-Hot Encoding (drop_first=True) for multi-class — final feature space: 159 columns
- Stratified 64/16/20 train/validation/test split preserving 8.06% default rate across all three sets
- StandardScaler fit on training data only — no leakage
- SMOTE applied exclusively on X_train_scaled — validation and test sets retain real-world class distribution
- Four models trained and tuned: Logistic Regression (GridSearchCV), Decision Tree, Random Forest, XGBoost (RandomizedSearchCV) — all with StratifiedKFold k=5

### 4. Execute Stage — Evaluation & Deployment Framework
- Champion model evaluated on fully held-out test set — AUC 0.7536, Recall 0.3992, F1 0.2998, AUC drift −0.004 (no overfitting)
- Threshold tuning across 0.30–0.50 — optimal operating point identified at 0.40
- Three-tier risk segmentation deployed on test set
- SHAP TreeExplainer used for global (beeswarm + bar) and local (waterfall) interpretability

---

## 🔍 Key Findings

| Metric | Value | Context |
|--------|-------|---------|
| Test AUC | 0.7536 | Strong discrimination across all thresholds |
| Test Recall | 39.92% | 2 in every 5 actual defaulters correctly identified |
| Test F1 | 0.2998 | Best precision-recall balance among all 4 models |
| AUC Drift (Val → Test) | −0.004 | Zero overfitting — model generalises cleanly |
| Class Imbalance | ~11:1 | Repaid vs Defaulted |
| Low Risk Default Rate | 2.20% | 33.14% of portfolio — Auto-approve |
| Medium Risk Default Rate | 6.57% | 44.06% of portfolio — Manual review |
| High Risk Default Rate | 19.52% | 22.80% of portfolio — Auto-decline |
| Default Rate Gap | 9× | Between Low Risk and High Risk tiers |
| Top SHAP Predictor | ext_source_mean | Dominant by impact magnitude across test set |

---

## ⚡ Strategic Recommendation Snapshot

- **Auto-approve Low Risk** — 33.14% of the portfolio defaults at just 2.20%. Eliminate manual underwriting cost for this segment entirely.
- **Human-in-the-loop for Medium Risk** — Largest segment at 44.06%. Surface model score and top SHAP drivers to underwriters for defensible, differentiated decisions.
- **Auto-decline High Risk** — 19.52% actual default rate — more than double the portfolio average. Threshold set at 0.40 based on F1-optimal tuning.
- **Apply occupational risk premia** — Low-skill laborers (17.15%), drivers (11.33%), and unemployed applicants (36.36%) all significantly above the 8.06% portfolio average.
- **Fix EXT_SOURCE data pipeline** — EXT_SOURCE_1 was missing for 56.38% of applicants at ingestion. The model's strongest signals are also its most incomplete — fixing the data feed directly improves model discrimination with zero architectural change.

*Full strategic breakdown with segment-level numbers and recommended steps available in the project notebook.*
---

## 🗂️ Project Assets

- 📓 **Notebook:** Full PACE-structured analysis with code, outputs, observations, and stage summaries
- 📊 **Executive Presentation:** 12-slide business storytelling deck — findings, risk tiers, SHAP interpretability, and recommendations

---

## 🛠️ Tools & Technologies

- **Python** (Pandas, NumPy, Matplotlib, Seaborn) — Data cleaning, EDA, visualisation, PACE notebook
- **Scikit-learn** — Logistic Regression, Decision Tree, Random Forest, preprocessing, model evaluation
- **XGBoost** — Champion model
- **Imbalanced-learn** — SMOTE for class imbalance handling
- **SHAP** — Model interpretability: TreeExplainer, summary plots, waterfall plots
- **Jupyter Notebook** — Full PACE-structured analysis

---

## 🚀 Final Takeaway

Default risk in this portfolio is not randomly distributed. It concentrates in identifiable applicant profiles — weak external credit scores, high loan-to-goods ratios, specific occupational and demographic segments — that the current manual underwriting system is systematically missing. The model prices risk the way a credit analyst would, but at scale and in milliseconds.
