# 💳 Indian Credit Risk Copilot

**ML + Explainable AI + GenAI Decision Support System**

A production-style, portfolio-quality credit-risk analytics application for
Indian retail lending contexts. It estimates probability of default,
explains predictions with SHAP, lets you bring your own dataset, assesses
hypothetical new customers, and includes an AI Copilot grounded in
calculated facts (not hallucinated numbers).

> ⚠️ **This is a decision-support and analytics tool only.** It never
> automatically approves or rejects a loan, does not access CIBIL or any
> credit bureau, and is not RBI-approved. See [Responsible AI &
> Limitations](#responsible-ai--limitations) below.

---

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Dataset Support](#dataset-support)
4. [Indian Context](#indian-context)
5. [Installation](#installation)
6. [Running the Application](#running-the-application)
7. [Dataset Upload Guide](#dataset-upload-guide)
8. [Model Training Guide](#model-training-guide)
9. [GenAI / Groq Setup](#genai--groq-setup)
10. [Testing](#testing)
11. [Limitations](#limitations)
12. [Responsible AI & Limitations](#responsible-ai--limitations)
13. [Future Improvements](#future-improvements)

---

## Features

- **Three usage modes**: built-in demo dataset, upload your own CSV/Excel, or manually assess a new customer via a dynamic form.
- **One central schema** (`src/config.py`) drives training, prediction, column mapping, the manual form, and validation — no duplicated feature lists.
- **Single sklearn Pipeline** (feature engineering → preprocessing → model) used identically for every prediction path, eliminating training/inference skew.
- **Automatic, user-confirmed column mapping** (exact / alias / fuzzy matching via RapidFuzz) so uploaded datasets don't need specific column names.
- **Target detection**: datasets with an outcome column can train a new model; datasets without one are scored with the built-in model or profiled via segmentation — never silently faked.
- **Leakage & PII detection**: flags post-outcome columns and personally identifiable/sensitive-attribute columns before training.
- **Explainability**: SHAP for tree models, with automatic fallback to coefficients/permutation importance.
- **Out-of-distribution detection & confidence scoring** for every prediction.
- **Portfolio analytics dashboard** with KPIs, risk distributions, and a searchable customer explorer.
- **AI Risk Copilot** (Groq + Llama) that only narrates pre-calculated, structured facts — it never invents metrics, SHAP values, or bureau access, and raw uploaded cell values are never inserted into prompts as free text (prompt-injection defense).
- **Responsible AI guardrails**: sensitive attributes (religion, caste, race, etc.) are never used as default features; PII columns are flagged for exclusion.

## Architecture

```
credit-risk-copilot/
├── app.py                       # Streamlit entry point, page router
├── data/sample/credit_data.csv  # Built-in SYNTHETIC demo dataset
├── artifacts/                   # Saved model + metadata (generated)
├── scripts/
│   ├── generate_sample_data.py  # Builds the synthetic demo dataset
│   └── train_default_model.py   # Trains & saves the built-in model
├── src/
│   ├── config.py                 # ⭐ Central schema (single source of truth)
│   ├── data/
│   │   ├── loader.py              # CSV/Excel loading, error handling
│   │   ├── profiler.py            # Missingness, dtypes, describe, imbalance
│   │   ├── column_mapper.py       # Exact/alias/fuzzy column mapping
│   │   ├── leakage_detector.py    # Keyword + correlation leakage checks
│   │   └── pii_detector.py        # PII / sensitive-attribute detection
│   ├── data_adapters/
│   │   ├── base.py
│   │   ├── generic_adapter.py     # Any uploaded dataset -> canonical schema
│   │   └── public_dataset_adapter.py  # Loads the built-in demo dataset
│   ├── modeling/
│   │   ├── feature_engineering.py # sklearn-compatible ratio features
│   │   ├── pipeline_builder.py    # ONE pipeline for train + predict
│   │   ├── train.py               # Training orchestration + checks
│   │   ├── evaluate.py            # Metrics, threshold optimization
│   │   ├── ood.py                 # Out-of-distribution + confidence
│   │   └── predict.py             # Single prediction fn for all sources
│   ├── explainability/explain.py  # SHAP + fallback explanations
│   ├── analytics/
│   │   ├── portfolio.py           # Portfolio KPIs & breakdowns
│   │   └── customer.py            # Customer profile & comparisons
│   ├── genai/
│   │   ├── tools.py                # Grounded "facts" functions for the LLM
│   │   ├── prompts.py              # System prompt + injection defense
│   │   └── copilot.py              # Intent routing + Groq call
│   ├── ui/                         # One module per Streamlit page
│   └── utils/validators.py         # Input validation against the schema
└── tests/                          # Unit + integration tests
```

### Data flow

```
Raw file (CSV/XLSX)
   -> loader.py           (parse, basic validation)
   -> profiler.py         (missingness, dtypes, describe, duplicates)
   -> pii_detector.py + leakage_detector.py   (safety flags)
   -> column_mapper.py    (suggest mappings; user confirms/edits)
   -> generic_adapter.py  (apply mapping -> canonical schema + unit derivations)
   -> [Type A: has target]  train.py -> pipeline_builder.py -> evaluate.py
   -> [Type B: no target]   predict.py (score w/ built-in model) OR segmentation
   -> predict.py / explain.py -> UI pages
   -> genai/tools.py (aggregate facts) -> genai/copilot.py -> Groq -> chat UI
```

### ML pipeline

Every prediction — built-in dataset row, uploaded dataset row, or a
manually-entered new customer — goes through the **same fitted
`sklearn.Pipeline`**:

```
FinancialFeatureEngineer()      # loan_to_income, EMI, DTI, etc. (robust to missing cols)
        -> ColumnTransformer     # median-impute+scale numeric, mode-impute+one-hot categorical
        -> Classifier            # Logistic Regression / Random Forest / (XGBoost/LightGBM/CatBoost if installed)
```

This is what `tests/test_predict.py::test_predict_single_matches_batch_for_same_row`
verifies directly: scoring the same customer through the single-prediction
path and the batch path produces bit-for-bit the same probability.

### How datasets without a target are handled

If no column maps to the outcome field, the app explicitly tells the user:
*"This dataset does not contain an observed repayment/default outcome, so a
supervised credit-default model cannot be trained from it."* It then offers
two clearly-labeled, non-deceptive options:

1. **Score with the pre-trained model** — uses the built-in model's
   probabilities: these ARE genuine model predictions, just not derived
   from this dataset's own (nonexistent) outcome labels.
2. **Segmentation / risk-proxy analysis** — KMeans clustering + outlier
   detection over numeric fields, explicitly labeled as *not* default
   predictions.

### How the GenAI Copilot is grounded

```
User question -> heuristic intent detection -> Python analytics function
(src/genai/tools.py, wrapping analytics/portfolio.py, analytics/customer.py,
explainability/explain.py, data/profiler.py) -> structured JSON facts
-> inserted into the Groq prompt as DATA (never as instructions)
-> LLM narrates the facts, restricted by a system prompt that forbids
   inventing numbers, claiming bureau access, or recommending approval/rejection.
```

Uploaded dataset cell values are only ever summarized into aggregates
(counts, percentages, column names) before reaching the LLM — never
inserted as raw free text — which defends against prompt injection via
malicious CSV content.

## Dataset Support

- **File types**: CSV, XLSX, XLS.
- **Column names**: flexible. `AnnualIncome`, `annual_inc`, `income`,
  `ApplicantIncome` all map to the canonical `annual_income` field via
  exact / case-insensitive / alias / fuzzy matching — you always confirm or
  correct the mapping before anything is used.
- **With a target column** (`default`, `loan_status`, `bad_loan`, etc.):
  trains and evaluates a new model (ROC-AUC, accuracy, precision, recall,
  F1, confusion matrix).
- **Without a target column**: scored with the built-in model, or profiled
  via segmentation — never silently trained as if labels existed.

### Built-in demonstration dataset

`data/sample/credit_data.csv` is **synthetically generated**
(`scripts/generate_sample_data.py`) with causally sensible relationships
(income, loan-to-income, EMI-to-income, credit utilization, delinquency
history driving default risk) so the ML pipeline, SHAP explanations, and
dashboards behave realistically. **It is not real customer data and is not
sourced from any credit bureau** — this is disclosed in-app on the Home and
Model Information pages, and tracked via `data_source_type: "synthetic"`
in every model's metadata.

## Indian Context

The UI uses Indian financial terminology throughout (₹ amounts, EMI,
Loan Tenure in months, Employment Type categories relevant to Indian
lending, Loan Purpose categories, Residential Status). The application does
**not** claim access to CIBIL, Experian India, Equifax India, CRIF High
Mark, PAN-linked income data, GST turnover, or bank statement data — see
[Responsible AI & Limitations](#responsible-ai--limitations).

## Installation

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it
#    macOS/Linux:
source venv/bin/activate
#    Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install -r requirements.txt
```

## Running the Application

```bash
# (Re)generate the synthetic demo dataset and built-in model, if needed:
python scripts/generate_sample_data.py
python scripts/train_default_model.py

# Launch the app
python -m streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

## Dataset Upload Guide

1. Go to **📤 Upload Dataset** and upload a CSV/XLSX file.
2. Review the preview, dimensions, dtypes, and data-quality warnings.
3. Review flagged PII / sensitive-attribute / leakage columns.
4. Confirm or edit the suggested column mapping (a dropdown per column).
5. If a target column was mapped:
   - If its values aren't already `0`/`1`, select which value(s) represent
     **default**.
   - Choose a model, threshold strategy, and any columns to exclude, then
     **Train Model**.
   - Review ROC-AUC / accuracy / precision / recall / F1 / confusion
     matrix, then **Set as Active Model** to use it across the app.
6. If no target column was found, choose to score with the built-in model
   or run segmentation/outlier analysis instead.

## Model Training Guide

- Models available out of the box: **Logistic Regression**, **Random
  Forest**. XGBoost/LightGBM/CatBoost are auto-detected and added to the
  dropdown if installed (`pip install xgboost lightgbm catboost`).
- The classification threshold is **optimized on validation data** (F1 by
  default; recall- or precision-focused strategies are also available) —
  it is a modeling choice, not a lending rule, and is always displayed.
- Small datasets (<200 rows) and severe class imbalance trigger explicit
  warnings rather than silently producing unreliable metrics.

## GenAI / Groq Setup

1. Get a free API key at <https://console.groq.com/keys>.
2. Copy `.env.example` to `.env` and set `GROQ_API_KEY`.
3. Restart the app. The **🤖 AI Risk Copilot** page will now respond;
   without a key, the rest of the application still works normally.

## Testing

```bash
python -m pytest tests/ -v
```

Covers feature engineering (including missing-column and divide-by-zero
robustness), column mapping, input validation, leakage/PII detection, and
an end-to-end guarantee that single-customer and batch predictions agree
(no training/inference skew).

## Limitations

- The built-in dataset is **synthetic**, not real Indian bureau data.
- SHAP TreeExplainer is used for tree-based models; linear models fall back
  to a linear SHAP explainer, and any SHAP failure falls back further to
  model coefficients or permutation importance.
- Fuzzy column matching can occasionally suggest an incorrect mapping for
  very short or ambiguous column names — always review suggestions.
- The AI Copilot's intent detection is heuristic (keyword-based), not a
  full NLU system; it will sometimes route to the closest-matching
  analytics tool rather than a perfectly disambiguated one.
- This is a single-session application: uploaded data and user-trained
  models are **not persisted** between restarts by design (see Privacy
  Notice in-app).

## Responsible AI & Limitations

> This application is an educational and analytical demonstration. It is
> not a regulated credit scoring system, does not access CIBIL, Experian
> India, Equifax India, CRIF High Mark, or any other credit bureau, bank,
> PAN, Aadhaar, or GST system, and should not be used as the sole basis for
> lending decisions. **Production deployment would require legal,
> regulatory, privacy, security, validation, and governance review.**

- The model may reflect bias present in its training data (synthetic or
  user-uploaded).
- Sensitive attributes (religion, caste, race, ethnicity, political
  affiliation, gender, etc.) are **never used as default model features**;
  if present in an upload, they are flagged and must be explicitly
  excluded by the user.
- Predictions are **probabilities**, not certainties — every prediction
  page displays a confidence indicator reflecting similarity to training
  data, not certainty about individual outcomes.
- Uploaded data is processed **in-session only** and is not permanently
  stored by default; do not upload PAN, Aadhaar, bank account, card
  numbers, passwords, or OTPs.

## Future Improvements

- Persist user-trained models across sessions (with explicit opt-in
  storage and encryption).
- Integrate real, licensed Indian credit datasets where legally available.
- Add fairness metrics (e.g. disparate impact ratio) computed across
  user-designated demographic groups, shown only when the user explicitly
  supplies such a column and opts in.
- Support batch scoring exports (CSV/PDF report generation).
- Expand the GenAI Copilot's intent detection with a proper classifier or
  function-calling schema instead of keyword heuristics.
