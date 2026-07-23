# Credit Card Fraud Detection

[![CI](https://github.com/Athmeeya2006/Fraud-Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Athmeeya2006/Fraud-Detection/actions/workflows/ci.yml)

An end-to-end, heavily-commented machine-learning pipeline for detecting
fraudulent credit-card transactions on the classic
[Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(284,807 transactions, 0.17 % fraud). It is built as a **teaching project**:
every module explains the *why* behind each decision, not just the *how*.

The pipeline covers the full lifecycle — EDA, leakage-safe preprocessing,
supervised and unsupervised modeling, evaluation, explainability, business
cost analysis, and a real-time scoring simulation.

---

## Pipeline phases

| # | Phase | Module | What it does |
|---|-------|--------|--------------|
| 1 | EDA | [`eda.py`](eda.py) | Class imbalance, amount/time patterns, feature separability |
| 2 | Preprocessing | [`preprocess.py`](preprocess.py) | Temporal split → **leakage-safe scaling** → SMOTE (train only) |
| 3 | Base models | [`models.py`](models.py) | Logistic Regression, Random Forest, XGBoost (weighted + SMOTE) |
| 4 | Advanced models | [`advanced_models.py`](advanced_models.py) | LightGBM, Neural Network (MLP) |
| 5 | Evaluation | [`evaluate.py`](evaluate.py) | Precision/Recall/F1/AUPRC/ROC-AUC, curves, threshold tuning |
| 6 | Cross-validation | [`cross_validation.py`](cross_validation.py) | Stratified K-fold stability (`mean ± std`) |
| 7 | Explainability | [`feature_importance.py`](feature_importance.py) | SHAP global + local + dependence plots |
| 8 | Anomaly detection | [`anomaly_detection.py`](anomaly_detection.py) | Isolation Forest & LOF (unsupervised) |
| 9 | Cost analysis | [`cost_analysis.py`](cost_analysis.py) | Dollar impact + cost-optimal threshold |
| 10 | Real-time sim | [`realtime_simulator.py`](realtime_simulator.py) | Streaming scoring, risk tiers, latency |

Orchestrated end-to-end by [`run_pipeline.py`](run_pipeline.py).

---

## Setup

```bash
# 1. Clone
git clone https://github.com/Athmeeya2006/Fraud-Detection.git
cd Fraud-Detection

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Get the dataset

The dataset (~150 MB) is **not** committed to the repo. Download
`creditcard.csv` from Kaggle and place it in the project root:

```bash
# Requires a configured Kaggle API token (~/.kaggle/kaggle.json)
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d .
```

Or download it manually from
<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud> and unzip it here.

---

## Usage

Run the entire pipeline (all 10 phases, plots saved to `outputs/`):

```bash
python run_pipeline.py
```

Or run any phase on its own — each module is runnable and self-contained:

```bash
python eda.py
python preprocess.py
python models.py
python evaluate.py          # etc.
```

All figures and result CSVs are written to [`outputs/`](outputs/).

---

## Testing

The test suite runs against a small **synthetic dataset** (generated in
[`tests/conftest.py`](tests/conftest.py)) with the same schema as the real
data, so it needs no Kaggle download and finishes in ~20 s while still
exercising every real code path.

```bash
pip install -r requirements-dev.txt
pytest              # run all tests
ruff check .        # lint
```

Continuous integration runs lint + import checks + the full test suite on
Python 3.10/3.11/3.12 via [GitHub Actions](.github/workflows/ci.yml).

---

## Design notes

- **No data leakage.** The `StandardScaler` is fit on the **training split
  only** and then applied to the test set; cross-validation refits the scaler
  fresh inside every fold. SMOTE is applied to training data exclusively.
- **AUPRC over accuracy.** With 0.17 % fraud, accuracy is meaningless. The
  pipeline selects the best model by **Area Under the Precision-Recall Curve**.
- **Business-aware thresholds.** [`cost_analysis.py`](cost_analysis.py) tunes
  the decision threshold to maximize net dollar savings, not just F1.
- **Explainability.** SHAP values explain both global feature importance and
  individual predictions — required for real-world fraud review and regulation.

---

## Project structure

```
Fraud-Detection/
├── eda.py                  # Phase 1  — exploratory data analysis
├── preprocess.py           # Phase 2  — scaling, split, SMOTE
├── models.py               # Phase 3  — base models
├── advanced_models.py      # Phase 4  — LightGBM + neural net
├── evaluate.py             # Phase 5  — metrics & threshold tuning
├── cross_validation.py     # Phase 6  — stratified K-fold
├── feature_importance.py   # Phase 7  — SHAP explainability
├── anomaly_detection.py    # Phase 8  — Isolation Forest + LOF
├── cost_analysis.py        # Phase 9  — cost-sensitive analysis
├── realtime_simulator.py   # Phase 10 — real-time scoring sim
├── run_pipeline.py         # runs all phases end-to-end
├── tests/                  # pytest suite (synthetic data)
├── outputs/                # generated plots & CSVs (git-ignored)
├── requirements.txt        # runtime dependencies
├── requirements-dev.txt    # + pytest, ruff
└── .github/workflows/ci.yml
```
