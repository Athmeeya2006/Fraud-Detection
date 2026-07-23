# Credit Card Fraud Detection

[![CI](https://github.com/Athmeeya2006/Fraud-Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Athmeeya2006/Fraud-Detection/actions/workflows/ci.yml)

A complete machine-learning system for detecting fraudulent credit-card
transactions. It takes the raw transaction data, cleans and prepares it,
trains and compares eight different models, evaluates them with the metrics
that actually matter for fraud, explains *why* the model flags a transaction,
translates the results into dollars saved, and simulates scoring live
transactions in real time.

Built on the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud):
**284,807 transactions, of which only 492 (0.17 %) are fraud.** That extreme
imbalance is the central challenge and shapes every decision in the pipeline.

---

## What the system does

The pipeline runs in ten stages. Each stage is a standalone Python module you
can run on its own, and [`run_pipeline.py`](run_pipeline.py) chains them all
together.

### 1. Exploratory data analysis — [`eda.py`](eda.py)
Loads the data and answers the basic questions before any modeling: How
imbalanced is it? Do fraudulent transactions have different amounts or happen
at different times? Which of the 28 anonymized features actually separate
fraud from legitimate activity? Produces five plots and a ranked list of the
most discriminative features.

### 2. Preprocessing — [`preprocess.py`](preprocess.py)
Prepares the data for modeling in three steps, in this specific order to avoid
cheating:
- **Split by time first.** The oldest 80 % of transactions become the training
  set, the newest 20 % the test set. This mirrors reality — you train on the
  past and predict the future.
- **Scale `Time` and `Amount`.** These two columns are on wildly different
  scales from the rest, so they get standardized. The scaler is fit on the
  **training set only** and then applied to the test set. Fitting it on all the
  data would leak information about the future into training and inflate the
  scores.
- **Balance the training data with SMOTE.** With only ~400 fraud examples,
  models struggle to learn the pattern. SMOTE synthesizes new fraud examples by
  interpolating between real ones. It is applied to the training set only — the
  test set stays untouched so the evaluation reflects real-world proportions.

### 3. Base models — [`models.py`](models.py)
Trains three algorithms — **Logistic Regression, Random Forest, XGBoost** —
each in two ways: once using class weights to handle the imbalance, and once on
the SMOTE-balanced data. Six models total, so you can compare which imbalance
strategy works better for each algorithm.

### 4. Advanced models — [`advanced_models.py`](advanced_models.py)
Adds two more model families: **LightGBM** (a faster gradient-boosting
framework) and a **neural network** (multi-layer perceptron). Same two
imbalance strategies as the base models.

### 5. Evaluation — [`evaluate.py`](evaluate.py)
Scores every model. **Accuracy is deliberately ignored** — a model that labels
everything "legitimate" would be 99.8 % accurate and catch zero fraud. Instead
it reports precision, recall, F1, ROC-AUC, and **AUPRC (area under the
precision-recall curve)**, which is the single most reliable metric for
imbalanced problems. Generates confusion matrices, PR/ROC curves, and a
threshold-tuning plot, then picks the best model by AUPRC.

### 6. Cross-validation — [`cross_validation.py`](cross_validation.py)
A single train/test split can be lucky or unlucky. This stage runs stratified
5-fold cross-validation (each fold keeps the same fraud ratio) and reports each
model's score as `mean ± std`. A low standard deviation means the model is
stable and you can trust its numbers.

### 7. Explainability — [`feature_importance.py`](feature_importance.py)
Uses **SHAP** to explain the model's decisions at two levels: globally (which
features matter most overall) and locally (for one specific transaction, which
feature values pushed it toward or away from "fraud"). This is what a fraud
analyst needs to act on a flag — and what regulators require for automated
decisions.

### 8. Anomaly detection — [`anomaly_detection.py`](anomaly_detection.py)
Detects fraud **without labels** using Isolation Forest and Local Outlier
Factor. Useful for catching brand-new fraud patterns that no labeled example
exists for yet, and as an independent second opinion alongside the supervised
models.

### 9. Cost analysis — [`cost_analysis.py`](cost_analysis.py)
Converts model performance into money. Each outcome has a real cost: a caught
fraud saves the transaction amount, a missed fraud loses it, a false alarm
costs an investigation fee. This stage finds the decision threshold that
**maximizes net dollars saved** — which is usually not the same threshold that
maximizes F1 — and reports the total savings versus having no fraud detection
at all.

### 10. Real-time simulation — [`realtime_simulator.py`](realtime_simulator.py)
Simulates a production scoring service: a stream of transactions arrives, each
is scored in milliseconds, and assigned a risk tier (BLOCK / FLAG / MONITOR /
APPROVE). Reports catch rate, false-alarm rate, and scoring latency
percentiles to show the model is fast enough for real use.

---

## Quick start

```bash
# Clone and enter the project
git clone https://github.com/Athmeeya2006/Fraud-Detection.git
cd Fraud-Detection

# Set up a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Get the dataset

The dataset (~150 MB) is **not** included in the repo. Download `creditcard.csv`
from Kaggle and place it in the project root:

```bash
# Requires a Kaggle API token at ~/.kaggle/kaggle.json
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d .
```

Or download it manually from
<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud> and unzip it here.

### Run it

```bash
# Run the full pipeline — all 10 stages, ~a few minutes
python run_pipeline.py

# Or run any single stage
python eda.py
python models.py
python evaluate.py
```

All plots and result CSVs are written to [`outputs/`](outputs/).

---

## Testing

The test suite runs against a small synthetic dataset generated on the fly (see
[`tests/conftest.py`](tests/conftest.py)), so it needs no Kaggle download and
finishes in about 20 seconds while exercising every module.

```bash
pip install -r requirements-dev.txt
pytest              # run all tests
ruff check .        # lint
```

CI runs the linter, an import check, and the full test suite on Python 3.10,
3.11, and 3.12 for every push and pull request — see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## Key design decisions

- **No data leakage.** The scaler is fit on the training split only;
  cross-validation refits it fresh inside each fold; SMOTE touches training
  data only. The test set is never seen during any fitting step.
- **AUPRC, not accuracy.** At 0.17 % fraud, accuracy is misleading. Models are
  selected by area under the precision-recall curve.
- **Thresholds tuned for money, not just F1.** The optimal cutoff depends on
  transaction amounts and investigation costs, so it is chosen to maximize net
  savings.
- **Every prediction is explainable.** SHAP shows the reason behind each flag,
  which is required for both analyst review and compliance.

---

## Project structure

```
Fraud-Detection/
├── eda.py                  # 1  exploratory data analysis
├── preprocess.py           # 2  split, scale, SMOTE
├── models.py               # 3  logistic regression, random forest, XGBoost
├── advanced_models.py      # 4  LightGBM, neural network
├── evaluate.py             # 5  metrics, curves, threshold tuning
├── cross_validation.py     # 6  stratified k-fold stability
├── feature_importance.py   # 7  SHAP explainability
├── anomaly_detection.py    # 8  Isolation Forest, LOF (unsupervised)
├── cost_analysis.py        # 9  dollar impact, cost-optimal threshold
├── realtime_simulator.py   # 10 real-time scoring simulation
├── run_pipeline.py         #    runs all stages end to end
├── tests/                  #    pytest suite (synthetic data)
├── outputs/                #    generated plots and CSVs (git-ignored)
├── requirements.txt        #    runtime dependencies
├── requirements-dev.txt    #    + pytest, ruff
├── pyproject.toml          #    ruff and pytest configuration
└── .github/workflows/ci.yml
```
