"""
========================================================================
TOPICS 5, 6, 7: Model Training
========================================================================
WHAT WE LEARN HERE:
  - Logistic Regression: The baseline linear model
  - Random Forest: Ensemble of decision trees
  - XGBoost: Gradient boosted trees (state-of-the-art for tabular data)

EACH MODEL IS TRAINED TWO WAYS:
  1. With class_weight/scale_pos_weight (built-in imbalance handling)
  2. With SMOTE-resampled data (external imbalance handling)

This lets you compare which approach works better.
========================================================================
"""

import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_logistic_regression(X_train, y_train, use_class_weight=True):
    """
    TOPIC: Logistic Regression
    ----------------------------
    WHAT IT IS:
      A linear model that computes:  P(fraud) = sigmoid(w1*V1 + w2*V2 + ... + b)
      where sigmoid squashes any number to [0, 1].

    WHY USE IT:
      - Fast to train (seconds, not minutes)
      - Interpretable: the weights (coefficients) tell you which features
        matter and in which direction
      - Good baseline - if a complex model can't beat it, something's wrong

    CLASS_WEIGHT='balanced':
      Instead of SMOTE, we tell the model to weight fraud examples more.
      With 1:577 imbalance, each fraud example is treated as 577× more
      important than a legit one during training.

    KEY HYPERPARAMETERS:
      - C: inverse regularization strength (higher = less regularization)
      - max_iter: maximum iterations for convergence
      - solver: optimization algorithm
    """
    config = {
        'C': 1.0,
        'max_iter': 1000,
        'solver': 'lbfgs',
        'random_state': 42,
    }

    if use_class_weight:
        config['class_weight'] = 'balanced'

    name = "LogisticRegression" + ("_weighted" if use_class_weight else "_SMOTE")
    print(f"\n  Training {name}...")

    model = LogisticRegression(**config)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"    Done in {elapsed:.2f}s")
    print(f"    Coefficients shape: {model.coef_.shape}")

    # Show top features by absolute coefficient value
    feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'f{i}' for i in range(X_train.shape[1])]
    coef_abs = np.abs(model.coef_[0])
    top_idx = np.argsort(coef_abs)[-5:][::-1]
    print("    Top 5 features by |coefficient|:")
    for idx in top_idx:
        print(f"      {feature_names[idx]}: {model.coef_[0][idx]:.4f}")

    return model, name


def train_random_forest(X_train, y_train, use_class_weight=True):
    """
    TOPIC: Random Forest
    ----------------------
    WHAT IT IS:
      An ensemble of many decision trees. Each tree:
      1. Trains on a random bootstrap sample of the data
      2. At each split, considers a random subset of features
      3. Makes its own prediction

      The final prediction is the MAJORITY VOTE of all trees.

    WHY ENSEMBLE?
      A single tree overfits easily. By combining many trees trained on
      different random subsets, the errors average out. This is called
      "bagging" (Bootstrap AGGregatING).

    WHY USE IT FOR FRAUD:
      - Handles non-linear patterns (fraud might be "high V14 AND low V12")
      - Robust to outliers
      - Feature importance tells you what matters
      - No need for feature scaling (trees don't care about scale)

    KEY HYPERPARAMETERS:
      - n_estimators: number of trees (more = better, but slower)
      - max_depth: how deep each tree can grow (deeper = more complex)
      - min_samples_leaf: minimum samples in a leaf (prevents overfitting)
    """
    config = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_leaf': 5,
        'random_state': 42,
        'n_jobs': -1,
    }

    if use_class_weight:
        config['class_weight'] = 'balanced'

    name = "RandomForest" + ("_weighted" if use_class_weight else "_SMOTE")
    print(f"\n  Training {name}...")

    model = RandomForestClassifier(**config)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"    Done in {elapsed:.2f}s")

    # Feature importance
    feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'f{i}' for i in range(X_train.shape[1])]
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-5:][::-1]
    print("    Top 5 features by importance:")
    for idx in top_idx:
        print(f"      {feature_names[idx]}: {importances[idx]:.4f}")

    return model, name


def train_xgboost(X_train, y_train, use_scale_pos=True):
    """
    TOPIC: XGBoost (eXtreme Gradient Boosting)
    ---------------------------------------------
    WHAT IT IS:
      Builds trees SEQUENTIALLY. Each new tree focuses on correcting
      the mistakes of the previous trees.

      Round 1: Build tree_1, compute errors
      Round 2: Build tree_2 that targets those errors
      Round 3: Build tree_3 that targets remaining errors
      ...
      Final: Prediction = sum of all trees' predictions

    HOW IS IT DIFFERENT FROM RANDOM FOREST?
      - Random Forest: trees are INDEPENDENT (parallel), then vote
      - XGBoost: trees are SEQUENTIAL, each corrects previous mistakes
      - XGBoost usually gets better performance but can overfit more

    SCALE_POS_WEIGHT:
      XGBoost's built-in way to handle class imbalance.
      Set it to (# negative / # positive) ≈ 577.
      This tells XGBoost that misclassifying a fraud costs 577× more.

    WHY IT'S OFTEN THE BEST:
      - Regularization built-in (prevents overfitting)
      - Handles missing values automatically
      - Extremely optimized implementation (fast)
      - Dominant on Kaggle competitions for tabular data
    """
    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    ratio = neg_count / max(pos_count, 1)

    config = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'eval_metric': 'aucpr',
        'n_jobs': -1,
        'tree_method': 'hist',
    }

    if use_scale_pos:
        config['scale_pos_weight'] = ratio

    name = "XGBoost" + ("_weighted" if use_scale_pos else "_SMOTE")
    print(f"\n  Training {name}...")

    if use_scale_pos:
        print(f"    scale_pos_weight = {ratio:.1f} (each fraud counts as {ratio:.0f} legit)")

    model = XGBClassifier(**config)
    start = time.time()
    model.fit(X_train, y_train, verbose=False)
    elapsed = time.time() - start

    print(f"    Done in {elapsed:.2f}s")

    # Feature importance
    feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'f{i}' for i in range(X_train.shape[1])]
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-5:][::-1]
    print("    Top 5 features by importance:")
    for idx in top_idx:
        print(f"      {feature_names[idx]}: {importances[idx]:.4f}")

    return model, name


def train_all_models(data):
    """
    Trains all 6 model variants:
      - 3 models × 2 approaches (class_weight vs SMOTE)

    Returns a dict of {name: model}
    """
    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    results = {}

    # ─── Approach 1: Class weights (train on original imbalanced data) ─
    print("\n▶ Approach 1: Class Weights (built-in imbalance handling)")
    print("-" * 50)

    model, name = train_logistic_regression(data['X_train'], data['y_train'], use_class_weight=True)
    results[name] = model

    model, name = train_random_forest(data['X_train'], data['y_train'], use_class_weight=True)
    results[name] = model

    model, name = train_xgboost(data['X_train'], data['y_train'], use_scale_pos=True)
    results[name] = model

    # ─── Approach 2: SMOTE (train on resampled balanced data) ──────────
    print("\n▶ Approach 2: SMOTE (synthetic oversampling)")
    print("-" * 50)

    model, name = train_logistic_regression(data['X_train_smote'], data['y_train_smote'], use_class_weight=False)
    results[name] = model

    model, name = train_random_forest(data['X_train_smote'], data['y_train_smote'], use_class_weight=False)
    results[name] = model

    model, name = train_xgboost(data['X_train_smote'], data['y_train_smote'], use_scale_pos=False)
    results[name] = model

    print(f"\n✅ Trained {len(results)} models total")
    return results
