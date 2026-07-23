"""
========================================================================
MODULE: Advanced Models — LightGBM + Neural Network (MLP)
========================================================================
WHAT WE LEARN HERE:
  - LightGBM: a faster, more efficient gradient boosting framework
  - Neural Network (MLP): deep learning approach to fraud detection
  - When to use which model type

LIGHTGBM vs XGBOOST:
  Both are gradient boosting frameworks, but they differ in HOW they build trees:

  XGBoost (level-wise):
    → Grows trees level by level (all nodes at depth 1, then depth 2, etc.)
    → More balanced trees, slightly more robust to overfitting
    → Slower for large datasets

  LightGBM (leaf-wise):
    → Grows the leaf that reduces loss the most, regardless of level
    → Can create unbalanced trees that fit the data better
    → Much faster (histogram-based algorithm)
    → Lower memory usage

  In practice, LightGBM often matches or beats XGBoost while training
  2-5× faster. It's the go-to for production fraud detection.

NEURAL NETWORK (MLP) for FRAUD:
  A Multi-Layer Perceptron learns non-linear decision boundaries by
  stacking layers of neurons with non-linear activation functions.

  Architecture: Input → Hidden1(128) → Hidden2(64) → Hidden3(32) → Output

  Each neuron computes: output = activation(weights · inputs + bias)
  ReLU activation: output = max(0, x) — simple but effective

  WHY IT'S DIFFERENT FROM TREES:
  - Trees partition feature space into boxes (axis-aligned splits)
  - Neural nets create smooth, curved decision boundaries
  - NNs can learn feature interactions automatically
  - But they need more data and are harder to interpret
========================================================================
"""

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
from sklearn.neural_network import MLPClassifier

# Try to import LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠ LightGBM not installed. Run: pip install lightgbm")


def train_lightgbm(X_train, y_train, use_scale_pos=True):
    """
    LIGHTGBM — Fast Gradient Boosting

    KEY INNOVATIONS:
      1. Histogram-based splits: Instead of sorting all values for each
         feature, LightGBM bins values into histograms (256 bins).
         This is O(n) instead of O(n log n). Massive speedup.

      2. Leaf-wise growth: Grows the leaf with the highest gain first.
         This can create deep, specialized paths for rare patterns
         (like fraud) — perfect for our use case!

      3. Gradient-based One-Side Sampling (GOSS): Keeps all samples
         with large gradients (hard examples) and randomly samples
         small-gradient samples. Focuses training on what matters.

      4. Exclusive Feature Bundling (EFB): Bundles mutually exclusive
         features together to reduce dimensionality. Our V1-V28 PCA
         features benefit from this.

    SCALE_POS_WEIGHT:
      Same concept as XGBoost — weights fraud examples more heavily.
      Set to (# legit / # fraud) ≈ 577.

    WHY USE IT:
      - 2-5× faster than XGBoost on this dataset
      - Lower memory footprint
      - Often slightly better accuracy
      - Native support for categorical features (not needed here)
    """
    if not HAS_LIGHTGBM:
        print("  ⚠ Skipping LightGBM (not installed)")
        return None, None

    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    ratio = neg_count / max(pos_count, 1)

    config = {
        'n_estimators': 300,
        'max_depth': 8,
        'learning_rate': 0.05,
        'num_leaves': 63,           # Key LightGBM param (2^max_depth - 1)
        'min_child_samples': 20,    # Min samples in a leaf
        'subsample': 0.8,           # Row sampling per tree
        'colsample_bytree': 0.8,    # Column sampling per tree
        'reg_alpha': 0.1,           # L1 regularization
        'reg_lambda': 1.0,          # L2 regularization
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }

    if use_scale_pos:
        config['scale_pos_weight'] = ratio

    name = "LightGBM" + ("_weighted" if use_scale_pos else "_SMOTE")
    print(f"\n  Training {name}...")

    if use_scale_pos:
        print(f"    scale_pos_weight = {ratio:.1f}")

    model = lgb.LGBMClassifier(**config)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"    Done in {elapsed:.2f}s")

    # Feature importance
    feature_names = X_train.columns if hasattr(X_train, 'columns') else \
        [f'f{i}' for i in range(X_train.shape[1])]
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-5:][::-1]
    print("    Top 5 features by importance:")
    for idx in top_idx:
        print(f"      {feature_names[idx]}: {importances[idx]}")

    return model, name


def train_neural_network(X_train, y_train, use_class_weight=True):
    """
    NEURAL NETWORK (Multi-Layer Perceptron) for Fraud Detection

    ARCHITECTURE:
      Input (30 features)
        → Dense(128, ReLU)    # First hidden layer: learns basic patterns
        → Dense(64, ReLU)     # Second: combines basic patterns
        → Dense(32, ReLU)     # Third: high-level fraud features
        → Output(2, Softmax)  # Probability of legit vs fraud

    WHY THESE LAYER SIZES?
      - Start wide (128): learn many low-level feature interactions
      - Progressively narrow (64 → 32): compress into abstract representations
      - This "funnel" architecture forces the network to learn compact
        representations of fraud patterns

    KEY HYPERPARAMETERS:
      hidden_layer_sizes: (128, 64, 32) — neurons per layer
      activation: 'relu' — max(0, x), prevents vanishing gradients
      solver: 'adam' — adaptive learning rate optimizer (best for most cases)
      alpha: L2 regularization strength (prevents overfitting)
      batch_size: mini-batch gradient descent (256 samples per update)
      early_stopping: stops training when validation score plateaus

    CLASS IMBALANCE HANDLING:
      sklearn's MLP doesn't natively support class_weight, so we
      handle it through SMOTE or by adjusting the loss function via
      sample weights in the training data.

    LIMITATIONS VS TREES:
      - Needs feature scaling (already done in preprocessing ✓)
      - Harder to interpret ("black box")
      - Slower to train
      - More sensitive to hyperparameters
      - But can learn more complex patterns!
    """
    name = "NeuralNet" + ("_weighted" if use_class_weight else "_SMOTE")
    print(f"\n  Training {name}...")

    config = {
        'hidden_layer_sizes': (128, 64, 32),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.001,             # L2 regularization
        'batch_size': 256,
        'learning_rate': 'adaptive',  # Reduces learning rate on plateau
        'learning_rate_init': 0.001,
        'max_iter': 100,
        'early_stopping': True,     # Stop when validation score plateaus
        'validation_fraction': 0.1, # Use 10% of training for early stopping
        'n_iter_no_change': 10,     # Patience: stop after 10 epochs without improvement
        'random_state': 42,
        'verbose': False,
    }

    model = MLPClassifier(**config)
    start = time.time()

    # Keep DataFrame column names so downstream predict_proba(X_test) calls
    # (which pass DataFrames) don't emit feature-name-mismatch warnings.
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"    Done in {elapsed:.2f}s")
    print(f"    Architecture: {model.hidden_layer_sizes}")
    print(f"    Epochs trained: {model.n_iter_}")
    print(f"    Final loss: {model.loss_:.6f}")

    if model.early_stopping:
        print(f"    Best validation score: {model.best_validation_score_:.6f}")

    return model, name


def train_advanced_models(data):
    """
    Trains all advanced model variants:
      - LightGBM (weighted + SMOTE)
      - Neural Network (original + SMOTE)

    RETURNS: dict of {name: model}
    """
    print("\n" + "=" * 60)
    print("ADVANCED MODEL TRAINING")
    print("=" * 60)

    results = {}

    # ─── Approach 1: Class weights (original imbalanced data) ─────────
    print("\n▶ Approach 1: Class Weights / Scale_pos_weight")
    print("-" * 50)

    if HAS_LIGHTGBM:
        model, name = train_lightgbm(data['X_train'], data['y_train'], use_scale_pos=True)
        if model is not None:
            results[name] = model

    model, name = train_neural_network(data['X_train'], data['y_train'], use_class_weight=True)
    results[name] = model

    # ─── Approach 2: SMOTE ────────────────────────────────────────────
    print("\n▶ Approach 2: SMOTE")
    print("-" * 50)

    if HAS_LIGHTGBM:
        model, name = train_lightgbm(data['X_train_smote'], data['y_train_smote'], use_scale_pos=False)
        if model is not None:
            results[name] = model

    model, name = train_neural_network(data['X_train_smote'], data['y_train_smote'], use_class_weight=False)
    results[name] = model

    print(f"\n✅ Trained {len(results)} advanced models")
    return results


if __name__ == '__main__':
    from preprocess import preprocess
    data = preprocess()
    models = train_advanced_models(data)

    # Quick evaluation
    from evaluate import evaluate_model
    for name, model in models.items():
        metrics, _ = evaluate_model(model, data['X_test'], data['y_test'], name)
        print(f"\n{name}: F1={metrics['F1']:.4f}, AUPRC={metrics['AUPRC']:.4f}")
