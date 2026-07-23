"""
========================================================================
MODULE: Feature Importance & SHAP Explainability
========================================================================
WHAT WE LEARN HERE:
  - Feature importance: WHICH features matter to the model?
  - SHAP values: WHY does the model make each specific prediction?
  - Global vs Local explanations:
      Global = "V14 is the most important feature overall"
      Local  = "For THIS transaction, V14=-5.2 pushed fraud probability up"

WHY IT MATTERS:
  In production, a fraud analyst can't just see "fraud detected."
  They need to know WHY the model flagged it. SHAP provides that.

  Regulators (like EU's GDPR) also require model explainability —
  you must be able to explain automated decisions to customers.

KEY CONCEPTS:
  - Shapley Values: From game theory. Each feature gets a "credit score"
    for how much it contributed to the prediction. All contributions
    sum to the difference between the prediction and the baseline.

  - SHAP (SHapley Additive exPlanations): An efficient algorithm to
    compute Shapley values for ML models. Uses tree-specific speedups
    for tree-based models (RandomForest, XGBoost, LightGBM).

  - Beeswarm Plot: Shows SHAP values for all samples at once.
    Each dot is one transaction. X-axis = SHAP value (impact on prediction).
    Color = feature value (red=high, blue=low).
========================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import os


def _select_fraud_class_shap(shap_values, n_features=None):
    """
    Reduce a SHAP result of any supported shape to a 2-D array
    (n_samples, n_features) holding the contributions for the fraud class (1).

    Handles list-of-arrays and 3-D ndarrays regardless of which axis holds
    the class dimension, so downstream plotting code can always assume 2-D.
    """
    # Older API: list [class_0_array, class_1_array]
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        # Find the axis that indexes classes (size 2) and take class 1 (fraud).
        if shap_values.shape[-1] == 2:            # (n_samples, n_features, 2)
            shap_values = shap_values[:, :, 1]
        elif shap_values.shape[0] == 2:           # (2, n_samples, n_features)
            shap_values = shap_values[1]
        else:
            # Fallback: collapse the last axis if it looks like the class axis.
            shap_values = shap_values[..., -1]

    return shap_values


def compute_shap_values(model, X_test, model_name, max_samples=500):
    """
    Compute SHAP values for a trained model.

    WHY max_samples=500?
      SHAP computation is expensive — O(n × features × 2^features) in theory.
      TreeExplainer is fast but still slow on 57,000 test samples.
      500 samples is enough to get reliable global importance estimates.

    WHAT ARE SHAP VALUES?
      For each prediction, each feature gets a SHAP value that tells you:
      - Positive SHAP → pushed prediction TOWARD fraud
      - Negative SHAP → pushed prediction AWAY from fraud
      - Magnitude = how MUCH it pushed

      Example: V14 = -5.2 has SHAP = +0.35
      → This very negative V14 value pushed the fraud probability up by 0.35
    """
    print(f"\n  Computing SHAP values for {model_name}...")

    # Subsample for speed
    if len(X_test) > max_samples:
        np.random.seed(42)
        idx = np.random.choice(len(X_test), max_samples, replace=False)
        X_sample = X_test.iloc[idx] if hasattr(X_test, 'iloc') else X_test[idx]
    else:
        X_sample = X_test

    # Choose the right explainer based on model type
    model_type = type(model).__name__

    if model_type in ['RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier']:
        # TreeExplainer is exact and fast for tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    else:
        # For non-tree models (LogisticRegression, NeuralNet), use KernelExplainer
        # KernelExplainer is model-agnostic but slower
        background = shap.sample(X_sample, min(50, len(X_sample)))
        explainer = shap.KernelExplainer(
            model.predict_proba,
            background
        )
        shap_values = explainer.shap_values(X_sample, nsamples=100)

    # Normalize the SHAP output to a 2-D array of shape (n_samples, n_features)
    # for the fraud class. Different SHAP versions / model types return the
    # binary-classification result in different shapes:
    #   - list [class_0_array, class_1_array]          (older API)
    #   - ndarray (n_samples, n_features, n_classes)   (newer sklearn trees)
    #   - ndarray (n_classes, n_samples, n_features)   (some explainers)
    #   - ndarray (n_samples, n_features)              (single-output, e.g. XGB)
    shap_values = _select_fraud_class_shap(shap_values)

    print(f"    SHAP values shape: {shap_values.shape}")
    print(f"    Computed on {len(X_sample)} samples")

    return shap_values, X_sample


def plot_global_importance(shap_values, X_sample, model_name, output_dir='outputs'):
    """
    GLOBAL IMPORTANCE: Which features matter MOST across all predictions?

    HOW IT WORKS:
      For each feature, compute mean(|SHAP value|) across all samples.
      Features with high mean |SHAP| consistently push predictions up or down.

    HOW TO READ:
      - V14 at the top → V14 is the most influential feature
      - Color shows direction: red dots = high feature value, blue = low
      - If V14's red dots are on the RIGHT (positive SHAP) and blue on LEFT,
        it means HIGH V14 → more fraud, LOW V14 → less fraud
    """
    print(f"\n  Plotting global feature importance for {model_name}...")

    feature_names = X_sample.columns if hasattr(X_sample, 'columns') else \
        [f'f{i}' for i in range(X_sample.shape[1])]

    # ─── Bar plot of mean |SHAP| ──────────────────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    top_n = 15
    top_idx = sorted_idx[:top_n]
    top_names = [feature_names[i] for i in top_idx]
    top_values = mean_abs_shap[top_idx]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: Bar chart
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, top_n))
    axes[0].barh(range(top_n), top_values[::-1], color=colors[::-1],
                        edgecolor='black', linewidth=0.3)
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(top_names[::-1], fontsize=10)
    axes[0].set_xlabel('Mean |SHAP Value|', fontsize=12)
    axes[0].set_title(f'Top {top_n} Features — {model_name}\n(Mean Absolute SHAP)',
                      fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')

    # Right: Beeswarm-style dot plot
    for i, feat_idx in enumerate(top_idx[:10]):
        feat_shap = shap_values[:, feat_idx]
        feat_vals = np.array(X_sample.iloc[:, feat_idx] if hasattr(X_sample, 'iloc')
                             else X_sample[:, feat_idx])

        # Normalize feature values for coloring
        vmin, vmax = feat_vals.min(), feat_vals.max()
        if vmax > vmin:
            norm_vals = (feat_vals - vmin) / (vmax - vmin)
        else:
            norm_vals = np.full_like(feat_vals, 0.5)

        y_pos = np.full_like(feat_shap, 9 - i) + np.random.uniform(-0.2, 0.2,
                                                                      size=len(feat_shap))

        axes[1].scatter(feat_shap, y_pos, c=norm_vals, cmap='coolwarm',
                        s=8, alpha=0.5, edgecolors='none')

    axes[1].set_yticks(range(10))
    axes[1].set_yticklabels([top_names[9 - i] for i in range(10)], fontsize=10)
    axes[1].set_xlabel('SHAP Value (impact on fraud prediction)', fontsize=12)
    axes[1].set_title('Feature Impact Distribution\n(Blue=Low value, Red=High value)',
                      fontsize=13, fontweight='bold')
    axes[1].axvline(x=0, color='black', linewidth=0.8, linestyle='--')
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '10_shap_global_importance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved: 10_shap_global_importance.png")

    # Print text summary
    print(f"\n  Global Feature Importance ({model_name}):")
    for name, val in zip(top_names[:10], top_values[:10]):
        print(f"    {name:>15s}: mean|SHAP| = {val:.4f}")


def explain_individual_predictions(shap_values, X_sample, y_test_sample,
                                   model, model_name, output_dir='outputs'):
    """
    LOCAL EXPLANATIONS: Why was THIS specific transaction flagged?

    THIS IS THE MOST POWERFUL PART.
    For a single transaction, we show:
    - The base prediction (average fraud probability)
    - Each feature's contribution (positive = toward fraud, negative = away)
    - The final prediction

    REAL-WORLD USE:
      Fraud analyst sees: "Transaction #47382 flagged as fraud"
      SHAP says: "V14=-5.2 (+0.35), V4=3.8 (+0.15), Amount=$1200 (+0.08)"
      → The analyst understands WHY and can make a judgment call.
    """
    print("\n  Generating individual prediction explanations...")

    # Find interesting cases: high-confidence fraud, high-confidence legit, borderline
    y_proba = model.predict_proba(X_sample)[:, 1]

    cases = {}

    # Highest fraud probability
    fraud_idx = np.argmax(y_proba)
    cases['High Confidence Fraud'] = fraud_idx

    # Lowest fraud probability
    legit_idx = np.argmin(y_proba)
    cases['High Confidence Legit'] = legit_idx

    # Most borderline (closest to 0.5)
    borderline_idx = np.argmin(np.abs(y_proba - 0.5))
    cases['Borderline Case'] = borderline_idx

    feature_names = list(X_sample.columns) if hasattr(X_sample, 'columns') else \
        [f'f{i}' for i in range(X_sample.shape[1])]

    fig, axes = plt.subplots(3, 1, figsize=(14, 18))

    for ax_idx, (case_name, sample_idx) in enumerate(cases.items()):
        ax = axes[ax_idx]

        # Get SHAP values for this sample
        sample_shap = shap_values[sample_idx]
        sample_values = np.array(X_sample.iloc[sample_idx] if hasattr(X_sample, 'iloc')
                                 else X_sample[sample_idx])
        prob = y_proba[sample_idx]

        # Sort by absolute SHAP value, show top 10
        sorted_idx = np.argsort(np.abs(sample_shap))[::-1][:10]
        top_shap = sample_shap[sorted_idx]
        top_names = [feature_names[i] for i in sorted_idx]
        top_feat_vals = sample_values[sorted_idx]

        # Waterfall-style horizontal bars
        colors_bar = ['#e74c3c' if s > 0 else '#2ecc71' for s in top_shap]
        ax.barh(range(10), top_shap[::-1], color=colors_bar[::-1],
                       edgecolor='black', linewidth=0.3)

        labels = [f"{n} = {v:.2f}" for n, v in zip(top_names, top_feat_vals)]
        ax.set_yticks(range(10))
        ax.set_yticklabels(labels[::-1], fontsize=9)
        ax.axvline(x=0, color='black', linewidth=1)
        ax.set_xlabel('SHAP Value (red=toward fraud, green=toward legit)')
        ax.set_title(f'{case_name} — P(fraud) = {prob:.4f}',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle(f'Individual Prediction Explanations — {model_name}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '11_shap_individual_explanations.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved: 11_shap_individual_explanations.png")


def plot_shap_dependence(shap_values, X_sample, model_name, output_dir='outputs'):
    """
    SHAP DEPENDENCE PLOTS: How does a feature's VALUE affect its SHAP impact?

    WHAT IT SHOWS:
      X-axis = actual feature value
      Y-axis = SHAP value (how much it pushes toward/away from fraud)
      Color  = interaction with another feature

    EXAMPLE:
      If V14's dependence plot shows a steep negative slope:
      → As V14 decreases, it pushes MORE toward fraud
      → Very negative V14 = strong fraud signal
    """
    print("\n  Plotting SHAP dependence plots...")

    feature_names = X_sample.columns if hasattr(X_sample, 'columns') else \
        [f'f{i}' for i in range(X_sample.shape[1])]

    # Find top 4 most important features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_4_idx = np.argsort(mean_abs_shap)[::-1][:4]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, feat_idx in enumerate(top_4_idx):
        ax = axes[i]
        feat_name = feature_names[feat_idx]

        feat_vals = np.array(X_sample.iloc[:, feat_idx] if hasattr(X_sample, 'iloc')
                             else X_sample[:, feat_idx])
        feat_shap = shap_values[:, feat_idx]

        scatter = ax.scatter(feat_vals, feat_shap, c=feat_shap, cmap='coolwarm',
                             s=10, alpha=0.6, edgecolors='none')
        ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel(f'{feat_name} value', fontsize=11)
        ax.set_ylabel(f'SHAP value for {feat_name}', fontsize=11)
        ax.set_title(f'{feat_name} Dependence', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='SHAP value')

    plt.suptitle(f'SHAP Dependence Plots — {model_name}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '12_shap_dependence.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved: 12_shap_dependence.png")


def run_feature_importance(best_model, best_model_name, data, output_dir='outputs'):
    """
    Full feature importance and explainability pipeline.

    Parameters:
      best_model: the trained model to explain
      best_model_name: name string
      data: dict from preprocess() containing X_test, y_test
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "█" * 60)
    print("█  FEATURE IMPORTANCE & SHAP EXPLAINABILITY")
    print("█" * 60)

    X_test = data['X_test']
    y_test = data['y_test']

    # Compute SHAP values
    shap_values, X_sample = compute_shap_values(
        best_model, X_test, best_model_name, max_samples=500
    )

    # Subsample y_test to match
    if len(X_test) > 500:
        np.random.seed(42)
        idx = np.random.choice(len(X_test), 500, replace=False)
        y_test_sample = y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]
    else:
        y_test_sample = y_test

    # Global importance
    plot_global_importance(shap_values, X_sample, best_model_name, output_dir)

    # Individual explanations
    explain_individual_predictions(
        shap_values, X_sample, y_test_sample,
        best_model, best_model_name, output_dir
    )

    # Dependence plots
    plot_shap_dependence(shap_values, X_sample, best_model_name, output_dir)

    print("\n✅ Feature importance analysis complete!")
    return shap_values, X_sample


if __name__ == '__main__':
    # Quick standalone test
    from preprocess import preprocess
    from models import train_random_forest

    data = preprocess()
    model, name = train_random_forest(data['X_train'], data['y_train'], use_class_weight=True)
    run_feature_importance(model, name, data)
