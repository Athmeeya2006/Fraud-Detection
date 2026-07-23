"""
========================================================================
TOPICS 2 & 8: Model Evaluation & Threshold Tuning
========================================================================
WHAT WE LEARN HERE:
  - Why accuracy is useless for imbalanced data
  - Precision, Recall, F1 — what they mean in fraud context
  - Confusion Matrix — visual breakdown of predictions
  - Precision-Recall Curve — the key curve for imbalanced problems
  - ROC Curve — overall model discrimination
  - AUPRC — the single best metric for imbalanced classification
  - Threshold tuning — finding the optimal cutoff

REAL-WORLD CONTEXT:
  A bank must decide: What's worse?
  - False Positive (flagging legit as fraud) → customer inconvenience
  - False Negative (missing actual fraud) → financial loss
  The answer determines your optimal threshold.
========================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, average_precision_score,
    roc_curve, roc_auc_score, f1_score,
    precision_score, recall_score
)
import os


def evaluate_model(model, X_test, y_test, model_name, output_dir='outputs'):
    """
    Comprehensive evaluation of a single model.

    Returns a dict of metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ─── CONFUSION MATRIX ─────────────────────────────────────────────
    """
    WHAT IS A CONFUSION MATRIX?

                        Predicted Legit    Predicted Fraud
    Actually Legit      TN (correct)       FP (false alarm)
    Actually Fraud      FN (missed!)       TP (caught!)

    For fraud detection:
    - TP (True Positive): Correctly caught fraud ✅
    - FP (False Positive): Flagged legit as fraud (annoys customer) ⚠️
    - FN (False Negative): Missed actual fraud (costs money!) ❌
    - TN (True Negative): Correctly identified legit ✅
    """
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # ─── COMPUTE ALL METRICS ──────────────────────────────────────────
    """
    METRIC EXPLANATIONS:

    Precision = TP / (TP + FP)
      "Of all transactions we flagged as fraud, how many were actually fraud?"
      High precision = fewer false alarms

    Recall = TP / (TP + FN)
      "Of all actual frauds, how many did we catch?"
      High recall = fewer missed frauds

    F1 = 2 × (Precision × Recall) / (Precision + Recall)
      Harmonic mean — balances precision and recall.
      A model with 100% precision but 1% recall gets a low F1.

    AUPRC = Area Under Precision-Recall Curve
      THE BEST METRIC for imbalanced data. It measures how well the
      model ranks fraud above legit across ALL thresholds.
      Random classifier AUPRC ≈ 0.0017 (the fraud rate).
      A perfect classifier AUPRC = 1.0.

    ROC-AUC = Area Under ROC Curve
      Measures how well the model separates classes.
      Can be misleadingly high with imbalanced data (always check AUPRC too).
    """
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auprc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    metrics = {
        'Model': model_name,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUPRC': auprc,
        'ROC-AUC': roc_auc,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
    }

    return metrics, y_proba


def plot_confusion_matrices(all_metrics, y_test, output_dir='outputs'):
    """Plot confusion matrices for all models."""
    n_models = len(all_metrics)
    cols = 3
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4.5*rows))
    if rows == 1:
        axes = [axes] if n_models == 1 else axes
    axes_flat = np.array(axes).flatten()

    for i, m in enumerate(all_metrics):
        ax = axes_flat[i]
        cm = np.array([[m['TN'], m['FP']], [m['FN'], m['TP']]])
        ax.imshow(cm, cmap='Blues', interpolation='nearest')

        ax.set_title(m['Model'], fontsize=10, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Legit', 'Fraud'])
        ax.set_yticklabels(['Legit', 'Fraud'])

        for row in range(2):
            for col in range(2):
                color = 'white' if cm[row, col] > cm.max()/2 else 'black'
                ax.text(col, row, f'{cm[row, col]:,}',
                        ha='center', va='center', fontsize=12,
                        fontweight='bold', color=color)

    for j in range(i+1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle('Confusion Matrices — All Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved: 06_confusion_matrices.png")


def plot_precision_recall_curves(all_probas, y_test, output_dir='outputs'):
    """
    TOPIC: Precision-Recall Curve
    ------------------------------
    WHAT IT SHOWS:
      For every possible threshold (0 to 1), it plots:
      - X-axis: Recall (how many frauds caught)
      - Y-axis: Precision (how many flagged were actually fraud)

    HOW TO READ IT:
      - Top-right corner = perfect (high precision AND high recall)
      - A curve that stays HIGH as recall increases = great model
      - The area under this curve (AUPRC) is the single best metric

    WHY NOT ROC?
      ROC looks at True Positive Rate vs False Positive Rate.
      With 99.83% legit, even a small FPR = tons of false alarms.
      PR curve is more informative for imbalanced problems.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Precision-Recall curves
    ax = axes[0]
    for name, y_proba in all_probas.items():
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax.plot(rec, prec, label=f'{name} (AUPRC={ap:.4f})', linewidth=1.5)

    baseline = y_test.mean()
    ax.axhline(y=baseline, color='gray', linestyle='--', label=f'Random (AUPRC={baseline:.4f})')
    ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax.set_xlabel('Recall (Fraud Caught)')
    ax.set_ylabel('Precision (Flag Accuracy)')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # ROC curves
    ax = axes[1]
    for name, y_proba in all_probas.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.4f})', linewidth=1.5)

    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random (AUC=0.5)')
    ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved: 07_curves.png")


def plot_threshold_analysis(best_model_name, best_proba, y_test, output_dir='outputs'):
    """
    TOPIC: Threshold Tuning
    -------------------------
    WHAT IT IS:
      By default, if P(fraud) > 0.5, we predict fraud.
      But 0.5 might not be optimal!

    THE BUSINESS TRADEOFF:
      - Lower threshold (e.g., 0.3): catch more fraud, but more false alarms
      - Higher threshold (e.g., 0.7): fewer false alarms, but miss more fraud

    HOW TO FIND THE OPTIMAL THRESHOLD:
      1. Compute precision, recall, F1 at every threshold
      2. Pick the threshold that maximizes F1 (balanced) or recall
         (if missing fraud is very costly)
    """
    thresholds = np.arange(0.05, 0.96, 0.01)
    precisions = []
    recalls = []
    f1s = []

    for t in thresholds:
        y_pred_t = (best_proba >= t).astype(int)
        p = precision_score(y_test, y_pred_t, zero_division=0)
        r = recall_score(y_test, y_pred_t, zero_division=0)
        f = f1_score(y_test, y_pred_t, zero_division=0)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    best_f1_idx = np.argmax(f1s)
    best_threshold = thresholds[best_f1_idx]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(thresholds, precisions, label='Precision', color='#3498db', linewidth=2)
    ax.plot(thresholds, recalls, label='Recall', color='#e74c3c', linewidth=2)
    ax.plot(thresholds, f1s, label='F1 Score', color='#2ecc71', linewidth=2)
    ax.axvline(x=best_threshold, color='purple', linestyle='--', linewidth=1.5,
               label=f'Best F1 threshold = {best_threshold:.2f}')
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1,
               label='Default threshold = 0.50')

    ax.set_title(f'Threshold Analysis — {best_model_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '08_threshold_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved: 08_threshold_analysis.png")

    # Report
    print(f"\n  THRESHOLD ANALYSIS for {best_model_name}:")
    print("    Default threshold (0.50):")
    default_idx = np.argmin(np.abs(thresholds - 0.5))
    print(f"      Precision: {precisions[default_idx]:.4f}")
    print(f"      Recall:    {recalls[default_idx]:.4f}")
    print(f"      F1:        {f1s[default_idx]:.4f}")
    print(f"\n    Optimal threshold ({best_threshold:.2f}):")
    print(f"      Precision: {precisions[best_f1_idx]:.4f}")
    print(f"      Recall:    {recalls[best_f1_idx]:.4f}")
    print(f"      F1:        {f1s[best_f1_idx]:.4f}")

    return best_threshold


def plot_model_comparison(all_metrics, output_dir='outputs'):
    """Creates a bar chart comparing all models across key metrics."""
    df = pd.DataFrame(all_metrics)
    metrics_to_plot = ['Precision', 'Recall', 'F1', 'AUPRC', 'ROC-AUC']

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(df))
    width = 0.15

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    for i, metric in enumerate(metrics_to_plot):
        bars = ax.bar(x + i * width, df[metric], width, label=metric,
                      color=colors[i], edgecolor='black', linewidth=0.3)
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_title('Model Comparison — All Metrics', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df['Model'], rotation=30, ha='right', fontsize=9)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '09_model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved: 09_model_comparison.png")


def evaluate_all(models, data, output_dir='outputs'):
    """
    Full evaluation pipeline for all models.
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    X_test = data['X_test']
    y_test = data['y_test']

    all_metrics = []
    all_probas = {}

    for name, model in models.items():
        print(f"\n--- {name} ---")
        metrics, y_proba = evaluate_model(model, X_test, y_test, name, output_dir)
        all_metrics.append(metrics)
        all_probas[name] = y_proba

        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall:    {metrics['Recall']:.4f}")
        print(f"  F1:        {metrics['F1']:.4f}")
        print(f"  AUPRC:     {metrics['AUPRC']:.4f}")
        print(f"  ROC-AUC:   {metrics['ROC-AUC']:.4f}")
        print(f"  Caught {metrics['TP']} / {metrics['TP'] + metrics['FN']} frauds "
              f"({metrics['TP']/(metrics['TP']+metrics['FN'])*100:.1f}%)")
        print(f"  False alarms: {metrics['FP']}")

    # ─── PLOTS ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GENERATING EVALUATION PLOTS")
    print("=" * 60)

    plot_confusion_matrices(all_metrics, y_test, output_dir)
    plot_precision_recall_curves(all_probas, y_test, output_dir)
    plot_model_comparison(all_metrics, output_dir)

    # ─── FIND BEST MODEL & THRESHOLD TUNE ─────────────────────────────
    best = max(all_metrics, key=lambda x: x['AUPRC'])
    print(f"\n🏆 Best model by AUPRC: {best['Model']} (AUPRC={best['AUPRC']:.4f})")

    optimal_threshold = plot_threshold_analysis(
        best['Model'], all_probas[best['Model']], y_test, output_dir
    )

    # ─── SUMMARY TABLE ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    df_results = pd.DataFrame(all_metrics)
    display_cols = ['Model', 'Precision', 'Recall', 'F1', 'AUPRC', 'ROC-AUC', 'TP', 'FN', 'FP']
    df_display = df_results[display_cols].sort_values('AUPRC', ascending=False)

    print(f"\n{df_display.to_string(index=False)}")

    df_display.to_csv(os.path.join(output_dir, 'results.csv'), index=False)
    print("\n  → Saved: results.csv")

    return all_metrics, optimal_threshold
