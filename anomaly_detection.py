"""
========================================================================
MODULE: Anomaly Detection (Unsupervised Fraud Detection)
========================================================================
WHAT WE LEARN HERE:
  - Unsupervised learning: detecting fraud WITHOUT labeled data
  - Isolation Forest: how it isolates anomalies using random trees
  - Comparison: supervised vs unsupervised approaches

WHY UNSUPERVISED MATTERS:
  In the real world, you often DON'T have labeled fraud data:
    - New fraud patterns haven't been reported yet
    - Labeling is expensive (requires manual investigation)
    - Zero-day fraud: completely new attack vectors

  Unsupervised methods detect "unusual" transactions even if they've
  never seen that type of fraud before.

KEY CONCEPTS:
  ISOLATION FOREST:
    Idea: Anomalies are FEW and DIFFERENT → they're EASY to isolate.

    How it works:
    1. Randomly select a feature
    2. Randomly select a split value between min and max
    3. Repeat until each point is isolated (alone in a partition)

    Anomalies get isolated in FEWER splits than normal points.
    → Short average path length = anomaly
    → Long average path length = normal

    Think of it like this: if you're picking a random person from a crowd,
    "the 7-foot-tall person with purple hair" gets isolated in 2 questions.
    "An average-height person with brown hair" takes many more questions.

  LOCAL OUTLIER FACTOR (LOF):
    Measures local density around each point. Points in low-density
    regions (far from their neighbors) are likely anomalies.

    LOF score > 1 = more anomalous than neighbors
    LOF score ≈ 1 = similar density to neighbors
========================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve, average_precision_score
)
import os
import time


def train_isolation_forest(X_train, contamination='auto', random_state=42):
    """
    ISOLATION FOREST — The Core Unsupervised Anomaly Detector

    HOW IT WORKS (step by step):
      1. Build many random trees (n_estimators)
      2. Each tree randomly selects features and split points
      3. Points that are "easy to isolate" (short path) = anomalies
      4. Average path length across all trees = anomaly score

    KEY PARAMETERS:
      contamination: Expected proportion of anomalies.
        'auto' uses a statistical threshold.
        0.00172 = the actual fraud rate in our data.
        This affects the decision boundary, NOT the scoring.

      n_estimators: Number of trees. More = more stable scores.

      max_samples: How many samples each tree sees.
        'auto' uses min(256, n_samples).
        Smaller = faster, more random, better for anomaly detection.
        (Anomalies are easier to find in small random samples!)

    RETURNS:
      -1 for anomalies (potential fraud), +1 for normal
    """
    print("\n  Training Isolation Forest...")

    iforest = IsolationForest(
        n_estimators=300,
        max_samples='auto',
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )

    start = time.time()
    iforest.fit(X_train)
    elapsed = time.time() - start

    print(f"    Done in {elapsed:.2f}s")
    print(f"    Trees: {iforest.n_estimators}")
    print(f"    Contamination: {contamination}")

    return iforest


def train_lof(X_train, contamination=0.00172, n_neighbors=20):
    """
    LOCAL OUTLIER FACTOR (LOF)

    HOW IT WORKS:
      1. For each point, find its k nearest neighbors
      2. Compute the local density (how close neighbors are)
      3. Compare each point's density to its neighbors' densities
      4. If a point is in a much lower-density region → anomaly

    INTUITION:
      Imagine a crowded mall. Most people are clustered in shops.
      A person standing alone in a dark corner is an outlier — their
      "local density" is much lower than their neighbors' density.

    WHY LOF + IFOREST?
      - Isolation Forest: global anomaly detection (random splits)
      - LOF: local anomaly detection (density-based)
      - They catch different types of anomalies!
      - Ensemble of both = more robust detection

    NOTE: LOF with novelty=True means we train on normal data and
    predict on new data. This is the realistic use case.
    """
    print("\n  Training Local Outlier Factor...")

    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,  # Allows prediction on new data
        n_jobs=-1,
    )

    start = time.time()
    lof.fit(X_train)
    elapsed = time.time() - start

    print(f"    Done in {elapsed:.2f}s")
    print(f"    Neighbors: {n_neighbors}")
    print(f"    Contamination: {contamination}")

    return lof


def evaluate_anomaly_detector(detector, X_test, y_test, name):
    """
    Evaluate an anomaly detector against ground truth labels.

    IMPORTANT MAPPING:
      Anomaly detectors output: -1 (anomaly) or +1 (normal)
      Our labels use: 1 (fraud) or 0 (legit)

      So we map: detector's -1 → our 1 (fraud)
                 detector's +1 → our 0 (legit)

    NOTE: We don't expect unsupervised methods to match supervised ones.
    The point is: they work WITHOUT LABELS. Getting 50% recall with
    no labels is impressive — it means we caught half the fraud with
    zero training labels!
    """
    print(f"\n  Evaluating {name}...")

    # Get predictions
    y_pred_raw = detector.predict(X_test)
    # Map: -1 (anomaly) → 1 (fraud), +1 (normal) → 0 (legit)
    y_pred = (y_pred_raw == -1).astype(int)

    # Get anomaly scores (lower = more anomalous)
    scores = detector.decision_function(X_test)
    # Negate so higher = more anomalous (matches P(fraud) intuition)
    anomaly_scores = -scores

    # Compute metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # AUPRC using continuous anomaly scores
    auprc = average_precision_score(y_test, anomaly_scores)

    metrics = {
        'Model': name,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUPRC': auprc,
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
    }

    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f} (caught {tp}/{tp+fn} frauds)")
    print(f"    F1:        {f1:.4f}")
    print(f"    AUPRC:     {auprc:.4f}")
    print(f"    False alarms: {fp:,}")

    return metrics, anomaly_scores


def plot_anomaly_results(all_metrics, all_scores, y_test, output_dir='outputs'):
    """
    Visualize anomaly detection results.

    PLOTS:
    1. Score distributions: How well do anomaly scores separate fraud/legit?
    2. Precision-Recall curves: How does performance change with threshold?
    3. Comparison bar chart: Supervised vs Unsupervised performance
    """
    print("\n  Generating anomaly detection plots...")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # ─── Plot 1: Score Distributions ──────────────────────────────────
    ax = axes[0]
    for name, scores in all_scores.items():
        fraud_scores = scores[y_test == 1]
        legit_scores = scores[y_test == 0]

        ax.hist(legit_scores, bins=50, alpha=0.5, density=True,
                label=f'{name} — Legit', color='#2ecc71')
        ax.hist(fraud_scores, bins=50, alpha=0.7, density=True,
                label=f'{name} — Fraud', color='#e74c3c')
        break  # Only show the first model for clarity

    ax.set_title('Anomaly Score Distribution', fontsize=13, fontweight='bold')
    ax.set_xlabel('Anomaly Score (higher = more suspicious)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ─── Plot 2: PR Curves ────────────────────────────────────────────
    ax = axes[1]
    for name, scores in all_scores.items():
        prec, rec, _ = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        ax.plot(rec, prec, label=f'{name} (AUPRC={ap:.4f})', linewidth=2)

    baseline = y_test.mean()
    ax.axhline(y=baseline, color='gray', linestyle='--', label=f'Random ({baseline:.4f})')
    ax.set_title('Precision-Recall Curves (Unsupervised)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ─── Plot 3: Metrics Comparison ───────────────────────────────────
    ax = axes[2]
    names = [m['Model'] for m in all_metrics]
    metrics_list = ['Precision', 'Recall', 'F1', 'AUPRC']
    x = np.arange(len(names))
    width = 0.2
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for i, metric in enumerate(metrics_list):
        vals = [m[metric] for m in all_metrics]
        ax.bar(x + i * width, vals, width, label=metric, color=colors[i],
               edgecolor='black', linewidth=0.3)

    ax.set_title('Unsupervised Model Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '13_anomaly_detection.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved: 13_anomaly_detection.png")


def run_anomaly_detection(data, output_dir='outputs'):
    """
    Full unsupervised anomaly detection pipeline.

    WORKFLOW:
    1. Train Isolation Forest on normal transactions (or all training data)
    2. Train LOF on normal transactions
    3. Evaluate both against ground truth
    4. Compare with supervised results
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "█" * 60)
    print("█  UNSUPERVISED ANOMALY DETECTION")
    print("█" * 60)

    X_train = data['X_train']
    X_test = data['X_test']
    y_test = data['y_test']

    # Compute contamination from training data
    fraud_rate = data['y_train'].mean()
    print(f"\n  Estimated fraud rate: {fraud_rate:.5f} ({fraud_rate*100:.3f}%)")

    all_metrics = []
    all_scores = {}

    # ─── Isolation Forest ─────────────────────────────────────────────
    iforest = train_isolation_forest(X_train, contamination=fraud_rate)
    metrics, scores = evaluate_anomaly_detector(iforest, X_test, y_test, 'IsolationForest')
    all_metrics.append(metrics)
    all_scores['IsolationForest'] = scores

    # ─── LOF ──────────────────────────────────────────────────────────
    lof = train_lof(X_train, contamination=fraud_rate)
    metrics, scores = evaluate_anomaly_detector(lof, X_test, y_test, 'LOF')
    all_metrics.append(metrics)
    all_scores['LOF'] = scores

    # ─── Plots ────────────────────────────────────────────────────────
    plot_anomaly_results(all_metrics, all_scores, y_test, output_dir)

    # ─── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("UNSUPERVISED VS SUPERVISED COMPARISON")
    print("=" * 60)
    print("\n  Key insight: Unsupervised methods achieve decent AUPRC")
    print("  WITHOUT seeing any labels. This is useful when:")
    print("    • You have no labeled fraud data")
    print("    • You want to detect novel fraud patterns")
    print("    • You want a second opinion alongside supervised models")

    df_results = pd.DataFrame(all_metrics)
    print(f"\n{df_results[['Model', 'Precision', 'Recall', 'F1', 'AUPRC']].to_string(index=False)}")

    return all_metrics, all_scores


if __name__ == '__main__':
    from preprocess import preprocess
    data = preprocess()
    run_anomaly_detection(data)
