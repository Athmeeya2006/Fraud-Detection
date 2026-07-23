"""
========================================================================
MODULE: Stratified K-Fold Cross-Validation
========================================================================
WHAT WE LEARN HERE:
  - Why a single train/test split can be misleading
  - K-Fold cross-validation: testing on EVERY part of the data
  - Stratified K-Fold: preserving class balance in each fold
  - Confidence intervals: "AUPRC = 0.82 ± 0.03" vs just "AUPRC = 0.82"

WHY CROSS-VALIDATION?
  Problem with a single split:
    Maybe we got "lucky" with our test set - it happened to contain
    easy-to-detect fraud. Or maybe we got unlucky and the test set
    has unusual fraud patterns. One split = one data point.

  K-Fold cross-validation:
    Split data into K parts. Train on K-1, test on 1. Repeat K times.
    Each sample gets to be in the test set EXACTLY ONCE.

    Fold 1: [TEST] [train] [train] [train] [train]
    Fold 2: [train] [TEST] [train] [train] [train]
    Fold 3: [train] [train] [TEST] [train] [train]
    Fold 4: [train] [train] [train] [TEST] [train]
    Fold 5: [train] [train] [train] [train] [TEST]

    → 5 AUPRC scores instead of 1 → mean ± std

  WHY "STRATIFIED"?
    Regular K-Fold might put ALL fraud into one fold and none in others.
    Stratified K-Fold ensures each fold has the SAME fraud percentage.
    This is CRITICAL for imbalanced data like ours (0.17% fraud).

IMPORTANT TRADEOFFS:
  - More folds = better estimate, but slower (5 folds = 5× training time)
  - Cross-validation tells you about model STABILITY, not just performance
  - High variance across folds = model is unstable (bad sign)
  - Low variance = you can trust the single-split results
========================================================================
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score
)
from xgboost import XGBClassifier
import time
import os


def get_model_configs():
    """
    Returns model configurations to cross-validate.

    WHY THESE SPECIFIC CONFIGS?
      We use the same hyperparameters as in models.py for consistency.
      The goal is to see how STABLE each model's performance is,
      not to tune hyperparameters (that would require nested CV).
    """
    configs = {
        'LogisticRegression': {
            'class': LogisticRegression,
            'params': {
                'C': 1.0, 'max_iter': 1000, 'solver': 'lbfgs',
                'class_weight': 'balanced', 'random_state': 42,
            }
        },
        'RandomForest': {
            'class': RandomForestClassifier,
            'params': {
                'n_estimators': 200, 'max_depth': 15, 'min_samples_leaf': 5,
                'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1,
            }
        },
        'XGBoost': {
            'class': XGBClassifier,
            'params': {
                'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1,
                'eval_metric': 'aucpr', 'n_jobs': -1, 'tree_method': 'hist',
                'random_state': 42,
                # scale_pos_weight computed per fold
            }
        },
    }
    return configs


def run_stratified_cv(data_path='creditcard.csv', n_splits=5, output_dir='outputs'):
    """
    Run Stratified K-Fold Cross-Validation on all models.

    WORKFLOW (per fold):
      1. Split into train/val using StratifiedKFold (preserves fraud ratio)
      2. Scale features (fit scaler on TRAIN, transform both)
      3. Apply SMOTE to training fold only (for SMOTE variants)
      4. Train model
      5. Evaluate on validation fold
      6. Record metrics

    CRITICAL: The scaler is fit FRESH on each fold's training data.
    Using a scaler fit on the full dataset would leak information!

    WHAT WE GET:
      For each model: mean ± std of all metrics across K folds.
      This tells us:
        - Is the model consistently good? (low std)
        - Or was the single-split result a fluke? (high std)
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "█" * 60)
    print("█  STRATIFIED K-FOLD CROSS-VALIDATION")
    print("█" * 60)

    # Load data fresh (we need unscaled data for proper CV)
    print(f"\n  Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    X = df.drop('Class', axis=1)
    y = df['Class']

    print(f"  Samples: {len(X):,}")
    print(f"  Fraud: {y.sum():,} ({y.mean()*100:.3f}%)")
    print(f"  Folds: {n_splits}")

    # Set up stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Get model configs
    configs = get_model_configs()

    # Results storage
    all_fold_results = {name: [] for name in configs}

    total_start = time.time()

    # ─── Run K-Fold ───────────────────────────────────────────────────
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'─' * 60}")
        print(f"FOLD {fold_idx + 1}/{n_splits}")
        print(f"{'─' * 60}")

        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        print(f"  Train: {len(X_train_fold):,} (fraud: {y_train_fold.sum()})")
        print(f"  Val:   {len(X_val_fold):,} (fraud: {y_val_fold.sum()})")

        # Scale features (fit on train, transform both)
        scaler = StandardScaler()
        X_train_scaled = X_train_fold.copy()
        X_val_scaled = X_val_fold.copy()

        X_train_scaled['Time'] = scaler.fit_transform(X_train_fold[['Time']])
        X_val_scaled['Time'] = scaler.transform(X_val_fold[['Time']])

        scaler2 = StandardScaler()
        X_train_scaled['Amount'] = scaler2.fit_transform(X_train_fold[['Amount']])
        X_val_scaled['Amount'] = scaler2.transform(X_val_fold[['Amount']])

        # Train and evaluate each model
        for model_name, config in configs.items():
            start = time.time()

            # Handle scale_pos_weight for XGBoost
            params = config['params'].copy()
            if model_name == 'XGBoost':
                ratio = (y_train_fold == 0).sum() / max((y_train_fold == 1).sum(), 1)
                params['scale_pos_weight'] = ratio

            model = config['class'](**params)

            if model_name == 'XGBoost':
                model.fit(X_train_scaled, y_train_fold, verbose=False)
            else:
                model.fit(X_train_scaled, y_train_fold)

            elapsed = time.time() - start

            # Evaluate
            y_proba = model.predict_proba(X_val_scaled)[:, 1]
            y_pred = model.predict(X_val_scaled)

            fold_metrics = {
                'Fold': fold_idx + 1,
                'Precision': precision_score(y_val_fold, y_pred, zero_division=0),
                'Recall': recall_score(y_val_fold, y_pred, zero_division=0),
                'F1': f1_score(y_val_fold, y_pred, zero_division=0),
                'AUPRC': average_precision_score(y_val_fold, y_proba),
                'ROC-AUC': roc_auc_score(y_val_fold, y_proba),
                'Time': elapsed,
            }

            all_fold_results[model_name].append(fold_metrics)
            print(f"    {model_name}: AUPRC={fold_metrics['AUPRC']:.4f}, "
                  f"F1={fold_metrics['F1']:.4f} ({elapsed:.1f}s)")

    total_elapsed = time.time() - total_start

    # ─── Aggregate Results ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)

    summary_rows = []
    fold_data = {}

    for model_name, fold_results in all_fold_results.items():
        df_folds = pd.DataFrame(fold_results)
        fold_data[model_name] = df_folds

        summary = {
            'Model': model_name,
        }

        for metric in ['Precision', 'Recall', 'F1', 'AUPRC', 'ROC-AUC']:
            mean = df_folds[metric].mean()
            std = df_folds[metric].std()
            summary[f'{metric}_mean'] = mean
            summary[f'{metric}_std'] = std
            summary[f'{metric}'] = f"{mean:.4f} ± {std:.4f}"

        summary_rows.append(summary)

        print(f"\n  {model_name}:")
        print(f"    AUPRC:   {summary['AUPRC']}")
        print(f"    F1:      {summary['F1']}")
        print(f"    Recall:  {summary['Recall']}")
        print(f"    ROC-AUC: {summary['ROC-AUC']}")

    # ─── Visualization ────────────────────────────────────────────────
    plot_cv_results(fold_data, output_dir)

    print(f"\n  Total CV time: {total_elapsed:.1f}s")

    # Save summary
    df_summary = pd.DataFrame(summary_rows)
    cv_cols = ['Model', 'AUPRC', 'F1', 'Precision', 'Recall', 'ROC-AUC']
    df_summary[cv_cols].to_csv(os.path.join(output_dir, 'cv_results.csv'), index=False)
    print("  → Saved: cv_results.csv")

    return summary_rows, fold_data


def plot_cv_results(fold_data, output_dir='outputs'):
    """
    Visualize cross-validation results.

    PLOTS:
    1. Box plot: Shows distribution of each metric across folds
       → Wide boxes = unstable model
       → Narrow boxes = consistent performance

    2. Fold-by-fold line plot: Shows if performance degrades on certain folds
       → Useful to spot if the model struggles with specific data patterns
    """
    print("\n  Generating cross-validation plots...")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    model_names = list(fold_data.keys())
    metrics = ['AUPRC', 'F1', 'Recall', 'Precision']
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']

    # ─── Plot 1: Box plots ────────────────────────────────────────────
    ax = axes[0]
    positions = []
    box_data = []
    box_labels = []
    box_colors = []

    for i, metric in enumerate(metrics):
        for j, model_name in enumerate(model_names):
            pos = i * (len(model_names) + 1) + j
            positions.append(pos)
            values = fold_data[model_name][metric].values
            box_data.append(values)
            box_labels.append(f"{model_name[:6]}")
            box_colors.append(colors[i])

    bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='black', markersize=6))

    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Add metric group labels
    group_centers = []
    for i, metric in enumerate(metrics):
        center = i * (len(model_names) + 1) + (len(model_names) - 1) / 2
        group_centers.append(center)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax.set_title('Cross-Validation Score Distribution', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend([plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.6)
               for c in ['#9b59b6', '#3498db', '#2ecc71']],
              model_names, loc='lower right', fontsize=9)

    # ─── Plot 2: AUPRC across folds ──────────────────────────────────
    ax = axes[1]
    markers = ['o', 's', '^']
    line_colors = ['#e74c3c', '#3498db', '#2ecc71']

    for i, (model_name, df_folds) in enumerate(fold_data.items()):
        folds = df_folds['Fold'].values
        auprc = df_folds['AUPRC'].values
        mean_auprc = auprc.mean()

        ax.plot(folds, auprc, marker=markers[i], linewidth=2, markersize=8,
                color=line_colors[i], label=f'{model_name} (μ={mean_auprc:.4f})')
        ax.axhline(y=mean_auprc, color=line_colors[i], linestyle='--',
                   alpha=0.4, linewidth=1)

    ax.set_title('AUPRC Across Folds', fontsize=13, fontweight='bold')
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('AUPRC', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, len(folds) + 1))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '14_cross_validation.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved: 14_cross_validation.png")


if __name__ == '__main__':
    run_stratified_cv()
