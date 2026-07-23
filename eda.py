"""
========================================================================
TOPIC 1 & 2: Exploratory Data Analysis (EDA)
========================================================================
WHAT WE LEARN HERE:
  - How to understand your data before modeling
  - Visualizing the class imbalance problem
  - Understanding feature distributions for fraud vs legit
  - Correlation analysis to find useful features

WHY IT MATTERS:
  You should NEVER jump straight to modeling. EDA tells you:
  - How severe the imbalance is (spoiler: extreme)
  - Which features already separate fraud from legit
  - Whether you need scaling, outlier handling, etc.
========================================================================
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def run_eda(data_path='creditcard.csv', output_dir='outputs'):
    """
    Performs full Exploratory Data Analysis on the credit card fraud dataset.

    TEACHING NOTES:
    ---------------
    EDA is always Step 1. You need to understand:
    1. The shape & types of your data
    2. Missing values
    3. Class distribution (imbalance!)
    4. Feature distributions — how do fraud vs legit look different?
    5. Correlations — which features might be predictive?
    """
    os.makedirs(output_dir, exist_ok=True)

    # ─── STEP 1: Load and inspect ─────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading & Inspecting Data")
    print("=" * 60)

    df = pd.read_csv(data_path)

    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumn types:\n{df.dtypes.value_counts()}")
    print(f"\nMissing values: {df.isnull().sum().sum()}")
    print("\nBasic stats for Amount:")
    print(df['Amount'].describe())
    print("\nBasic stats for Time:")
    print(df['Time'].describe())

    # ─── STEP 2: Class Distribution ───────────────────────────────────
    # THIS IS THE MOST IMPORTANT PLOT.
    # It shows WHY accuracy is useless — 99.83% of data is one class.
    print("\n" + "=" * 60)
    print("STEP 2: Class Distribution (The Imbalance Problem)")
    print("=" * 60)

    class_counts = df['Class'].value_counts()
    print(f"\nLegit (0): {class_counts[0]:,} ({class_counts[0]/len(df)*100:.3f}%)")
    print(f"Fraud (1): {class_counts[1]:,} ({class_counts[1]/len(df)*100:.3f}%)")
    print(f"Ratio: 1 fraud per {class_counts[0]//class_counts[1]} legit transactions")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    colors = ['#2ecc71', '#e74c3c']
    bars = axes[0].bar(['Legit (0)', 'Fraud (1)'],
                       [class_counts[0], class_counts[1]],
                       color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_title('Transaction Count by Class', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Count')
    for bar, count in zip(bars, [class_counts[0], class_counts[1]]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                     f'{count:,}', ha='center', fontweight='bold')

    # Log scale to actually see the fraud bar
    axes[1].bar(['Legit (0)', 'Fraud (1)'],
                [class_counts[0], class_counts[1]],
                color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_yscale('log')
    axes[1].set_title('Transaction Count (Log Scale)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Count (log)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_class_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved: 01_class_distribution.png")

    # ─── STEP 3: Amount Distribution ──────────────────────────────────
    # Fraud transactions tend to have different amount patterns
    print("\n" + "=" * 60)
    print("STEP 3: Transaction Amount Analysis")
    print("=" * 60)

    fraud = df[df['Class'] == 1]
    legit = df[df['Class'] == 0]

    print(f"\nFraud amount — Mean: ${fraud['Amount'].mean():.2f}, "
          f"Median: ${fraud['Amount'].median():.2f}, "
          f"Max: ${fraud['Amount'].max():.2f}")
    print(f"Legit amount — Mean: ${legit['Amount'].mean():.2f}, "
          f"Median: ${legit['Amount'].median():.2f}, "
          f"Max: ${legit['Amount'].max():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(legit['Amount'], bins=50, alpha=0.7, color='#2ecc71',
                 label='Legit', density=True)
    axes[0].hist(fraud['Amount'], bins=50, alpha=0.7, color='#e74c3c',
                 label='Fraud', density=True)
    axes[0].set_title('Amount Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Amount ($)')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].set_xlim(0, 500)

    axes[1].boxplot([legit['Amount'], fraud['Amount']],
                    tick_labels=['Legit', 'Fraud'],
                    patch_artist=True,
                    boxprops=dict(facecolor='#3498db', alpha=0.7))
    axes[1].set_title('Amount Boxplot', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Amount ($)')
    axes[1].set_ylim(0, 500)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_amount_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved: 02_amount_distribution.png")

    # ─── STEP 4: Time Distribution ────────────────────────────────────
    # Does fraud happen at certain times of day?
    print("\n" + "=" * 60)
    print("STEP 4: Temporal Pattern Analysis")
    print("=" * 60)

    # Time is in seconds from first transaction in the dataset (~48 hours)
    df['Hour'] = (df['Time'] / 3600).astype(int) % 24

    fig, ax = plt.subplots(figsize=(12, 5))
    legit_hours = df[df['Class'] == 0]['Hour']
    fraud_hours = df[df['Class'] == 1]['Hour']

    ax.hist(legit_hours, bins=24, alpha=0.5, color='#2ecc71', label='Legit',
            density=True, range=(0, 24))
    ax.hist(fraud_hours, bins=24, alpha=0.7, color='#e74c3c', label='Fraud',
            density=True, range=(0, 24))
    ax.set_title('Transaction Distribution by Hour', fontsize=14, fontweight='bold')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_xticks(range(0, 24))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_time_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved: 03_time_distribution.png")
    df.drop('Hour', axis=1, inplace=True)

    # ─── STEP 5: Feature Distributions (Fraud vs Legit) ───────────────
    # Which V-features separate fraud from legit?
    # Features where the distributions look very DIFFERENT are likely
    # strong predictors.
    print("\n" + "=" * 60)
    print("STEP 5: Feature Distribution Comparison (V1-V28)")
    print("=" * 60)

    v_features = [f'V{i}' for i in range(1, 29)]

    fig, axes = plt.subplots(7, 4, figsize=(20, 28))
    axes = axes.flatten()

    for i, feature in enumerate(v_features):
        axes[i].hist(legit[feature], bins=50, alpha=0.5, color='#2ecc71',
                     label='Legit', density=True)
        axes[i].hist(fraud[feature], bins=50, alpha=0.5, color='#e74c3c',
                     label='Fraud', density=True)
        axes[i].set_title(feature, fontsize=11, fontweight='bold')
        axes[i].set_xlim(-5, 5)
        if i == 0:
            axes[i].legend(fontsize=8)

    plt.suptitle('Feature Distributions: Fraud vs Legit',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_feature_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved: 04_feature_distributions.png")

    # Identify the most discriminative features
    print("\n  Top discriminative features (largest mean difference):")
    mean_diffs = {}
    for f in v_features:
        diff = abs(fraud[f].mean() - legit[f].mean())
        mean_diffs[f] = diff
    sorted_diffs = sorted(mean_diffs.items(), key=lambda x: x[1], reverse=True)
    for feat, diff in sorted_diffs[:10]:
        print(f"    {feat}: mean_diff = {diff:.4f}  "
              f"(fraud_mean={fraud[feat].mean():.3f}, legit_mean={legit[feat].mean():.3f})")

    # ─── STEP 6: Correlation Heatmap ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Feature Correlations with Fraud Class")
    print("=" * 60)

    correlations = df.corr()['Class'].drop('Class').sort_values()

    fig, ax = plt.subplots(figsize=(10, 8))
    colors_corr = ['#e74c3c' if v < 0 else '#2ecc71' for v in correlations.values]
    correlations.plot(kind='barh', ax=ax, color=colors_corr, edgecolor='black', linewidth=0.3)
    ax.set_title('Feature Correlation with Fraud (Class)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Pearson Correlation')
    ax.axvline(x=0, color='black', linewidth=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_correlation_with_class.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved: 05_correlation_with_class.png")

    print("\n  Most negatively correlated (fraud-indicator):")
    for feat, corr in correlations.head(5).items():
        print(f"    {feat}: {corr:.4f}")
    print("\n  Most positively correlated (fraud-indicator):")
    for feat, corr in correlations.tail(5).items():
        print(f"    {feat}: {corr:.4f}")

    print("\n" + "=" * 60)
    print("EDA COMPLETE — Check outputs/ for all plots")
    print("=" * 60)

    return df


if __name__ == '__main__':
    run_eda()
