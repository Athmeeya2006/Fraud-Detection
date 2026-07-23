"""
========================================================================
MODULE: Cost-Sensitive Business Analysis
========================================================================
WHAT WE LEARN HERE:
  - Why pure ML metrics don't tell the whole business story
  - How to translate model performance into actual dollar costs
  - Optimal threshold selection based on FINANCIAL loss, not F1

THE BUSINESS PROBLEM:
  A bank needs to answer: "How much money does this model save us?"

  Four outcomes, each with a different cost:
  ┌─────────────────────────────────────────────────────────────────┐
  │  True Positive (catch fraud):    Save the transaction amount    │
  │  False Negative (miss fraud):    Lose the transaction amount    │
  │  False Positive (false alarm):   Investigation cost (~$5-50)    │
  │  True Negative (correct legit):  No cost                        │
  └─────────────────────────────────────────────────────────────────┘

  The OPTIMAL threshold depends on these costs:
  - If investigation cost is cheap ($5): lower threshold → catch more fraud
  - If investigation cost is expensive ($50): higher threshold → fewer false alarms

WHY THIS MATTERS:
  A model with 90% precision and 60% recall might save MORE money
  than a model with 60% precision and 90% recall - it depends on
  the transaction amounts and investigation costs!

  Example:
    Model A catches 60 frauds (avg $500) with 10 false alarms ($50 each)
    → Saves: 60 × $500 - 10 × $50 = $29,500

    Model B catches 80 frauds (avg $500) with 200 false alarms ($50 each)
    → Saves: 80 × $500 - 200 × $50 = $30,000

    Model B is slightly better in dollar terms despite worse precision!
========================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def compute_transaction_costs(y_test, y_pred, y_proba, amounts,
                              investigation_cost=25.0, chargeback_rate=1.0):
    """
    Compute the financial impact of model predictions.

    PARAMETERS:
      investigation_cost: Cost to investigate a flagged transaction ($)
        → This includes analyst time, customer notification, card replacement
        → Typical range: $5-$50 depending on automation level

      chargeback_rate: Multiplier for fraud loss
        → 1.0 = lose the transaction amount
        → 1.5 = lose amount + chargeback fees + penalties
        → In reality, banks also face reputational damage (hard to quantify)

    RETURNS:
      A dict with:
        - total_fraud_prevented: $ saved by catching fraud
        - total_fraud_missed: $ lost from missed fraud
        - total_investigation_cost: $ spent investigating flagged transactions
        - net_savings: total_fraud_prevented - total_fraud_missed - total_investigation_cost
    """
    amounts = np.array(amounts)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    # True Positives: fraud caught → money saved
    tp_mask = (y_test == 1) & (y_pred == 1)
    fraud_prevented = amounts[tp_mask].sum() * chargeback_rate

    # False Negatives: fraud missed → money lost
    fn_mask = (y_test == 1) & (y_pred == 0)
    fraud_missed = amounts[fn_mask].sum() * chargeback_rate

    # False Positives: legit flagged → investigation cost
    fp_mask = (y_test == 0) & (y_pred == 1)
    fp_count = fp_mask.sum()
    investigation_total = fp_count * investigation_cost

    # True Negatives: correct, no cost
    tn_mask = (y_test == 0) & (y_pred == 0)

    net_savings = fraud_prevented - fraud_missed - investigation_total

    return {
        'fraud_prevented': fraud_prevented,
        'fraud_missed': fraud_missed,
        'investigation_cost': investigation_total,
        'net_savings': net_savings,
        'tp_count': tp_mask.sum(),
        'fn_count': fn_mask.sum(),
        'fp_count': fp_count,
        'tn_count': tn_mask.sum(),
        'avg_fraud_amount': amounts[y_test == 1].mean(),
        'avg_legit_amount': amounts[y_test == 0].mean(),
    }


def find_cost_optimal_threshold(y_test, y_proba, amounts,
                                investigation_cost=25.0, chargeback_rate=1.0):
    """
    Find the threshold that MINIMIZES total business cost.

    THIS IS DIFFERENT FROM F1-OPTIMAL THRESHOLD:
      F1 treats all predictions equally.
      Cost-optimal considers that:
        - Missing a $10,000 fraud is WAY worse than missing a $5 fraud
        - A false alarm costs $25 regardless of transaction amount

    HOW:
      For each threshold from 0.01 to 0.99:
        1. Compute predictions at this threshold
        2. Calculate total business cost
        3. Track which threshold gives the lowest cost

    THE KEY INSIGHT:
      Low thresholds: catch more fraud (save money) but many false alarms (spend money)
      High thresholds: few false alarms but miss expensive fraud
      There's a sweet spot that minimizes total cost!
    """
    thresholds = np.arange(0.01, 0.99, 0.01)

    results = []
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        costs = compute_transaction_costs(
            y_test, y_pred_t, y_proba, amounts,
            investigation_cost, chargeback_rate
        )
        costs['threshold'] = t
        results.append(costs)

    df = pd.DataFrame(results)

    # Find optimal threshold (max net savings)
    best_idx = df['net_savings'].idxmax()
    best_threshold = df.loc[best_idx, 'threshold']

    return df, best_threshold


def plot_cost_analysis(cost_df, best_threshold, model_name,
                       investigation_cost, output_dir='outputs'):
    """
    Visualize the cost-sensitive analysis.

    PLOTS:
    1. Cost Components vs Threshold:
       Shows how each cost type changes with threshold
       → Helps understand the tradeoff visually

    2. Net Savings vs Threshold:
       Shows the "sweet spot" - the threshold that maximizes savings
       → Compare with F1-optimal threshold

    3. Cost Breakdown at Optimal Threshold:
       Pie chart showing where the money goes
    """
    print("\n  Generating cost analysis plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ─── Plot 1: Cost Components ──────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(cost_df['threshold'], cost_df['fraud_prevented'],
            color='#2ecc71', linewidth=2, label='Fraud Prevented ($)')
    ax.plot(cost_df['threshold'], cost_df['fraud_missed'],
            color='#e74c3c', linewidth=2, label='Fraud Missed ($)')
    ax.plot(cost_df['threshold'], cost_df['investigation_cost'],
            color='#f39c12', linewidth=2, label='Investigation Cost ($)')

    ax.axvline(x=best_threshold, color='purple', linestyle='--',
               linewidth=1.5, label=f'Cost-optimal = {best_threshold:.2f}')
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1,
               label='Default = 0.50')

    ax.set_title('Cost Components vs Threshold', fontsize=13, fontweight='bold')
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Amount ($)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ─── Plot 2: Net Savings ──────────────────────────────────────────
    ax = axes[0, 1]
    ax.fill_between(cost_df['threshold'], cost_df['net_savings'],
                    alpha=0.3, color='#2ecc71')
    ax.plot(cost_df['threshold'], cost_df['net_savings'],
            color='#2ecc71', linewidth=2.5)

    best_savings = cost_df.loc[cost_df['threshold'] == best_threshold, 'net_savings'].values[0]
    ax.axvline(x=best_threshold, color='purple', linestyle='--', linewidth=1.5)
    ax.scatter([best_threshold], [best_savings], color='purple', s=100, zorder=5,
               label=f'Max savings: ${best_savings:,.0f}\nat threshold {best_threshold:.2f}')

    ax.set_title('Net Savings vs Threshold', fontsize=13, fontweight='bold')
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Net Savings ($)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ─── Plot 3: Optimal Threshold Breakdown ──────────────────────────
    ax = axes[1, 0]
    best_row = cost_df[cost_df['threshold'] == best_threshold].iloc[0]

    categories = ['Fraud\nPrevented', 'Fraud\nMissed', 'Investigation\nCost']
    values = [best_row['fraud_prevented'], best_row['fraud_missed'],
              best_row['investigation_cost']]
    bar_colors = ['#2ecc71', '#e74c3c', '#f39c12']

    bars = ax.bar(categories, values, color=bar_colors, edgecolor='black',
                  linewidth=0.5, width=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                f'${val:,.0f}', ha='center', fontweight='bold', fontsize=11)

    ax.set_title(f'Cost Breakdown at Optimal Threshold ({best_threshold:.2f})',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Amount ($)')
    ax.grid(True, alpha=0.2, axis='y')

    # ─── Plot 4: Sensitivity to Investigation Cost ────────────────────
    ax = axes[1, 1]

    # Show how optimal threshold changes with investigation cost
    inv_costs = [5, 10, 25, 50, 100, 200]
    opt_thresholds = []
    opt_savings = []

    for ic in inv_costs:
        savings = (cost_df['fraud_prevented']
                   - cost_df['fraud_missed']
                   - cost_df['fp_count'] * ic)
        best_t = cost_df['threshold'].iloc[savings.idxmax()]
        opt_thresholds.append(best_t)
        opt_savings.append(savings.max())

    ax2 = ax.twinx()

    bars = ax.bar(range(len(inv_costs)), opt_thresholds, color='#3498db',
                  alpha=0.7, edgecolor='black', linewidth=0.3, width=0.4)
    ax.set_xticks(range(len(inv_costs)))
    ax.set_xticklabels([f'${ic}' for ic in inv_costs])
    ax.set_xlabel('Investigation Cost per False Alarm')
    ax.set_ylabel('Optimal Threshold', color='#3498db')
    ax.tick_params(axis='y', labelcolor='#3498db')

    ax2.plot(range(len(inv_costs)), opt_savings, color='#e74c3c',
                    marker='o', linewidth=2, markersize=8)
    ax2.set_ylabel('Max Net Savings ($)', color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')

    ax.set_title('Sensitivity Analysis:\nOptimal Threshold vs Investigation Cost',
                 fontsize=13, fontweight='bold')

    plt.suptitle(f'Cost-Sensitive Analysis - {model_name}\n(Investigation cost: ${investigation_cost})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '15_cost_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved: 15_cost_analysis.png")


def run_cost_analysis(best_model, best_model_name, data,
                      investigation_cost=25.0, chargeback_rate=1.5,
                      output_dir='outputs'):
    """
    Full cost-sensitive analysis pipeline.

    PARAMETERS:
      investigation_cost: Cost per false alarm ($25 default)
      chargeback_rate: Multiplier for fraud loss
        1.0 = lose just the transaction amount
        1.5 = lose amount + 50% for chargeback fees, penalties, etc.

    WHAT THIS TELLS YOU:
      "Using [model] with threshold [X], the bank would save $[Y] per
      [N] transactions compared to no fraud detection system."
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "█" * 60)
    print("█  COST-SENSITIVE BUSINESS ANALYSIS")
    print("█" * 60)

    X_test = data['X_test']
    y_test = data['y_test']

    # Get test set transaction amounts.
    # The Amount column was scaled during preprocessing, so we need the
    # ORIGINAL unscaled amounts for a meaningful dollar analysis. These are
    # carried through the preprocess() output as 'amount_test', already aligned
    # with X_test/y_test (no re-reading or re-splitting of the raw CSV).
    test_amounts = np.asarray(data['amount_test'])

    print(f"\n  Test set: {len(y_test):,} transactions")
    print(f"  Fraud in test set: {y_test.sum()} transactions")
    print(f"  Total fraud amount: ${test_amounts[y_test.values == 1].sum():,.2f}")
    print(f"  Average fraud amount: ${test_amounts[y_test.values == 1].mean():,.2f}")
    print(f"  Investigation cost per false alarm: ${investigation_cost}")
    print(f"  Chargeback rate: {chargeback_rate}×")

    # Get model probabilities
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # ─── At Default Threshold (0.5) ───────────────────────────────────
    print("\n" + "=" * 60)
    print("COST AT DEFAULT THRESHOLD (0.50)")
    print("=" * 60)

    y_pred_default = (y_proba >= 0.5).astype(int)
    default_costs = compute_transaction_costs(
        y_test, y_pred_default, y_proba, test_amounts,
        investigation_cost, chargeback_rate
    )

    print(f"  Fraud prevented: ${default_costs['fraud_prevented']:,.2f} "
          f"({default_costs['tp_count']} transactions)")
    print(f"  Fraud missed:    ${default_costs['fraud_missed']:,.2f} "
          f"({default_costs['fn_count']} transactions)")
    print(f"  Investigation:   ${default_costs['investigation_cost']:,.2f} "
          f"({default_costs['fp_count']} false alarms)")
    print(f"  NET SAVINGS:     ${default_costs['net_savings']:,.2f}")

    # ─── Find Cost-Optimal Threshold ──────────────────────────────────
    print("\n" + "=" * 60)
    print("FINDING COST-OPTIMAL THRESHOLD")
    print("=" * 60)

    cost_df, optimal_threshold = find_cost_optimal_threshold(
        y_test, y_proba, test_amounts,
        investigation_cost, chargeback_rate
    )

    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    optimal_costs = compute_transaction_costs(
        y_test, y_pred_optimal, y_proba, test_amounts,
        investigation_cost, chargeback_rate
    )

    print(f"\n  Cost-optimal threshold: {optimal_threshold:.2f}")
    print(f"  Fraud prevented: ${optimal_costs['fraud_prevented']:,.2f} "
          f"({optimal_costs['tp_count']} transactions)")
    print(f"  Fraud missed:    ${optimal_costs['fraud_missed']:,.2f} "
          f"({optimal_costs['fn_count']} transactions)")
    print(f"  Investigation:   ${optimal_costs['investigation_cost']:,.2f} "
          f"({optimal_costs['fp_count']} false alarms)")
    print(f"  NET SAVINGS:     ${optimal_costs['net_savings']:,.2f}")

    improvement = optimal_costs['net_savings'] - default_costs['net_savings']
    print(f"\n  💰 Threshold tuning improvement: ${improvement:,.2f}")
    print(f"     ({improvement/max(abs(default_costs['net_savings']),1)*100:.1f}% better than default)")

    # ─── No Model Baseline ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON: WITH vs WITHOUT FRAUD DETECTION")
    print("=" * 60)

    total_fraud_amount = test_amounts[y_test.values == 1].sum() * chargeback_rate
    print("\n  Without fraud detection:")
    print(f"    Total fraud loss: ${total_fraud_amount:,.2f}")
    print(f"\n  With {best_model_name} (threshold={optimal_threshold:.2f}):")
    print(f"    Net savings:      ${optimal_costs['net_savings']:,.2f}")
    print(f"    Fraud caught:     {optimal_costs['tp_count']}/{optimal_costs['tp_count']+optimal_costs['fn_count']} "
          f"({optimal_costs['tp_count']/(optimal_costs['tp_count']+optimal_costs['fn_count'])*100:.1f}%)")

    roi = (optimal_costs['net_savings'] / max(optimal_costs['investigation_cost'], 1)) * 100
    print(f"    ROI on investigation: {roi:.0f}%")

    # ─── Plots ────────────────────────────────────────────────────────
    plot_cost_analysis(cost_df, optimal_threshold, best_model_name,
                       investigation_cost, output_dir)

    print("\n✅ Cost-sensitive analysis complete!")
    return cost_df, optimal_threshold, optimal_costs


if __name__ == '__main__':
    from preprocess import preprocess
    from models import train_random_forest

    data = preprocess()
    model, name = train_random_forest(data['X_train'], data['y_train'])
    run_cost_analysis(model, name, data)
