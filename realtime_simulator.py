"""
========================================================================
MODULE: Real-Time Fraud Scoring Simulator
========================================================================
WHAT WE LEARN HERE:
  - How fraud detection works in a production environment
  - Transaction scoring: from raw data to fraud/legit decision
  - Alert systems: how banks actually process model predictions
  - Latency requirements: scoring must happen in milliseconds

HOW REAL-TIME FRAUD DETECTION WORKS IN PRODUCTION:
  ┌──────────────────────────────────────────────────────────────────┐
  │                    REAL-TIME PIPELINE                            │
  │                                                                  │
  │  Customer swipes card                                            │
  │       ↓                                                          │
  │  Transaction data → Feature Engineering → Model Scoring          │
  │       ↓                                                          │
  │  Score > threshold? ─── YES → BLOCK transaction + Alert analyst  │
  │       │                                                          │
  │       NO                                                         │
  │       ↓                                                          │
  │  APPROVE transaction                                             │
  │       ↓                                                          │
  │  Log for batch retraining                                        │
  └──────────────────────────────────────────────────────────────────┘

  LATENCY REQUIREMENT:
    The entire scoring process must happen in < 100ms.
    Customer is waiting at the checkout counter!

    Breakdown:
    - Feature engineering: ~10ms
    - Model scoring:       ~5ms  (tree models are fast!)
    - Decision logic:      ~1ms
    - Network overhead:    ~50ms
    - Buffer:              ~34ms

RISK TIERS IN PRODUCTION:
  Banks don't just use a single threshold. They use TIERS:

  Score > 0.9: BLOCK immediately + alert fraud team
  Score > 0.7: FLAG for manual review within 1 hour
  Score > 0.4: MONITOR - add to watchlist
  Score < 0.4: APPROVE - normal transaction

  This reduces false positives while catching high-risk fraud.
========================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime, timedelta


def simulate_transaction_stream(X_test, y_test, amounts, n_transactions=100,
                                random_state=42):
    """
    Simulate a stream of incoming transactions.

    IN REAL LIFE:
      Transactions arrive from payment networks (Visa, Mastercard) via APIs.
      Each transaction has: card number, merchant, amount, time, location, etc.

    IN OUR SIMULATION:
      We sample from the test set to create a realistic stream.
      We add synthetic timestamps to simulate real-time arrival.

    FRAUD INJECTION:
      We ensure the stream includes some fraud to demonstrate detection.
      In reality, fraud rate ≈ 0.1-0.5% of transactions.
    """
    np.random.seed(random_state)

    len(X_test)

    # Sample indices, ensuring we include some fraud
    fraud_idx = np.where(y_test == 1)[0]
    legit_idx = np.where(y_test == 0)[0]

    # Include ~5-10% fraud for demonstration (higher than real rate)
    n_fraud = min(max(int(n_transactions * 0.08), 3), len(fraud_idx))
    n_legit = n_transactions - n_fraud

    selected_fraud = np.random.choice(fraud_idx, n_fraud, replace=False)
    selected_legit = np.random.choice(legit_idx, n_legit, replace=False)

    selected_idx = np.concatenate([selected_fraud, selected_legit])
    np.random.shuffle(selected_idx)

    # Generate timestamps (transactions arriving over ~10 minutes)
    base_time = datetime.now()
    timestamps = [base_time + timedelta(seconds=np.random.exponential(6))
                  for _ in range(n_transactions)]
    timestamps.sort()

    stream = []
    for i, idx in enumerate(selected_idx):
        stream.append({
            'transaction_id': f'TXN-{i+1:04d}',
            'timestamp': timestamps[i],
            'features_idx': idx,
            'amount': amounts[idx],
            'true_label': int(y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]),
        })

    return stream


def score_transaction(model, X_test, idx, threshold=0.5):
    """
    Score a single transaction - the core of real-time fraud detection.

    WHAT HAPPENS:
      1. Extract feature vector for this transaction
      2. Run through the model to get P(fraud)
      3. Compare against threshold(s)
      4. Return risk tier and score

    LATENCY:
      In production, this function must complete in < 5ms.
      Our tree-based models achieve this easily.
      Neural networks might need GPU acceleration for real-time use.

    RISK TIERS:
      The output isn't just "fraud or not" - it's a RISK LEVEL:
      - CRITICAL (>0.9): Almost certainly fraud. Block immediately.
      - HIGH (>0.7): Very suspicious. Flag for urgent review.
      - MEDIUM (>0.4): Somewhat suspicious. Monitor this card.
      - LOW (<0.4): Appears legitimate. Approve normally.
    """
    start_time = time.perf_counter()

    # Get features for this transaction
    if hasattr(X_test, 'iloc'):
        features = X_test.iloc[[idx]]
    else:
        features = X_test[idx:idx+1]

    # Score
    fraud_prob = model.predict_proba(features)[0, 1]

    # Determine risk tier
    if fraud_prob >= 0.9:
        risk_tier = 'CRITICAL'
        action = 'BLOCK'
        color = '\033[91m'  # Red
    elif fraud_prob >= 0.7:
        risk_tier = 'HIGH'
        action = 'FLAG'
        color = '\033[93m'  # Yellow
    elif fraud_prob >= 0.4:
        risk_tier = 'MEDIUM'
        action = 'MONITOR'
        color = '\033[94m'  # Blue
    else:
        risk_tier = 'LOW'
        action = 'APPROVE'
        color = '\033[92m'  # Green

    latency_ms = (time.perf_counter() - start_time) * 1000

    return {
        'fraud_probability': fraud_prob,
        'risk_tier': risk_tier,
        'action': action,
        'color': color,
        'latency_ms': latency_ms,
    }


def run_realtime_simulation(model, model_name, data, n_transactions=100,
                            threshold=0.5, output_dir='outputs'):
    """
    Run a full real-time fraud detection simulation.

    SIMULATES:
    1. Transaction stream arriving in real-time
    2. Each transaction scored by the model
    3. Risk-tier decisions made
    4. Alerts generated for suspicious transactions
    5. Summary statistics computed

    THIS IS WHAT A FRAUD ANALYST DASHBOARD WOULD SHOW:
    - Live feed of transactions with risk scores
    - Alerts for high-risk transactions
    - Running statistics (catch rate, false alarm rate)
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "█" * 60)
    print("█  REAL-TIME FRAUD SCORING SIMULATOR")
    print("█" * 60)

    X_test = data['X_test']
    y_test = data['y_test']

    # Original (unscaled) test amounts, carried through preprocess() and
    # already aligned with X_test/y_test.
    test_amounts = np.asarray(data['amount_test'])

    # Generate transaction stream
    print(f"\n  Generating {n_transactions} transaction stream...")
    stream = simulate_transaction_stream(X_test, y_test, test_amounts, n_transactions)

    actual_fraud = sum(1 for t in stream if t['true_label'] == 1)
    print(f"  Stream contains {actual_fraud} fraud transactions "
          f"({actual_fraud/n_transactions*100:.1f}%)")

    # ─── Process Stream ───────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print(f"  {'TXN ID':<12} {'TIME':<12} {'AMOUNT':>10} {'SCORE':>8} "
          f"{'RISK':<10} {'ACTION':<8} {'ACTUAL':<8} {'LATENCY'}")
    print(f"{'─' * 80}")

    results = []
    alerts = []
    latencies = []

    for txn in stream:
        # Score the transaction
        score_result = score_transaction(
            model, X_test, txn['features_idx'], threshold
        )

        actual = "⚠ FRAUD" if txn['true_label'] == 1 else "  legit"
        color = score_result['color']
        reset = '\033[0m'

        latencies.append(score_result['latency_ms'])

        result = {
            'transaction_id': txn['transaction_id'],
            'timestamp': txn['timestamp'],
            'amount': txn['amount'],
            'fraud_probability': score_result['fraud_probability'],
            'risk_tier': score_result['risk_tier'],
            'action': score_result['action'],
            'true_label': txn['true_label'],
            'latency_ms': score_result['latency_ms'],
            'correct': (score_result['action'] != 'APPROVE') == (txn['true_label'] == 1),
        }
        results.append(result)

        # Print live feed
        print(f"  {txn['transaction_id']:<12} "
              f"{txn['timestamp'].strftime('%H:%M:%S'):<12} "
              f"${txn['amount']:>9,.2f} "
              f"{score_result['fraud_probability']:>7.4f} "
              f"{color}{score_result['risk_tier']:<10}{reset} "
              f"{score_result['action']:<8} "
              f"{actual:<8} "
              f"{score_result['latency_ms']:.1f}ms")

        # Generate alert for high-risk transactions
        if score_result['risk_tier'] in ['CRITICAL', 'HIGH']:
            alerts.append(result)

    # ─── Summary Statistics ───────────────────────────────────────────
    print(f"\n{'═' * 80}")
    print("SIMULATION SUMMARY")
    print(f"{'═' * 80}")

    df_results = pd.DataFrame(results)

    # Detection stats
    fraud_txns = df_results[df_results['true_label'] == 1]
    legit_txns = df_results[df_results['true_label'] == 0]

    caught = fraud_txns[fraud_txns['action'] != 'APPROVE']
    missed = fraud_txns[fraud_txns['action'] == 'APPROVE']
    false_alarms = legit_txns[legit_txns['action'] != 'APPROVE']

    print("\n  📊 Detection Performance:")
    print(f"     Transactions processed: {len(df_results)}")
    print(f"     Fraud caught:          {len(caught)}/{len(fraud_txns)} "
          f"({len(caught)/max(len(fraud_txns),1)*100:.1f}%)")
    print(f"     Fraud missed:          {len(missed)}/{len(fraud_txns)}")
    print(f"     False alarms:          {len(false_alarms)}/{len(legit_txns)} "
          f"({len(false_alarms)/max(len(legit_txns),1)*100:.2f}%)")

    print("\n  ⚡ Latency Statistics:")
    print(f"     Mean:   {np.mean(latencies):.2f}ms")
    print(f"     Median: {np.median(latencies):.2f}ms")
    print(f"     P95:    {np.percentile(latencies, 95):.2f}ms")
    print(f"     P99:    {np.percentile(latencies, 99):.2f}ms")
    print(f"     Max:    {np.max(latencies):.2f}ms")

    if alerts:
        print(f"\n  🚨 Alerts Generated: {len(alerts)}")
        for alert in alerts[:5]:
            fraud_label = "FRAUD" if alert['true_label'] == 1 else "FALSE ALARM"
            print(f"     {alert['transaction_id']} | ${alert['amount']:,.2f} | "
                  f"Score: {alert['fraud_probability']:.4f} | "
                  f"{alert['risk_tier']} | {fraud_label}")

    # Risk tier distribution
    print("\n  📈 Risk Tier Distribution:")
    for tier in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        count = len(df_results[df_results['risk_tier'] == tier])
        pct = count / len(df_results) * 100
        bar = '█' * int(pct)
        print(f"     {tier:<10} {count:>4} ({pct:>5.1f}%) {bar}")

    # ─── Visualization ────────────────────────────────────────────────
    plot_simulation_results(df_results, model_name, output_dir)

    print("\n✅ Real-time simulation complete!")
    return df_results


def plot_simulation_results(df_results, model_name, output_dir='outputs'):
    """
    Visualize the real-time simulation results.

    PLOTS:
    1. Score distribution colored by actual class
    2. Risk tier breakdown with fraud detection accuracy
    3. Latency histogram (showing model is production-ready)
    4. Score timeline (simulating a live dashboard)
    """
    print("\n  Generating simulation plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ─── Plot 1: Score Distribution ───────────────────────────────────
    ax = axes[0, 0]
    fraud_scores = df_results[df_results['true_label'] == 1]['fraud_probability']
    legit_scores = df_results[df_results['true_label'] == 0]['fraud_probability']

    ax.hist(legit_scores, bins=30, alpha=0.6, color='#2ecc71',
            label=f'Legit (n={len(legit_scores)})', density=True)
    ax.hist(fraud_scores, bins=30, alpha=0.8, color='#e74c3c',
            label=f'Fraud (n={len(fraud_scores)})', density=True)

    # Add threshold lines
    for thresh, label, color in [(0.4, 'Monitor', '#3498db'),
                                  (0.7, 'Flag', '#f39c12'),
                                  (0.9, 'Block', '#e74c3c')]:
        ax.axvline(x=thresh, color=color, linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'{label} ({thresh})')

    ax.set_title('Fraud Score Distribution', fontsize=13, fontweight='bold')
    ax.set_xlabel('Fraud Probability')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ─── Plot 2: Risk Tier Breakdown ──────────────────────────────────
    ax = axes[0, 1]
    tiers = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']

    fraud_per_tier = []
    legit_per_tier = []
    for tier in tiers:
        tier_data = df_results[df_results['risk_tier'] == tier]
        fraud_per_tier.append(len(tier_data[tier_data['true_label'] == 1]))
        legit_per_tier.append(len(tier_data[tier_data['true_label'] == 0]))

    x = np.arange(len(tiers))
    width = 0.35

    ax.bar(x - width/2, fraud_per_tier, width, label='Fraud',
                   color='#e74c3c', edgecolor='black', linewidth=0.3)
    ax.bar(x + width/2, legit_per_tier, width, label='Legit',
                   color='#2ecc71', edgecolor='black', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_title('Risk Tier Breakdown', fontsize=13, fontweight='bold')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')

    # ─── Plot 3: Latency Distribution ─────────────────────────────────
    ax = axes[1, 0]
    latencies = df_results['latency_ms']

    ax.hist(latencies, bins=30, color='#9b59b6', edgecolor='black',
            linewidth=0.3, alpha=0.7)
    ax.axvline(x=latencies.median(), color='red', linestyle='--',
               linewidth=2, label=f'Median: {latencies.median():.2f}ms')
    ax.axvline(x=np.percentile(latencies, 95), color='orange', linestyle='--',
               linewidth=2, label=f'P95: {np.percentile(latencies, 95):.2f}ms')
    ax.axvline(x=100, color='gray', linestyle=':', linewidth=2,
               label='SLA: 100ms')

    ax.set_title('Scoring Latency Distribution', fontsize=13, fontweight='bold')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ─── Plot 4: Score Timeline ───────────────────────────────────────
    ax = axes[1, 1]

    fraud_mask = df_results['true_label'] == 1
    legit_mask = df_results['true_label'] == 0

    ax.scatter(range(len(df_results[legit_mask])),
               df_results[legit_mask]['fraud_probability'],
               c='#2ecc71', s=20, alpha=0.6, label='Legit', zorder=2)
    ax.scatter(range(len(df_results[fraud_mask])),
               df_results[fraud_mask]['fraud_probability'],
               c='#e74c3c', s=80, marker='*', alpha=0.9, label='Fraud', zorder=3)

    # Add threshold zones
    ax.axhspan(0.9, 1.0, alpha=0.1, color='red', label='BLOCK zone')
    ax.axhspan(0.7, 0.9, alpha=0.1, color='orange', label='FLAG zone')
    ax.axhspan(0.4, 0.7, alpha=0.1, color='blue', label='MONITOR zone')

    ax.set_title('Live Transaction Score Timeline', fontsize=13, fontweight='bold')
    ax.set_xlabel('Transaction Sequence')
    ax.set_ylabel('Fraud Score')
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(0.01, 0.5))
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.suptitle(f'Real-Time Fraud Scoring Simulation - {model_name}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '16_realtime_simulation.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved: 16_realtime_simulation.png")


if __name__ == '__main__':
    from preprocess import preprocess
    from models import train_random_forest

    data = preprocess()
    model, name = train_random_forest(data['X_train'], data['y_train'])
    run_realtime_simulation(model, name, data, n_transactions=50)
