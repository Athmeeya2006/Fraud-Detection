"""Cost-sensitive business analysis: accounting identities and threshold search."""
import os

import numpy as np

from cost_analysis import (
    compute_transaction_costs,
    find_cost_optimal_threshold,
    run_cost_analysis,
)


def test_compute_transaction_costs_accounting():
    y_test = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])
    amounts = np.array([100.0, 200.0, 50.0, 10.0, 300.0])

    costs = compute_transaction_costs(
        y_test, y_pred, None, amounts,
        investigation_cost=25.0, chargeback_rate=1.0,
    )
    # TP amounts: rows 0 and 4 -> 100 + 300 = 400
    assert costs["fraud_prevented"] == 400.0
    # FN amounts: row 1 -> 200
    assert costs["fraud_missed"] == 200.0
    # FP count: row 2 -> 1 investigation -> $25
    assert costs["investigation_cost"] == 25.0
    assert costs["net_savings"] == 400.0 - 200.0 - 25.0
    assert costs["tp_count"] == 2 and costs["fn_count"] == 1 and costs["fp_count"] == 1


def test_find_cost_optimal_threshold_returns_valid():
    y_test = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    y_proba = np.array([0.9, 0.1, 0.8, 0.2, 0.6, 0.05, 0.3, 0.95])
    amounts = np.array([500.0, 20.0, 400.0, 10.0, 300.0, 5.0, 15.0, 600.0])

    df, best_t = find_cost_optimal_threshold(y_test, y_proba, amounts)
    assert 0.0 < best_t < 1.0
    assert df["net_savings"].max() == df.loc[df["threshold"] == best_t, "net_savings"].values[0]


def test_run_cost_analysis(tree_model, data, output_dir):
    model, name = tree_model
    cost_df, threshold, costs = run_cost_analysis(
        model, name, data, output_dir=output_dir
    )
    assert 0.0 < threshold < 1.0
    assert "net_savings" in costs
    assert os.path.exists(os.path.join(output_dir, "15_cost_analysis.png"))
