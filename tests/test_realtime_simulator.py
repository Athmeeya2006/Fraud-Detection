"""Real-time scoring simulator produces a scored transaction stream."""
import os

from realtime_simulator import (
    simulate_transaction_stream,
    score_transaction,
    run_realtime_simulation,
)


def test_simulate_transaction_stream_size(data):
    stream = simulate_transaction_stream(
        data["X_test"], data["y_test"], data["amount_test"], n_transactions=40
    )
    assert len(stream) == 40
    assert any(t["true_label"] == 1 for t in stream)  # fraud injected
    for t in stream:
        assert {"transaction_id", "timestamp", "features_idx", "amount", "true_label"} <= set(t)


def test_score_transaction_tiers(tree_model, data):
    model, _ = tree_model
    result = score_transaction(model, data["X_test"], idx=0)
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["risk_tier"] in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
    assert result["action"] in {"BLOCK", "FLAG", "MONITOR", "APPROVE"}


def test_run_realtime_simulation(tree_model, data, output_dir):
    model, name = tree_model
    df_results = run_realtime_simulation(
        model, name, data, n_transactions=40, output_dir=output_dir
    )
    assert len(df_results) == 40
    assert os.path.exists(os.path.join(output_dir, "16_realtime_simulation.png"))
