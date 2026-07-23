"""Unsupervised anomaly detection runs and returns aligned scores."""
import os

from anomaly_detection import run_anomaly_detection


def test_run_anomaly_detection(data, output_dir):
    all_metrics, all_scores = run_anomaly_detection(data, output_dir=output_dir)

    names = {m["Model"] for m in all_metrics}
    assert {"IsolationForest", "LOF"} <= names

    for name, scores in all_scores.items():
        # One anomaly score per test transaction.
        assert len(scores) == len(data["y_test"]), name

    assert os.path.exists(os.path.join(output_dir, "13_anomaly_detection.png"))
