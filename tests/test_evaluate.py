"""Evaluation metrics, plots, and the results CSV."""
import os

from evaluate import evaluate_model, evaluate_all


def test_evaluate_single_model(base_models, data):
    name = "RandomForest_weighted"
    metrics, proba = evaluate_model(base_models[name], data["X_test"], data["y_test"], name)

    for key in ["Precision", "Recall", "F1", "AUPRC", "ROC-AUC", "TP", "FP", "FN", "TN"]:
        assert key in metrics
    # Confusion-matrix cells sum to the test-set size.
    assert metrics["TP"] + metrics["FP"] + metrics["FN"] + metrics["TN"] == len(data["y_test"])
    for k in ["Precision", "Recall", "F1", "AUPRC", "ROC-AUC"]:
        assert 0.0 <= metrics[k] <= 1.0


def test_evaluate_all_writes_outputs(base_models, data, output_dir):
    all_metrics, threshold = evaluate_all(base_models, data, output_dir=output_dir)

    assert len(all_metrics) == len(base_models)
    assert 0.0 < threshold < 1.0
    for fname in [
        "06_confusion_matrices.png", "07_curves.png",
        "08_threshold_analysis.png", "09_model_comparison.png", "results.csv",
    ]:
        assert os.path.exists(os.path.join(output_dir, fname)), fname
