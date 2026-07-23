"""SHAP explainability, including the multi-shape class-selection helper."""
import os

import numpy as np

from feature_importance import (
    _select_fraud_class_shap,
    compute_shap_values,
    run_feature_importance,
)


def test_select_fraud_class_shap_handles_all_shapes():
    n, f = 5, 3
    two_d = np.ones((n, f))
    assert _select_fraud_class_shap(two_d).shape == (n, f)

    # list [class0, class1]
    as_list = [np.zeros((n, f)), np.ones((n, f))]
    out = _select_fraud_class_shap(as_list)
    assert out.shape == (n, f) and np.allclose(out, 1.0)

    # 3-D (n, f, classes)
    cube = np.stack([np.zeros((n, f)), np.ones((n, f))], axis=-1)
    out = _select_fraud_class_shap(cube)
    assert out.shape == (n, f) and np.allclose(out, 1.0)

    # 3-D (classes, n, f)
    cube2 = np.stack([np.zeros((n, f)), np.ones((n, f))], axis=0)
    out = _select_fraud_class_shap(cube2)
    assert out.shape == (n, f) and np.allclose(out, 1.0)


def test_compute_shap_values_shape(tree_model, data):
    model, name = tree_model
    shap_values, X_sample = compute_shap_values(model, data["X_test"], name, max_samples=100)
    assert shap_values.ndim == 2
    assert shap_values.shape[0] == len(X_sample)
    assert shap_values.shape[1] == data["X_test"].shape[1]


def test_run_feature_importance_writes_plots(tree_model, data, output_dir):
    model, name = tree_model
    shap_values, X_sample = run_feature_importance(model, name, data, output_dir=output_dir)
    assert shap_values.ndim == 2
    for fname in [
        "10_shap_global_importance.png",
        "11_shap_individual_explanations.png",
        "12_shap_dependence.png",
    ]:
        assert os.path.exists(os.path.join(output_dir, fname)), fname
