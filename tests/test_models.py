"""Base model training produces usable, probabilistic classifiers."""
import numpy as np


def test_train_all_models_count_and_names(base_models):
    # 3 algorithms x 2 imbalance strategies = 6 variants.
    assert len(base_models) == 6
    for name in [
        "LogisticRegression_weighted", "RandomForest_weighted", "XGBoost_weighted",
        "LogisticRegression_SMOTE", "RandomForest_SMOTE", "XGBoost_SMOTE",
    ]:
        assert name in base_models


def test_models_predict_proba(base_models, data):
    X_test = data["X_test"]
    for name, model in base_models.items():
        proba = model.predict_proba(X_test)[:, 1]
        assert proba.shape[0] == len(X_test)
        assert np.all((proba >= 0) & (proba <= 1)), f"{name} proba out of range"
