"""Advanced models (LightGBM + neural net) train and predict."""
import numpy as np

from advanced_models import train_advanced_models, train_neural_network


def test_train_advanced_models(data):
    models = train_advanced_models(data)
    # At least the two neural-net variants; LightGBM adds two more if installed.
    assert len(models) >= 2
    for name, model in models.items():
        proba = model.predict_proba(data["X_test"])[:, 1]
        assert proba.shape[0] == len(data["X_test"])
        assert np.all((proba >= 0) & (proba <= 1)), name


def test_neural_network_returns_named_model(data):
    model, name = train_neural_network(
        data["X_train"], data["y_train"], use_class_weight=True
    )
    assert name == "NeuralNet_weighted"
    assert hasattr(model, "predict_proba")
