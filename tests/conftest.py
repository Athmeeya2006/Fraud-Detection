"""
Shared pytest fixtures.

The real Kaggle `creditcard.csv` (~150 MB) is intentionally NOT committed, so
the tests run against a small SYNTHETIC dataset with the exact same schema
(Time, V1..V28, Amount, Class). This keeps CI fast and self-contained while
still exercising every real code path in the pipeline.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make the project root importable regardless of where pytest is invoked from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def make_synthetic_frame(n_rows=3000, n_fraud=45, seed=42):
    """
    Build a creditcard-like DataFrame.

    Fraud is made mildly separable (a few V-features are shifted) so the
    models produce non-degenerate scores, and fraud is spread across the whole
    time range so BOTH halves of a temporal split contain fraud.
    """
    rng = np.random.default_rng(seed)

    # 48 hours of transactions, unsorted times (preprocess will sort them).
    time = rng.uniform(0, 172_800, size=n_rows)
    v_features = rng.standard_normal((n_rows, 28))
    amount = np.abs(rng.gamma(shape=2.0, scale=40.0, size=n_rows))
    label = np.zeros(n_rows, dtype=int)

    # Pick fraud rows spread across the ordering so a temporal split sees fraud
    # on both sides.
    fraud_idx = rng.choice(n_rows, size=n_fraud, replace=False)
    label[fraud_idx] = 1
    # Shift a handful of features + inflate amount for fraud → learnable signal.
    v_features[fraud_idx, 13] -= 4.0   # mimics the real V14 fraud signal
    v_features[fraud_idx, 3] += 3.0
    v_features[fraud_idx, 11] -= 3.0
    amount[fraud_idx] *= 3.0

    cols = {"Time": time}
    for i in range(28):
        cols[f"V{i + 1}"] = v_features[:, i]
    cols["Amount"] = amount
    cols["Class"] = label
    return pd.DataFrame(cols)


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory):
    """Write the synthetic dataset to a temp CSV and return its path."""
    path = tmp_path_factory.mktemp("data") / "creditcard.csv"
    make_synthetic_frame().to_csv(path, index=False)
    return str(path)


@pytest.fixture(scope="session")
def data(synthetic_csv):
    """Run the real preprocessing pipeline on the synthetic CSV."""
    from preprocess import preprocess

    return preprocess(data_path=synthetic_csv)


@pytest.fixture()
def output_dir(tmp_path):
    """A throwaway output directory for plot/CSV artifacts."""
    d = tmp_path / "outputs"
    d.mkdir()
    return str(d)


@pytest.fixture(scope="session")
def base_models(data):
    """All six base-model variants trained once for the whole test session."""
    from models import train_all_models

    return train_all_models(data)


@pytest.fixture(scope="session")
def tree_model(data):
    """A single lightweight tree model reused by SHAP / cost / real-time tests."""
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=40, max_depth=8, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    model.fit(data["X_train"], data["y_train"])
    return model, "RandomForest_test"
