"""Tests for preprocessing: correct shapes, no data leakage, SMOTE behavior."""
import numpy as np

from preprocess import temporal_train_test_split, scale_features
import pandas as pd


def test_preprocess_keys_and_shapes(data):
    for key in [
        "X_train", "X_test", "y_train", "y_test",
        "X_train_smote", "y_train_smote", "amount_test", "feature_names",
    ]:
        assert key in data, f"missing key {key}"

    # X_train / X_test share the same feature columns and count.
    assert list(data["X_train"].columns) == list(data["X_test"].columns)
    assert data["X_train"].shape[1] == data["X_test"].shape[1]
    assert len(data["X_test"]) == len(data["y_test"]) == len(data["amount_test"])

    # Scaled columns exist; raw Time/Amount were dropped.
    cols = data["X_train"].columns
    assert "Time_scaled" in cols and "Amount_scaled" in cols
    assert "Time" not in cols and "Amount" not in cols


def test_no_nans(data):
    assert not data["X_train"].isnull().values.any()
    assert not data["X_test"].isnull().values.any()
    assert not np.isnan(data["amount_test"]).any()


def test_smote_balances_training_data(data):
    # SMOTE should oversample the minority class to parity, and only on train.
    y = data["y_train_smote"]
    assert (y == 0).sum() == (y == 1).sum()
    assert len(data["y_train_smote"]) >= len(data["y_train"])
    # Test set is untouched by SMOTE (still highly imbalanced).
    assert data["y_test"].mean() < 0.5


def test_scaler_fit_on_train_only(synthetic_csv):
    """The scaler must be fit on train only → train mean≈0, test mean≠0 exactly."""
    df = pd.read_csv(synthetic_csv)
    train_df, test_df = temporal_train_test_split(df, test_ratio=0.2)
    train_scaled, test_scaled, _, _ = scale_features(train_df, test_df)

    # Train scaled features are centered (fit on train).
    assert abs(train_scaled["Amount_scaled"].mean()) < 1e-6
    # Test scaled features are NOT forced to zero mean (they were only
    # transformed) - this proves no leakage / no refit on test.
    assert abs(test_scaled["Amount_scaled"].mean()) > 1e-9


def test_both_splits_contain_fraud(data):
    assert data["y_train"].sum() > 0
    assert data["y_test"].sum() > 0
