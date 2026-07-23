"""Stratified K-fold cross-validation runs and reports per-model summaries."""
import os

from cross_validation import run_stratified_cv


def test_run_stratified_cv(synthetic_csv, output_dir):
    summary_rows, fold_data = run_stratified_cv(
        data_path=synthetic_csv, n_splits=3, output_dir=output_dir
    )

    model_names = {row["Model"] for row in summary_rows}
    assert {"LogisticRegression", "RandomForest", "XGBoost"} <= model_names

    # Every model has one metrics row per fold.
    for name, df_folds in fold_data.items():
        assert len(df_folds) == 3
        assert {"AUPRC", "F1", "Recall", "Precision", "ROC-AUC"} <= set(df_folds.columns)

    assert os.path.exists(os.path.join(output_dir, "14_cross_validation.png"))
    assert os.path.exists(os.path.join(output_dir, "cv_results.csv"))
