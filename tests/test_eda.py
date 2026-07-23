"""EDA runs end-to-end and produces its expected plot artifacts."""
import os

from eda import run_eda


def test_run_eda_produces_plots(synthetic_csv, output_dir):
    df = run_eda(data_path=synthetic_csv, output_dir=output_dir)
    assert df is not None and len(df) > 0

    expected = [
        "01_class_distribution.png",
        "02_amount_distribution.png",
        "03_time_distribution.png",
        "04_feature_distributions.png",
        "05_correlation_with_class.png",
    ]
    for fname in expected:
        assert os.path.exists(os.path.join(output_dir, fname)), fname

    # EDA must not mutate the frame it hands back (Hour column is temporary).
    assert "Hour" not in df.columns
