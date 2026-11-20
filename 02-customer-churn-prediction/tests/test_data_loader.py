"""
Unit Tests for Data Ingestion and Dataset Generation Module.
"""

from pathlib import Path
import pytest
import pandas as pd
from src.data_loader import generate_churn_dataset, load_churn_data


def test_generate_churn_dataset_shape_and_columns():
    """Verify generated dataset contains expected rows and essential features."""
    df = generate_churn_dataset(n_samples=200, seed=123)
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 200  # includes injected duplicates
    assert "churn" in df.columns
    assert "monthly_charges" in df.columns
    assert "tenure_months" in df.columns
    assert "contract" in df.columns
    assert "payment_method" in df.columns
    assert "internet_service" in df.columns


def test_generate_churn_dataset_minimum_samples_validation():
    """Verify that requesting fewer than 100 samples raises ValueError."""
    with pytest.raises(ValueError, match="n_samples must be at least 100"):
        generate_churn_dataset(n_samples=50)


def test_generate_churn_dataset_reproducibility():
    """Verify that identical seeds produce deterministic datasets."""
    df1 = generate_churn_dataset(n_samples=150, seed=42)
    df2 = generate_churn_dataset(n_samples=150, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_churn_dataset_save_to_file(tmp_path: Path):
    """Verify that specifying output_path saves CSV to disk."""
    out_file = tmp_path / "subfolder" / "test_churn.csv"
    df = generate_churn_dataset(n_samples=120, seed=99, output_path=out_file)
    assert out_file.exists()
    loaded = pd.read_csv(out_file)
    assert len(loaded) == len(df)


def test_load_churn_data_from_existing_file(tmp_path: Path):
    """Verify loading from an existing CSV file."""
    csv_file = tmp_path / "sample.csv"
    df = generate_churn_dataset(n_samples=100, seed=1, output_path=csv_file)
    loaded = load_churn_data(filepath=csv_file, ensure_exists=False)
    assert len(loaded) == len(df)


def test_load_churn_data_missing_file_raises_error(tmp_path: Path):
    """Verify that FileNotFoundError is raised when file is missing and ensure_exists=False."""
    missing = tmp_path / "non_existent.csv"
    with pytest.raises(FileNotFoundError):
        load_churn_data(filepath=missing, ensure_exists=False)


def test_load_churn_data_auto_generate_if_missing(tmp_path: Path):
    """Verify auto-generation if filepath does not exist and ensure_exists=True."""
    missing = tmp_path / "auto_generated.csv"
    loaded = load_churn_data(filepath=missing, ensure_exists=True)
    assert missing.exists()
    assert isinstance(loaded, pd.DataFrame)
    assert len(loaded) >= 100
