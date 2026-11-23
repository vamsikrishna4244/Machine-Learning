"""
Unit Tests for Customer Data Ingestion and Dataset Generator Module.
"""

from pathlib import Path
import pytest
import pandas as pd
from src.data_loader import generate_customer_data, load_customer_data, CUSTOMER_FEATURES


def test_generate_customer_data_shape_and_columns():
    """Verify generated customer dataset rows, columns, and bounded ranges."""
    df = generate_customer_data(n_samples=300, seed=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 300
    assert "CustomerID" in df.columns
    for feat in CUSTOMER_FEATURES:
        assert feat in df.columns
        assert df[feat].isnull().sum() == 0


def test_generate_customer_data_minimum_samples_validation():
    """Verify ValueError when requesting fewer than 200 samples."""
    with pytest.raises(ValueError, match="n_samples must be at least 200"):
        generate_customer_data(n_samples=100)


def test_generate_customer_data_save_file(tmp_path: Path):
    """Verify saving customer dataset directly to a CSV path."""
    out_csv = tmp_path / "customers.csv"
    df = generate_customer_data(n_samples=250, seed=1, output_path=out_csv)
    assert out_csv.exists()
    loaded = pd.read_csv(out_csv)
    assert len(loaded) == len(df)


def test_load_customer_data_existing_and_missing(tmp_path: Path):
    """Verify loading from existing CSV and raising FileNotFoundError on missing file."""
    csv_file = tmp_path / "retail.csv"
    df = generate_customer_data(n_samples=200, seed=12, output_path=csv_file)
    loaded = load_customer_data(filepath=csv_file, ensure_exists=False)
    assert len(loaded) == len(df)

    fake_file = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        load_customer_data(filepath=fake_file, ensure_exists=False)


def test_load_customer_data_auto_generate(tmp_path: Path):
    """Verify auto-generation if CSV does not exist and ensure_exists=True."""
    auto_file = tmp_path / "auto.csv"
    df = load_customer_data(filepath=auto_file, ensure_exists=True)
    assert auto_file.exists()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 200
