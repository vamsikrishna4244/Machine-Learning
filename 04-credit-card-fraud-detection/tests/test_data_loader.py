"""
Unit Tests for Credit Card Fraud Data Generator and Ingestion Loader.
"""

from pathlib import Path
import pytest
import pandas as pd
from src.data_loader import generate_credit_card_data, load_fraud_data, ALL_COLUMNS


def test_generate_credit_card_data_shape_and_columns():
    """Verify generated dataset attributes, column names, and fraud class balance."""
    df = generate_credit_card_data(n_samples=600, fraud_rate=0.01, seed=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 600
    for col in ALL_COLUMNS:
        assert col in df.columns
    assert set(df["Class"].unique()).issubset({0, 1})
    assert df["Class"].sum() > 0


def test_generate_credit_card_data_validations():
    """Verify ValueError on invalid sample counts and fraud rates."""
    with pytest.raises(ValueError, match="n_samples must be at least 500"):
        generate_credit_card_data(n_samples=200)

    with pytest.raises(ValueError, match="fraud_rate must be between"):
        generate_credit_card_data(n_samples=600, fraud_rate=0.50)


def test_generate_credit_card_data_save_file(tmp_path: Path):
    """Verify saving generated transactions directly to a CSV file."""
    out_file = tmp_path / "card_txns.csv"
    df = generate_credit_card_data(n_samples=500, seed=1, output_path=out_file)
    assert out_file.exists()
    loaded = pd.read_csv(out_file)
    assert len(loaded) == len(df)


def test_load_fraud_data_existing_and_missing(tmp_path: Path):
    """Verify loading from existing CSV and raising FileNotFoundError when missing."""
    csv_file = tmp_path / "sample_fraud.csv"
    df = generate_credit_card_data(n_samples=500, seed=99, output_path=csv_file)
    loaded = load_fraud_data(filepath=csv_file, ensure_exists=False)
    assert len(loaded) == len(df)

    fake_file = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        load_fraud_data(filepath=fake_file, ensure_exists=False)


def test_load_fraud_data_auto_generate(tmp_path: Path):
    """Verify auto-generation if CSV does not exist and ensure_exists=True."""
    auto_file = tmp_path / "auto_fraud.csv"
    df = load_fraud_data(filepath=auto_file, ensure_exists=True)
    assert auto_file.exists()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 500
