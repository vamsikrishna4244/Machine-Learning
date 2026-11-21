"""
Unit Tests for Email Dataset Generation and Loader Module.
"""

from pathlib import Path
import pytest
import pandas as pd
from src.data_loader import generate_spam_dataset, load_email_data


def test_generate_spam_dataset_shape_and_columns():
    """Verify generated dataset attributes and label encoding."""
    df = generate_spam_dataset(n_samples=150, spam_ratio=0.20, seed=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 150  # includes injected duplicate test rows
    assert "subject" in df.columns
    assert "body" in df.columns
    assert "message" in df.columns
    assert "label" in df.columns
    assert set(df["label"].unique()).issubset({0, 1})


def test_generate_spam_dataset_validations():
    """Verify input validation checks for sample count and ratio."""
    with pytest.raises(ValueError, match="n_samples must be at least 100"):
        generate_spam_dataset(n_samples=50)

    with pytest.raises(ValueError, match="spam_ratio must be between 0.01 and 0.99"):
        generate_spam_dataset(n_samples=150, spam_ratio=1.5)


def test_generate_spam_dataset_save_file(tmp_path: Path):
    """Verify saving generated emails directly to a CSV path."""
    out_csv = tmp_path / "emails.csv"
    df = generate_spam_dataset(n_samples=100, seed=1, output_path=out_csv)
    assert out_csv.exists()
    loaded = pd.read_csv(out_csv)
    assert len(loaded) == len(df)


def test_load_email_data_existing_and_missing(tmp_path: Path):
    """Verify loading from existing file and raising on missing when ensure_exists=False."""
    csv_file = tmp_path / "corpus.csv"
    df = generate_spam_dataset(n_samples=100, seed=12, output_path=csv_file)
    loaded = load_email_data(filepath=csv_file, ensure_exists=False)
    assert len(loaded) == len(df)

    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_email_data(filepath=missing_path, ensure_exists=False)


def test_load_email_data_auto_generate(tmp_path: Path):
    """Verify automatic dataset generation when filepath is missing and ensure_exists=True."""
    auto_path = tmp_path / "auto.csv"
    df = load_email_data(filepath=auto_path, ensure_exists=True)
    assert auto_path.exists()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 100
