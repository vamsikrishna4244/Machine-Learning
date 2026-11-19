"""
Unit tests for data loader and generator module.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import pandas as pd
from src.data_loader import (
    generate_housing_dataset,
    load_housing_data,
    NEIGHBORHOODS,
)


def test_generate_housing_dataset_shape_and_columns():
    df = generate_housing_dataset(n_samples=100, seed=123)
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 100
    expected_cols = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "view_quality",
        "condition",
        "grade",
        "yr_built",
        "yr_renovated",
        "neighborhood",
        "price",
    ]
    for col in expected_cols:
        assert col in df.columns
    assert set(df["neighborhood"].unique()).issubset(set(NEIGHBORHOODS))


def test_generate_housing_dataset_invalid_samples():
    with pytest.raises(ValueError, match="n_samples must be at least 50"):
        generate_housing_dataset(n_samples=20)


def test_generate_housing_dataset_file_persistence(tmp_path: Path):
    target = tmp_path / "test_data.csv"
    df = generate_housing_dataset(n_samples=60, output_path=target)
    assert target.exists()
    loaded = pd.read_csv(target)
    assert len(loaded) == len(df)


def test_load_housing_data_file_not_found(tmp_path: Path):
    missing_file = tmp_path / "non_existent.csv"
    with pytest.raises(FileNotFoundError):
        load_housing_data(filepath=missing_file, ensure_exists=False)


def test_load_housing_data_ensure_exists(tmp_path: Path):
    target = tmp_path / "auto_gen.csv"
    df = load_housing_data(filepath=target, ensure_exists=True)
    assert target.exists()
    assert len(df) == 2512  # 2500 + 12 duplicates
