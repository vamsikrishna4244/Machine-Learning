"""Unit tests for employee attrition data loader and synthetic generator."""

import pytest
import pandas as pd
from pathlib import Path
from src.data_loader import generate_attrition_dataset, load_attrition_data, DEPARTMENTS, JOB_ROLES


def test_generate_attrition_dataset_shape_and_columns():
    df = generate_attrition_dataset(n_samples=250, attrition_rate=0.16, seed=42)
    assert isinstance(df, pd.DataFrame)
    # n_samples + 10 duplicate rows injected
    assert len(df) == 260
    assert "EmployeeID" in df.columns
    assert "Attrition" in df.columns
    assert "MonthlyIncome" in df.columns
    assert "JobRole" in df.columns
    assert "Department" in df.columns
    assert set(df["Department"].unique()).issubset(set(DEPARTMENTS))


def test_generate_attrition_dataset_invalid_parameters():
    with pytest.raises(ValueError, match="n_samples must be at least 200"):
        generate_attrition_dataset(n_samples=50)

    with pytest.raises(ValueError, match="attrition_rate must be between 0.01 and 0.50"):
        generate_attrition_dataset(n_samples=250, attrition_rate=0.75)


def test_generate_attrition_dataset_calibration_branches():
    # Test high rate calibration
    df_high = generate_attrition_dataset(n_samples=300, attrition_rate=0.40, seed=123)
    rate_high = df_high["Attrition"].mean()
    assert 0.30 <= rate_high <= 0.50

    # Test low rate calibration
    df_low = generate_attrition_dataset(n_samples=300, attrition_rate=0.08, seed=456)
    rate_low = df_low["Attrition"].mean()
    assert 0.04 <= rate_low <= 0.15


def test_generate_attrition_dataset_file_output(tmp_path: Path):
    target_csv = tmp_path / "subfolder" / "test_attrition.csv"
    df = generate_attrition_dataset(n_samples=200, output_path=target_csv, seed=7)
    assert target_csv.exists()
    loaded = pd.read_csv(target_csv)
    assert len(loaded) == len(df)


def test_load_attrition_data_existing(tmp_path: Path):
    target_csv = tmp_path / "custom.csv"
    df_gen = generate_attrition_dataset(n_samples=200, output_path=target_csv, seed=1)
    df_loaded = load_attrition_data(filepath=target_csv, ensure_exists=False)
    assert len(df_loaded) == len(df_gen)


def test_load_attrition_data_missing_raises(tmp_path: Path):
    non_existent = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match="Attrition dataset not found"):
        load_attrition_data(filepath=non_existent, ensure_exists=False)


def test_load_attrition_data_ensure_exists(tmp_path: Path):
    non_existent = tmp_path / "auto_created.csv"
    df = load_attrition_data(filepath=non_existent, ensure_exists=True)
    assert non_existent.exists()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
