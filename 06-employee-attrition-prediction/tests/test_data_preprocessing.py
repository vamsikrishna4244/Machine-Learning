"""Unit tests for employee attrition data preprocessing and feature transformations."""

import pytest
import pandas as pd
import numpy as np
from src.data_loader import generate_attrition_dataset
from src.data_preprocessing import (
    clean_attrition_data,
    build_attrition_preprocessor,
    prepare_attrition_train_test,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)


@pytest.fixture
def raw_dataset() -> pd.DataFrame:
    return generate_attrition_dataset(n_samples=250, seed=42)


def test_clean_attrition_data_deduplication(raw_dataset: pd.DataFrame):
    # raw_dataset has 10 injected duplicates
    cleaned, report = clean_attrition_data(raw_dataset)
    assert report["initial_records"] == 260
    assert report["duplicates_removed"] >= 10
    assert len(cleaned) == 250
    assert cleaned.duplicated().sum() == 0
    assert "attrition_rate_pct" in report


def test_clean_attrition_data_with_missing_values(raw_dataset: pd.DataFrame):
    dirty = raw_dataset.copy()
    dirty.loc[0, "MonthlyIncome"] = np.nan
    dirty.loc[1, "Department"] = np.nan

    cleaned, report = clean_attrition_data(dirty)
    assert cleaned["MonthlyIncome"].isna().sum() == 0
    assert cleaned["Department"].isna().sum() == 0


def test_build_attrition_preprocessor(raw_dataset: pd.DataFrame):
    cleaned, _ = clean_attrition_data(raw_dataset)
    preprocessor = build_attrition_preprocessor()
    X = cleaned[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

    transformed = preprocessor.fit_transform(X)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == len(cleaned)
    assert transformed.shape[1] >= len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)

    feature_names = preprocessor.get_feature_names_out()
    assert len(feature_names) == transformed.shape[1]


def test_prepare_attrition_train_test(raw_dataset: pd.DataFrame):
    cleaned, _ = clean_attrition_data(raw_dataset)
    X_train, X_test, y_train, y_test = prepare_attrition_train_test(cleaned, test_size=0.25, random_state=42)

    assert len(X_train) + len(X_test) == len(cleaned)
    assert abs(len(X_test) - int(len(cleaned) * 0.25)) <= 1
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert TARGET_COLUMN not in X_train.columns
    assert "EmployeeID" not in X_train.columns

    # Stratification check: proportions should be approximately equal
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    assert abs(train_rate - test_rate) < 0.05
