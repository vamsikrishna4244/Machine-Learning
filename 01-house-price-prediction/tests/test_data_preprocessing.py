"""
Unit tests for data preprocessing and pipeline module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.data_preprocessing import (
    clean_housing_data,
    detect_and_handle_outliers,
    build_preprocessing_pipeline,
    prepare_train_test_data,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    records = [
        {
            "bedrooms": 3,
            "bathrooms": 2.0,
            "sqft_living": 2000.0,
            "sqft_lot": 6000.0,
            "floors": 1.0,
            "waterfront": 0,
            "view_quality": 1,
            "condition": 3,
            "grade": 7,
            "yr_built": 1990,
            "yr_renovated": np.nan,
            "neighborhood": "Suburbs",
            "price": 450000.0,
        },
        {
            "bedrooms": 4,
            "bathrooms": 3.0,
            "sqft_living": 3200.0,
            "sqft_lot": 8000.0,
            "floors": 2.0,
            "waterfront": 1,
            "view_quality": 3,
            "condition": 4,
            "grade": 9,
            "yr_built": 2010,
            "yr_renovated": 2018.0,
            "neighborhood": "Downtown",
            "price": 850000.0,
        },
        # Duplicate record
        {
            "bedrooms": 3,
            "bathrooms": 2.0,
            "sqft_living": 2000.0,
            "sqft_lot": 6000.0,
            "floors": 1.0,
            "waterfront": 0,
            "view_quality": 1,
            "condition": 3,
            "grade": 7,
            "yr_built": 1990,
            "yr_renovated": np.nan,
            "neighborhood": "Suburbs",
            "price": 450000.0,
        },
    ]
    return pd.DataFrame(records)


def test_clean_housing_data(sample_raw_data: pd.DataFrame):
    cleaned, report = clean_housing_data(sample_raw_data)
    assert report["duplicates_removed"] == 1
    assert len(cleaned) == 2
    assert "house_age" in cleaned.columns
    assert "is_renovated" in cleaned.columns
    assert "sqft_per_bedroom" in cleaned.columns
    assert "bath_to_bed_ratio" in cleaned.columns
    assert not cleaned["yr_renovated"].isnull().any()


def test_detect_and_handle_outliers():
    df = pd.DataFrame({
        "sqft_living": [1000.0, 1100.0, 1200.0, 1150.0, 1050.0, 1120.0, 15000.0],
        "sqft_lot": [4000.0, 4200.0, 4100.0, 4300.0, 4050.0, 4150.0, 90000.0],
        "price": [200000.0, 210000.0, 205000.0, 215000.0, 202000.0, 208000.0, 3000000.0],
    })
    # Test capping
    capped, summary = detect_and_handle_outliers(df, iqr_multiplier=1.5, action="cap")
    assert summary["sqft_living"] == 1
    assert capped["sqft_living"].max() < 15000.0
    assert len(capped) == len(df)

    # Test dropping
    dropped, summary_drop = detect_and_handle_outliers(df, iqr_multiplier=1.5, action="drop")
    assert len(dropped) < len(df)


def test_build_preprocessing_pipeline(sample_raw_data: pd.DataFrame):
    cleaned, _ = clean_housing_data(sample_raw_data)
    pipeline = build_preprocessing_pipeline()
    transformed = pipeline.fit_transform(cleaned)
    assert isinstance(transformed, np.ndarray)
    # Numerical (13) + OneHot categories
    assert transformed.shape[0] == len(cleaned)
    assert transformed.shape[1] >= len(NUMERICAL_FEATURES)


def test_prepare_train_test_data(sample_raw_data: pd.DataFrame):
    cleaned, _ = clean_housing_data(sample_raw_data)
    # duplicate to have enough for split
    larger_df = pd.concat([cleaned] * 5, ignore_index=True)
    X_train, X_test, y_train, y_test = prepare_train_test_data(larger_df, test_size=0.2, random_state=42)
    assert len(X_train) == 8
    assert len(X_test) == 2
    assert "price" not in X_train.columns
    assert len(y_train) == 8
