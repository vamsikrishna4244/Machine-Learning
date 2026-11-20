"""
Unit Tests for Data Cleaning, Feature Engineering, and Transformation Pipelines.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from src.data_loader import generate_churn_dataset
from src.data_preprocessing import (
    clean_churn_data,
    build_churn_preprocessing_pipeline,
    prepare_churn_train_test_data,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)


def test_clean_churn_data_deduplication_and_imputation():
    """Verify deduplication, missing total_charges imputation, and feature engineering."""
    raw_df = generate_churn_dataset(n_samples=200, seed=42)
    cleaned, report = clean_churn_data(raw_df)

    assert len(cleaned) < len(raw_df)  # Deduplication removed injected copies
    assert report["duplicates_removed"] > 0
    assert cleaned["total_charges"].isnull().sum() == 0  # Imputation resolved NaNs
    assert "tenure_years" in cleaned.columns
    assert "monthly_to_total_ratio" in cleaned.columns
    assert "services_count" in cleaned.columns
    assert "has_tech_and_security" in cleaned.columns


def test_build_churn_preprocessing_pipeline():
    """Verify ColumnTransformer composition and transformer configurations."""
    preprocessor = build_churn_preprocessing_pipeline()
    assert isinstance(preprocessor, ColumnTransformer)
    trans_names = [name for name, _, _ in preprocessor.transformers]
    assert "num" in trans_names
    assert "cat" in trans_names


def test_preprocessing_pipeline_fit_transform():
    """Verify that the ColumnTransformer transforms sample customer records without NaNs."""
    raw_df = generate_churn_dataset(n_samples=150, seed=42)
    cleaned, _ = clean_churn_data(raw_df)
    preprocessor = build_churn_preprocessing_pipeline()

    transformed = preprocessor.fit_transform(cleaned)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == len(cleaned)
    assert not np.isnan(transformed).any()


def test_prepare_churn_train_test_data_stratification():
    """Verify train-test split sizes and preserved class proportions."""
    raw_df = generate_churn_dataset(n_samples=300, seed=42)
    cleaned, _ = clean_churn_data(raw_df)
    X_train, X_test, y_train, y_test = prepare_churn_train_test_data(
        cleaned, test_size=0.25, random_state=42
    )

    total_samples = len(cleaned)
    assert len(X_train) == int(np.round(total_samples * 0.75))
    assert len(X_test) == len(cleaned) - len(X_train)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    # Verify stratified balance (churn rate within 2% margin)
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    assert abs(train_rate - test_rate) < 0.03
