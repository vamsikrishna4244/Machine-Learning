"""
Unit Tests for Data Preprocessing, Scaling, and Resampling.
"""

import pandas as pd
import numpy as np
import pytest
from src.data_loader import generate_credit_card_data
from src.data_preprocessing import (
    preprocess_fraud_data,
    build_transaction_preprocessor,
    prepare_stratified_train_test,
    resample_training_data,
)


def test_preprocess_fraud_data():
    """Verify deduplication, null removal, and diagnostic summary statistics."""
    raw = generate_credit_card_data(n_samples=600, fraud_rate=0.02, seed=42)
    cleaned, report = preprocess_fraud_data(raw)

    assert len(cleaned) < len(raw)  # Injected duplicates removed
    assert report["duplicates_removed"] > 0
    assert report["fraud_count"] > 0
    assert report["legitimate_count"] > 0
    assert report["final_records"] == len(cleaned)


def test_build_transaction_preprocessor():
    """Verify ColumnTransformer configuration and transform outputs."""
    preprocessor = build_transaction_preprocessor()
    raw = generate_credit_card_data(n_samples=500, seed=42)
    cleaned, _ = preprocess_fraud_data(raw)
    feature_cols = [c for c in cleaned.columns if c != "Class"]

    transformed = preprocessor.fit_transform(cleaned[feature_cols])
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == len(cleaned)
    assert transformed.shape[1] == len(feature_cols)


def test_prepare_stratified_train_test():
    """Verify stratified split sizes and preserved minority fraud proportion."""
    raw = generate_credit_card_data(n_samples=700, fraud_rate=0.03, seed=42)
    cleaned, _ = preprocess_fraud_data(raw)

    X_train, X_test, y_train, y_test = prepare_stratified_train_test(cleaned, test_size=0.25, random_state=42)
    assert len(X_train) + len(X_test) == len(cleaned)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    train_rate = y_train.mean()
    test_rate = y_test.mean()
    assert abs(train_rate - test_rate) < 0.01


def test_resample_training_data_undersample_and_oversample():
    """Verify training-only resampling mechanisms without leaking test samples."""
    raw = generate_credit_card_data(n_samples=600, fraud_rate=0.02, seed=42)
    cleaned, _ = preprocess_fraud_data(raw)
    X_train, _, y_train, _ = prepare_stratified_train_test(cleaned, test_size=0.20, random_state=42)

    # 1. Undersampling
    X_u, y_u = resample_training_data(X_train, y_train, method="undersample", ratio=0.20)
    assert len(X_u) < len(X_train)
    assert y_u.mean() > y_train.mean()

    # 2. Oversampling
    X_o, y_o = resample_training_data(X_train, y_train, method="oversample", ratio=0.10)
    assert len(X_o) > len(X_train)
    assert y_o.mean() > y_train.mean()

    # 3. Invalid method raises ValueError
    with pytest.raises(ValueError, match="Unsupported resampling method"):
        resample_training_data(X_train, y_train, method="invalid")
