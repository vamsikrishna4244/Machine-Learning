"""
Unit Tests for Data Preprocessing, Standardization, and PCA Dimensionality Reduction.
"""

import pandas as pd
import numpy as np
from src.data_loader import generate_customer_data
from src.data_preprocessing import (
    preprocess_customer_data,
    scale_customer_features,
    apply_pca_reduction,
)


def test_preprocess_customer_data():
    """Verify deduplication, null removal, and diagnostic report dictionary."""
    raw = generate_customer_data(n_samples=250, seed=42)
    cleaned, report = preprocess_customer_data(raw)

    assert len(cleaned) < len(raw)  # Injected duplicates removed
    assert report["duplicates_removed"] > 0
    assert report["final_records"] == len(cleaned)
    assert report["features_count"] == 5


def test_scale_customer_features():
    """Verify StandardScaler normalizes features to mean ~ 0 and std ~ 1."""
    raw = generate_customer_data(n_samples=300, seed=42)
    cleaned, _ = preprocess_customer_data(raw)

    X_scaled, scaler = scale_customer_features(cleaned)
    assert isinstance(X_scaled, np.ndarray)
    assert X_scaled.shape[0] == len(cleaned)
    assert X_scaled.shape[1] == 5

    # Check zero mean and unit variance
    means = np.mean(X_scaled, axis=0)
    stds = np.std(X_scaled, axis=0)
    assert np.allclose(means, 0.0, atol=1e-2)
    assert np.allclose(stds, 1.0, atol=1e-2)


def test_apply_pca_reduction():
    """Verify 2D PCA projection coordinates and explained variance ratio."""
    raw = generate_customer_data(n_samples=300, seed=42)
    cleaned, _ = preprocess_customer_data(raw)
    X_scaled, _ = scale_customer_features(cleaned)

    X_pca, pca_model, exp_var = apply_pca_reduction(X_scaled, n_components=2)
    assert X_pca.shape[0] == len(cleaned)
    assert X_pca.shape[1] == 2
    assert 0.50 <= exp_var <= 1.00
