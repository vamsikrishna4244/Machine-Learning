"""
Data Preprocessing, Feature Standardization, and Dimensionality Reduction Module.

Standardizes customer behavioral attributes using StandardScaler to ensure equal distance weighting,
removes duplicate records, and computes 2D PCA projections for cluster visual interpretation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .data_loader import CUSTOMER_FEATURES


ID_COLUMN = "CustomerID"


def preprocess_customer_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clean raw customer records, remove duplicates, and validate feature bounds.

    Parameters
    ----------
    df : pd.DataFrame
        Raw customer behavior records.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, int]]
        Cleaned dataframe and diagnostic report.
    """
    initial_count = len(df)
    cleaned = df.copy()

    # Deduplication
    dupes = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # Missing value handling
    cleaned = cleaned.dropna().reset_index(drop=True)

    report = {
        "initial_records": initial_count,
        "final_records": len(cleaned),
        "duplicates_removed": dupes,
        "features_count": len(CUSTOMER_FEATURES),
    }

    return cleaned, report


def build_scaling_pipeline() -> StandardScaler:
    """Build StandardScaler for unsupervised clustering distance metrics.

    Returns
    -------
    StandardScaler
        Fitted or unfitted standard scaler.
    """
    return StandardScaler()


def scale_customer_features(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, StandardScaler]:
    """Standardize customer features to mean=0 and variance=1.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned customer records.
    features : List[str], optional
        Feature names to scale. Defaults to CUSTOMER_FEATURES.
    scaler : StandardScaler, optional
        Pre-fitted scaler instance, or None to fit a new one.

    Returns
    -------
    Tuple[np.ndarray, StandardScaler]
        Scaled feature matrix (X_scaled) and fitted StandardScaler.
    """
    if features is None:
        features = CUSTOMER_FEATURES

    X = df[features].to_numpy(dtype=float)

    if scaler is None:
        scaler = build_scaling_pipeline()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, scaler


def apply_pca_reduction(
    X_scaled: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, PCA, float]:
    """Project high-dimensional customer space onto 2D PCA coordinates for visualization.

    Parameters
    ----------
    X_scaled : np.ndarray
        Standardized feature matrix.
    n_components : int, default=2
        Number of principal components.
    random_state : int, default=42
        Reproducible random seed.

    Returns
    -------
    Tuple[np.ndarray, PCA, float]
        Transformed 2D coordinates, fitted PCA object, and total explained variance ratio.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    total_explained_variance = float(np.sum(pca.explained_variance_ratio_))

    return X_pca, pca, round(total_explained_variance, 4)
