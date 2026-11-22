"""
Data Preprocessing, Outlier-Resistant Scaling, and Leakage-Free Resampling.

Applies RobustScaler to Amount and Time to handle heavy outliers, enforces stratified
train-test splitting, and provides training-only resampling methods to prevent data leakage.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


FEATURE_NAMES = [f"V{i}" for i in range(1, 29)]
SCALED_COLS = ["Time", "Amount"]
TARGET_COL = "Class"


def preprocess_fraud_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clean transaction dataframe, remove duplicate records, and compute summary metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction records.

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

    # Validate no missing values
    cleaned = cleaned.dropna().reset_index(drop=True)

    fraud_count = int(cleaned[TARGET_COL].sum())
    legit_count = int((cleaned[TARGET_COL] == 0).sum())

    report = {
        "initial_records": initial_count,
        "final_records": len(cleaned),
        "duplicates_removed": dupes,
        "legitimate_count": legit_count,
        "fraud_count": fraud_count,
        "fraud_ratio_pct": round(float(fraud_count / len(cleaned) * 100.0), 3),
    }

    return cleaned, report


def build_transaction_preprocessor() -> ColumnTransformer:
    """Build Scikit-learn ColumnTransformer using RobustScaler for Amount and Time.

    RobustScaler uses median and interquartile range (IQR), making it resistant to
    extreme payment amount spikes. Features V1-V28 are passed through untouched.

    Returns
    -------
    ColumnTransformer
        Transformer configured for transaction features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("robust_scaler", RobustScaler(), SCALED_COLS),
        ],
        remainder="passthrough",  # V1-V28 are preserved as-is
    )
    return preprocessor


def prepare_stratified_train_test(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split transaction data into train and test sets with strict class stratification.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction records.
    test_size : float, default=0.20
        Fraction of data reserved for held-out evaluation.
    random_state : int, default=42
        Reproducible random seed.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols]
    y = df[TARGET_COL]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def resample_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "undersample",
    ratio: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Resample training data strictly on the train partition to avoid data leakage.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target labels.
    method : {'undersample', 'oversample'}, default='undersample'
        Resampling strategy.
    ratio : float, default=0.20
        Target minority-to-majority ratio (e.g. 0.20 means 1 fraud per 5 legitimate).
    random_state : int, default=42
        Reproducible random seed.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Resampled X_train_resampled, y_train_resampled
    """
    rng = np.random.default_rng(random_state)

    X_train_df = X_train.copy().reset_index(drop=True)
    y_train_s = y_train.copy().reset_index(drop=True)

    fraud_idx = y_train_s[y_train_s == 1].index.to_numpy()
    legit_idx = y_train_s[y_train_s == 0].index.to_numpy()

    n_fraud = len(fraud_idx)
    n_legit = len(legit_idx)

    if method == "undersample":
        # Keep all fraud, sample subset of legit
        target_legit = int(np.round(n_fraud / ratio))
        if target_legit < n_legit:
            sampled_legit_idx = rng.choice(legit_idx, size=target_legit, replace=False)
        else:
            sampled_legit_idx = legit_idx
        combined_idx = np.concatenate([fraud_idx, sampled_legit_idx])
    elif method == "oversample":
        # Keep all legit, duplicate fraud with small Gaussian noise to prevent exact copies
        target_fraud = int(np.round(n_legit * ratio))
        oversampled_fraud_idx = rng.choice(fraud_idx, size=target_fraud, replace=True)
        combined_idx = np.concatenate([legit_idx, oversampled_fraud_idx])
    else:
        raise ValueError(f"Unsupported resampling method '{method}'. Choose 'undersample' or 'oversample'.")

    rng.shuffle(combined_idx)
    return X_train_df.iloc[combined_idx].reset_index(drop=True), y_train_s.iloc[combined_idx].reset_index(drop=True)
