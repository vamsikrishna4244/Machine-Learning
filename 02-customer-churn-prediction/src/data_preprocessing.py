"""
Data Preprocessing, Cleaning, Feature Engineering, and Transformation Pipelines.

Prepares subscriber data for binary classification, builds Scikit-learn ColumnTransformer
pipelines with OneHotEncoding and StandardScaling, and guarantees stratified train-test splits.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERICAL_FEATURES: List[str] = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "tenure_years",
    "services_count",
    "monthly_to_total_ratio",
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    "has_tech_and_security",
]

TARGET_COLUMN: str = "churn"
ID_COLUMN: str = "customer_id"


def clean_churn_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clean raw customer churn data, resolve duplicates, and engineer domain features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw subscriber records.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, int]]
        Cleaned dataframe and diagnostic report dictionary.
    """
    initial_count = len(df)
    cleaned = df.copy()

    # 1. Deduplication
    dupes = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # 2. Total charges imputation (handle blank strings or NaNs)
    if "total_charges" in cleaned.columns:
        cleaned["total_charges"] = pd.to_numeric(cleaned["total_charges"], errors="coerce")
        # Impute missing total_charges as monthly_charges * tenure_months
        impute_mask = cleaned["total_charges"].isnull()
        cleaned.loc[impute_mask, "total_charges"] = (
            cleaned.loc[impute_mask, "monthly_charges"] * cleaned.loc[impute_mask, "tenure_months"]
        )

    # 3. Domain Feature Engineering
    cleaned["tenure_years"] = np.round(cleaned["tenure_months"] / 12.0, 2)
    cleaned["monthly_to_total_ratio"] = np.round(
        cleaned["monthly_charges"] / (cleaned["total_charges"] + 1.0), 4
    )

    # Active services count
    service_cols = [
        "phone_service",
        "multiple_lines",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ]
    cleaned["services_count"] = 0
    for col in service_cols:
        if col in cleaned.columns:
            cleaned["services_count"] += (cleaned[col] == "Yes").astype(int)

    # High-retention bundled feature: Tech Support + Online Security
    cleaned["has_tech_and_security"] = np.where(
        (cleaned["online_security"] == "Yes") & (cleaned["tech_support"] == "Yes"),
        "Yes",
        "No",
    )

    report = {
        "initial_records": initial_count,
        "final_records": len(cleaned),
        "duplicates_removed": dupes,
        "churn_rate_pct": round(float(cleaned[TARGET_COLUMN].mean() * 100.0), 2),
    }

    return cleaned, report


def build_churn_preprocessing_pipeline() -> ColumnTransformer:
    """Build a Scikit-learn ColumnTransformer for customer churn classification.

    Applies median imputation and standard scaling to continuous variables,
    and most-frequent imputation and one-hot encoding to categorical variables.

    Returns
    -------
    ColumnTransformer
        Configured feature transformation pipeline.
    """
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_churn_train_test_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset with class stratification to preserve positive churn proportion.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned customer records.
    test_size : float, default=0.2
        Fraction of samples reserved for test set.
    random_state : int, default=42
        Reproducible random seed.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    feature_cols = [c for c in df.columns if c not in [TARGET_COLUMN, ID_COLUMN]]
    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test
