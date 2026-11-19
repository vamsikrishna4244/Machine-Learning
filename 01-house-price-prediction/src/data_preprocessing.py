"""
Data Preprocessing, Cleaning, Outlier Treatment, and Scikit-learn Pipeline Module.

Implements robust data sanitization, IQR boundary outlier treatment, feature engineering,
and reusable ColumnTransformer pipelines for tabular housing regression.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


NUMERICAL_FEATURES: List[str] = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view_quality",
    "condition",
    "grade",
    "house_age",
    "is_renovated",
    "sqft_per_bedroom",
    "bath_to_bed_ratio",
]

CATEGORICAL_FEATURES: List[str] = [
    "neighborhood",
]

TARGET_COLUMN: str = "price"


def clean_housing_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Deduplicate records, impute missing values, and validate schema integrity.

    Parameters
    ----------
    df : pd.DataFrame
        Raw housing records.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, int]]
        Cleaned dataframe and diagnostic cleaning report dictionary.
    """
    initial_rows = len(df)
    cleaned = df.copy()

    # 1. Deduplication
    duplicates_count = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # 2. Missing value detection & imputation
    missing_before = cleaned.isnull().sum().to_dict()
    if "yr_renovated" in cleaned.columns:
        # Default unrenovated houses to 0
        cleaned["yr_renovated"] = cleaned["yr_renovated"].fillna(0.0)

    # Impute any remaining numeric NaNs with median
    num_cols = cleaned.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if cleaned[c].isnull().any():
            cleaned[c] = cleaned[c].fillna(cleaned[c].median())

    # 3. Feature Engineering
    current_year = 2024
    cleaned["house_age"] = (current_year - cleaned["yr_built"]).clip(lower=0)
    cleaned["is_renovated"] = (cleaned["yr_renovated"] > 0).astype(int)
    cleaned["sqft_per_bedroom"] = (
        cleaned["sqft_living"] / cleaned["bedrooms"].clip(lower=1)
    ).round(2)
    cleaned["bath_to_bed_ratio"] = (
        cleaned["bathrooms"] / cleaned["bedrooms"].clip(lower=1)
    ).round(2)

    report = {
        "initial_rows": initial_rows,
        "final_rows": len(cleaned),
        "duplicates_removed": duplicates_count,
        "missing_values_handled": sum(missing_before.values()),
    }
    return cleaned, report


def detect_and_handle_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    iqr_multiplier: float = 2.5,
    action: str = "cap",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Identify and treat extreme numerical outliers via Interquartile Range (IQR) boundaries.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to inspect.
    columns : List[str], optional
        Columns to evaluate. Defaults to ['sqft_living', 'sqft_lot', 'price'].
    iqr_multiplier : float, default=2.5
        IQR threshold multiplier (2.5 preserves legitimate luxury real estate while trimming errors).
    action : {"cap", "drop"}, default="cap"
        Whether to winsorize (cap at boundaries) or drop outlier rows.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, int]]
        Treated dataframe and count of outliers modified per column.
    """
    if columns is None:
        columns = [c for c in ["sqft_living", "sqft_lot", "price"] if c in df.columns]

    processed = df.copy()
    outlier_summary: Dict[str, int] = {}

    for col in columns:
        q25 = processed[col].quantile(0.25)
        q75 = processed[col].quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - (iqr_multiplier * iqr)
        upper_bound = q75 + (iqr_multiplier * iqr)

        outlier_mask = (processed[col] < lower_bound) | (processed[col] > upper_bound)
        count = int(outlier_mask.sum())
        outlier_summary[col] = count

        if action == "cap":
            processed[col] = processed[col].clip(lower=lower_bound, upper=upper_bound)
        elif action == "drop":
            processed = processed[~outlier_mask].reset_index(drop=True)

    return processed, outlier_summary


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Build a reusable Scikit-learn ColumnTransformer for housing feature preprocessing.

    Numerical pipeline: SimpleImputer(median) -> StandardScaler
    Categorical pipeline: SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore')

    Returns
    -------
    ColumnTransformer
        Preconfigured Scikit-learn transformer.
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


def prepare_train_test_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Extract features and target, and perform reproducible train-test split.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned housing dataframe.
    test_size : float, default=0.2
        Fraction of records reserved for test set.
    random_state : int, default=42
        Reproducible random seed.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
