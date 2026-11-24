"""
Data Preprocessing, Cleaning, Feature Pipeline, and Stratified Splitting for Attrition.

Builds reusable Scikit-learn ColumnTransformer with StandardScaler and OneHotEncoder,
guaranteeing leak-free preprocessing and stratified train-test splits.
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
    "Age",
    "MonthlyIncome",
    "TotalWorkingYears",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "DistanceFromHome",
    "NumCompaniesWorked",
    "JobLevel",
    "StockOptionLevel",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "WorkLifeBalance",
    "JobInvolvement",
]

CATEGORICAL_FEATURES: List[str] = [
    "Gender",
    "MaritalStatus",
    "Department",
    "JobRole",
    "OverTime",
    "BusinessTravel",
]

TARGET_COLUMN = "Attrition"
ID_COLUMN = "EmployeeID"


def clean_attrition_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean raw employee records, remove duplicates, and validate columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw employee records.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Cleaned dataframe and diagnostic summary report.
    """
    initial_count = len(df)
    cleaned = df.copy()

    # Deduplication
    dupes = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # Missing value cleaning
    cleaned = cleaned.dropna().reset_index(drop=True)

    attrition_count = int(cleaned[TARGET_COLUMN].sum())
    retained_count = int((cleaned[TARGET_COLUMN] == 0).sum())

    report = {
        "initial_records": initial_count,
        "final_records": len(cleaned),
        "duplicates_removed": dupes,
        "retained_count": retained_count,
        "attrition_count": attrition_count,
        "attrition_rate_pct": round(float(attrition_count / len(cleaned) * 100.0), 2),
        "numerical_features_count": len(NUMERICAL_FEATURES),
        "categorical_features_count": len(CATEGORICAL_FEATURES),
    }

    return cleaned, report


def build_attrition_preprocessor() -> ColumnTransformer:
    """Build Scikit-learn ColumnTransformer for enterprise employee attrition.

    Applies median imputation and StandardScaler to continuous & ordinal features,
    and most-frequent imputation and OneHotEncoder to categorical features.

    Returns
    -------
    ColumnTransformer
        Preconfigured feature transformation pipeline.
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


def prepare_attrition_train_test(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset with class stratification to preserve positive attrition proportion.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned employee records.
    test_size : float, default=0.20
        Fraction reserved for held-out evaluation.
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

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
