"""
Model Training, Cost-Weighted Classifiers, and Pipeline Serialization for Fraud Detection.

Constructs and trains baseline unweighted, cost-sensitive balanced, and resampled
classification pipelines across Logistic Regression, Decision Tree, and Random Forest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .data_preprocessing import build_transaction_preprocessor, resample_training_data


def build_fraud_pipeline(classifier: BaseEstimator) -> Pipeline:
    """Compose RobustScaler preprocessor and classifier into an end-to-end Pipeline.

    Parameters
    ----------
    classifier : BaseEstimator
        Scikit-learn classification estimator.

    Returns
    -------
    Pipeline
        End-to-end transaction scoring pipeline.
    """
    preprocessor = build_transaction_preprocessor()
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def train_fraud_classifiers(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Dict[str, Pipeline]:
    """Train candidate classification models under different balancing and algorithmic regimes.

    Models Trained:
    1. Baseline Logistic Regression (Unweighted - demonstrates failure of nominal accuracy)
    2. Weighted Logistic Regression (class_weight="balanced")
    3. Undersampled Logistic Regression (resampled majority class)
    4. Balanced Decision Tree
    5. Balanced Random Forest

    Parameters
    ----------
    X_train : pd.DataFrame
        Training transaction features.
    y_train : pd.Series
        Training labels (0 = Legit, 1 = Fraud).
    random_state : int, default=42
        Reproducible random seed.

    Returns
    -------
    Dict[str, Pipeline]
        Dictionary of trained model pipelines.
    """
    pipelines: Dict[str, Pipeline] = {}

    # 1. Baseline Logistic Regression (unweighted)
    pipe_base_lr = build_fraud_pipeline(
        LogisticRegression(max_iter=1000, random_state=random_state)
    )
    pipe_base_lr.fit(X_train, y_train)
    pipelines["Baseline Logistic Regression"] = pipe_base_lr

    # 2. Weighted Logistic Regression (class_weight="balanced")
    pipe_weighted_lr = build_fraud_pipeline(
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)
    )
    pipe_weighted_lr.fit(X_train, y_train)
    pipelines["Weighted Logistic Regression"] = pipe_weighted_lr

    # 3. Undersampled Logistic Regression
    X_train_under, y_train_under = resample_training_data(
        X_train, y_train, method="undersample", ratio=0.10, random_state=random_state
    )
    pipe_under_lr = build_fraud_pipeline(
        LogisticRegression(max_iter=1000, random_state=random_state)
    )
    pipe_under_lr.fit(X_train_under, y_train_under)
    pipelines["Undersampled Logistic Regression"] = pipe_under_lr

    # 4. Balanced Decision Tree
    pipe_dt = build_fraud_pipeline(
        DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
        )
    )
    pipe_dt.fit(X_train, y_train)
    pipelines["Balanced Decision Tree"] = pipe_dt

    # 5. Balanced Random Forest
    pipe_rf = build_fraud_pipeline(
        RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
    )
    pipe_rf.fit(X_train, y_train)
    pipelines["Balanced Random Forest"] = pipe_rf

    return pipelines


def save_best_fraud_model(
    pipeline: Pipeline,
    model_name: str,
    metrics: Dict[str, float],
    output_dir: Optional[Path | str] = None,
) -> Path:
    """Serialize the best-performing fraud detection pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        Trained Scikit-learn Pipeline.
    model_name : str
        Algorithm name.
    metrics : Dict[str, float]
        Evaluation metric dictionary.
    output_dir : Path or str, optional
        Target directory. Defaults to `models/`.

    Returns
    -------
    Path
        Path to saved Joblib artifact.
    """
    if output_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root / "models"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "best_fraud_detector.joblib"

    artifact = {
        "model_name": model_name,
        "pipeline": pipeline,
        "metrics": metrics,
    }
    joblib.dump(artifact, target_path)
    return target_path


def load_saved_fraud_model(model_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Load serialized credit card fraud detection pipeline from disk.

    Parameters
    ----------
    model_path : Path or str, optional
        Path to artifact. Defaults to `models/best_fraud_detector.joblib`.

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'model_name', 'pipeline', and 'metrics'.
    """
    if model_path is None:
        project_root = Path(__file__).resolve().parent.parent
        model_path = project_root / "models" / "best_fraud_detector.joblib"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Fraud detection model not found at: {model_path.resolve()}")

    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError("Loaded artifact does not contain a valid pipeline dictionary.")

    return artifact
