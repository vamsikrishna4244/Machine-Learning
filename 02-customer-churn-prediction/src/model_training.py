"""
Classification Model Training and Serialization Module for Customer Churn.

Builds and trains Logistic Regression, Decision Tree Classifier, and Random Forest Classifier
pipelines with balanced class weighting to address churn class imbalance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .data_preprocessing import build_churn_preprocessing_pipeline


def build_churn_model_pipeline(classifier: BaseEstimator) -> Pipeline:
    """Compose preprocessing ColumnTransformer and classifier into a Scikit-learn Pipeline.

    Parameters
    ----------
    classifier : BaseEstimator
        Scikit-learn classifier.

    Returns
    -------
    Pipeline
        End-to-end classification pipeline.
    """
    preprocessor = build_churn_preprocessing_pipeline()
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def train_churn_classifiers(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Dict[str, Pipeline]:
    """Train candidate classification models under identical stratified cross-validation folds.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target labels (0 = Retained, 1 = Churned).
    random_state : int, default=42
        Reproducible random seed.

    Returns
    -------
    Dict[str, Pipeline]
        Dictionary of trained model pipelines.
    """
    candidates: Dict[str, BaseEstimator] = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            min_samples_split=12,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
    }

    trained: Dict[str, Pipeline] = {}
    for name, estimator in candidates.items():
        pipe = build_churn_model_pipeline(estimator)
        pipe.fit(X_train, y_train)
        trained[name] = pipe

    return trained


def save_best_churn_model(
    pipeline: Pipeline,
    model_name: str,
    metrics: Dict[str, float],
    output_dir: Optional[Path | str] = None,
) -> Path:
    """Serialize the best-performing churn pipeline along with evaluation metadata.

    Parameters
    ----------
    pipeline : Pipeline
        Trained Scikit-learn Pipeline.
    model_name : str
        Algorithm name.
    metrics : Dict[str, float]
        Evaluation score dictionary (accuracy, precision, recall, f1, roc_auc).
    output_dir : Path or str, optional
        Target directory. Defaults to `models/`.

    Returns
    -------
    Path
        Path to persisted Joblib artifact.
    """
    if output_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root / "models"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / "best_customer_churn_model.joblib"

    artifact = {
        "model_name": model_name,
        "pipeline": pipeline,
        "metrics": metrics,
    }
    joblib.dump(artifact, filepath)
    return filepath


def load_saved_churn_model(model_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Load persisted customer churn model from disk.

    Parameters
    ----------
    model_path : Path or str, optional
        Path to joblib file.

    Returns
    -------
    Dict[str, Any]
        Model artifact dictionary containing pipeline and metadata.
    """
    if model_path is None:
        project_root = Path(__file__).resolve().parent.parent
        model_path = project_root / "models" / "best_customer_churn_model.joblib"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Churn model artifact not found at: {model_path.resolve()}")

    return joblib.load(model_path)
