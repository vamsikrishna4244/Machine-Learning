"""
Model Training and Persistence Module for House Price Prediction.

Trains baseline Linear Regression, Decision Tree Regressor, and Random Forest Regressor
within unified Scikit-learn Pipeline abstractions to eliminate data leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from .data_preprocessing import build_preprocessing_pipeline


def build_model_pipeline(regressor: BaseEstimator) -> Pipeline:
    """Combine data preprocessing ColumnTransformer and regression estimator into a Pipeline.

    Parameters
    ----------
    regressor : BaseEstimator
        Scikit-learn regressor (e.g. LinearRegression, DecisionTreeRegressor).

    Returns
    -------
    Pipeline
        Full composite Scikit-learn Pipeline.
    """
    preprocessor = build_preprocessing_pipeline()
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])


def train_regression_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Dict[str, Pipeline]:
    """Train baseline Linear Regression, Decision Tree, and Random Forest pipelines.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature records.
    y_train : pd.Series
        Training target values (property prices).
    random_state : int, default=42
        Reproducible random seed for tree estimators.

    Returns
    -------
    Dict[str, Pipeline]
        Dictionary of trained model pipelines keyed by algorithm name.
    """
    models_to_train: Dict[str, BaseEstimator] = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=random_state,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=16,
            min_samples_split=6,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        ),
    }

    trained_pipelines: Dict[str, Pipeline] = {}
    for name, estimator in models_to_train.items():
        pipeline = build_model_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

    return trained_pipelines


def save_best_model(
    pipeline: Pipeline,
    model_name: str,
    metrics: Dict[str, float],
    output_dir: Optional[Path | str] = None,
) -> Path:
    """Persist best trained pipeline model along with metadata via Joblib.

    Parameters
    ----------
    pipeline : Pipeline
        Trained Scikit-learn Pipeline.
    model_name : str
        Algorithm name (e.g., 'Random Forest').
    metrics : Dict[str, float]
        Evaluation metrics dictionary.
    output_dir : Path or str, optional
        Target directory. Defaults to `models/` relative to project root.

    Returns
    -------
    Path
        Absolute filepath to the saved model artifact.
    """
    if output_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root / "models"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / "best_house_price_model.joblib"

    artifact = {
        "model_name": model_name,
        "pipeline": pipeline,
        "metrics": metrics,
    }
    joblib.dump(artifact, filename)
    return filename


def load_saved_model(model_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Load persisted model artifact from disk.

    Parameters
    ----------
    model_path : Path or str, optional
        Path to joblib file. Defaults to `models/best_house_price_model.joblib`.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing pipeline and metadata.
    """
    if model_path is None:
        project_root = Path(__file__).resolve().parent.parent
        model_path = project_root / "models" / "best_house_price_model.joblib"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {model_path.resolve()}")

    return joblib.load(model_path)
