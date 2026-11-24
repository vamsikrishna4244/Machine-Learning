"""
Model Training, Stratified Cross-Validation, GridSearchCV Tuning, and Artifact Persistence.

Composes ColumnTransformer and candidate classifiers into end-to-end Pipelines,
executes 5-fold cross-validation, tunes hyperparameters via GridSearchCV, and serializes artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .data_preprocessing import build_attrition_preprocessor


def build_attrition_pipeline(classifier: BaseEstimator) -> Pipeline:
    """Compose ColumnTransformer preprocessor and classifier into an end-to-end Pipeline.

    Parameters
    ----------
    classifier : BaseEstimator
        Scikit-learn classification estimator.

    Returns
    -------
    Pipeline
        End-to-end classification pipeline.
    """
    preprocessor = build_attrition_preprocessor()
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def train_baseline_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Tuple[Dict[str, Pipeline], pd.DataFrame]:
    """Train candidate classification models and evaluate 5-fold cross-validation ROC-AUC.

    Models:
    1. Logistic Regression (balanced)
    2. Decision Tree Classifier (balanced)
    3. Random Forest Classifier (balanced)

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target labels.
    cv_folds : int, default=5
        Number of stratified cross-validation folds.
    random_state : int, default=42
        Reproducible random seed.

    Returns
    -------
    Tuple[Dict[str, Pipeline], pd.DataFrame]
        Fitted candidate pipelines and cross-validation score benchmark table.
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
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    trained_pipes: Dict[str, Pipeline] = {}
    cv_records = []

    for name, estimator in candidates.items():
        pipe = build_attrition_pipeline(estimator)
        # Compute cross-validation scores
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        trained_pipes[name] = pipe

        cv_records.append({
            "Model": name,
            "CV_Mean_ROC_AUC": round(float(scores.mean()), 4),
            "CV_Std_ROC_AUC": round(float(scores.std()), 4),
        })

    cv_df = pd.DataFrame(cv_records).sort_values(by="CV_Mean_ROC_AUC", ascending=False).reset_index(drop=True)
    return trained_pipes, cv_df


def tune_best_model_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, Any], float]:
    """Execute hyperparameter optimization on Random Forest via GridSearchCV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    cv_folds : int, default=5
        Number of cross-validation splits.
    random_state : int, default=42
        Reproducible random seed.

    Returns
    -------
    Tuple[Pipeline, Dict[str, Any], float]
        Best tuned pipeline, dictionary of optimal hyperparameters, and best CV ROC-AUC score.
    """
    base_clf = RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=1)
    pipe = build_attrition_pipeline(base_clf)

    param_grid = {
        "classifier__n_estimators": [80, 120],
        "classifier__max_depth": [6, 10],
        "classifier__min_samples_leaf": [2, 5],
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = float(grid_search.best_score_)

    return best_pipeline, best_params, round(best_score, 4)


def save_attrition_pipeline(
    pipeline: Pipeline,
    model_name: str,
    metrics: Dict[str, float],
    best_params: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path | str] = None,
) -> Path:
    """Serialize the top-performing attrition pipeline and tuning metadata.

    Parameters
    ----------
    pipeline : Pipeline
        Trained Scikit-learn Pipeline.
    model_name : str
        Algorithm name.
    metrics : Dict[str, float]
        Held-out evaluation metrics.
    best_params : Dict[str, Any], optional
        Optimal hyperparameter settings.
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
    target_path = output_dir / "best_attrition_model.joblib"

    artifact = {
        "model_name": model_name,
        "pipeline": pipeline,
        "metrics": metrics,
        "best_params": best_params or {},
    }
    joblib.dump(artifact, target_path)
    return target_path


def load_attrition_pipeline(model_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Load serialized employee attrition model pipeline from disk.

    Parameters
    ----------
    model_path : Path or str, optional
        Path to artifact. Defaults to `models/best_attrition_model.joblib`.

    Returns
    -------
    Dict[str, Any]
        Loaded artifact dictionary.
    """
    if model_path is None:
        project_root = Path(__file__).resolve().parent.parent
        model_path = project_root / "models" / "best_attrition_model.joblib"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Attrition model not found at: {model_path.resolve()}")

    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError("Loaded artifact does not contain a valid pipeline dictionary.")

    return artifact
