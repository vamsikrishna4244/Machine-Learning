"""
Spam Classification Model Training, Pipeline Construction, and Model Serialization.

Constructs end-to-end Scikit-learn pipelines pairing Bag-of-Words and TF-IDF vectorizers
with Multinomial Naive Bayes, Logistic Regression, and Support Vector Machines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .feature_extraction import build_bow_vectorizer, build_tfidf_vectorizer


def build_spam_pipeline(
    vectorizer_type: str = "tfidf",
    classifier_type: str = "logistic_regression",
    random_state: int = 42,
) -> Pipeline:
    """Build a complete end-to-end text classification pipeline.

    Parameters
    ----------
    vectorizer_type : {'tfidf', 'bow'}, default='tfidf'
        Text feature extraction method.
    classifier_type : {'multinomial_nb', 'logistic_regression', 'svm'}, default='logistic_regression'
        Machine learning classification algorithm.
    random_state : int, default=42
        Reproducible random seed.

    Returns
    -------
    Pipeline
        Composed Scikit-learn Pipeline (vectorizer -> classifier).
    """
    if vectorizer_type.lower() == "bow":
        vectorizer = build_bow_vectorizer()
    elif vectorizer_type.lower() == "tfidf":
        vectorizer = build_tfidf_vectorizer()
    else:
        raise ValueError(f"Unsupported vectorizer_type '{vectorizer_type}'. Choose 'bow' or 'tfidf'.")

    c_type = classifier_type.lower()
    if c_type in ["multinomial_nb", "naive_bayes", "nb"]:
        classifier: BaseEstimator = MultinomialNB(alpha=1.0)
    elif c_type in ["logistic_regression", "lr"]:
        classifier = LogisticRegression(
            C=2.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )
    elif c_type in ["svm", "svc", "support_vector_machine"]:
        classifier = SVC(
            C=1.0,
            kernel="linear",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        )
    else:
        raise ValueError(
            f"Unsupported classifier_type '{classifier_type}'. "
            "Choose 'multinomial_nb', 'logistic_regression', or 'svm'."
        )

    return Pipeline(steps=[("vectorizer", vectorizer), ("classifier", classifier)])


def train_spam_classifiers(
    X_train: pd.Series,
    y_train: pd.Series,
    random_state: int = 42,
) -> Dict[str, Pipeline]:
    """Train all 6 combinations of vectorizers and classifiers on training data.

    Combinations:
    1. BoW + Multinomial Naive Bayes
    2. BoW + Logistic Regression
    3. BoW + Support Vector Machine
    4. TF-IDF + Multinomial Naive Bayes
    5. TF-IDF + Logistic Regression
    6. TF-IDF + Support Vector Machine

    Parameters
    ----------
    X_train : pd.Series
        Cleaned text training messages.
    y_train : pd.Series
        Binary target labels (0 = Ham, 1 = Spam).
    random_state : int, default=42
        Reproducible random seed.

    Returns
    -------
    Dict[str, Pipeline]
        Dictionary of trained model pipelines.
    """
    configurations = [
        ("BoW + Naive Bayes", "bow", "multinomial_nb"),
        ("BoW + Logistic Regression", "bow", "logistic_regression"),
        ("BoW + Support Vector Machine", "bow", "svm"),
        ("TF-IDF + Naive Bayes", "tfidf", "multinomial_nb"),
        ("TF-IDF + Logistic Regression", "tfidf", "logistic_regression"),
        ("TF-IDF + Support Vector Machine", "tfidf", "svm"),
    ]

    trained_pipelines: Dict[str, Pipeline] = {}
    for name, vec_type, clf_type in configurations:
        pipe = build_spam_pipeline(vec_type, clf_type, random_state=random_state)
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

    return trained_pipelines


def save_best_spam_pipeline(
    pipeline: Pipeline,
    model_name: str,
    metrics: Dict[str, float],
    output_dir: Optional[Path | str] = None,
) -> Path:
    """Serialize the top-performing pipeline along with evaluation metadata.

    Parameters
    ----------
    pipeline : Pipeline
        Trained Scikit-learn pipeline.
    model_name : str
        Configuration name.
    metrics : Dict[str, float]
        Evaluation metrics dictionary.
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
    target_path = output_dir / "best_spam_classifier.joblib"

    artifact = {
        "model_name": model_name,
        "pipeline": pipeline,
        "metrics": metrics,
    }
    joblib.dump(artifact, target_path)
    return target_path


def load_saved_spam_pipeline(model_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Load serialized email spam classification pipeline from disk.

    Parameters
    ----------
    model_path : Path or str, optional
        Path to artifact. Defaults to `models/best_spam_classifier.joblib`.

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'model_name', 'pipeline', and 'metrics'.
    """
    if model_path is None:
        project_root = Path(__file__).resolve().parent.parent
        model_path = project_root / "models" / "best_spam_classifier.joblib"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Spam classification model not found at: {model_path.resolve()}")

    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError("Loaded artifact is corrupted or does not contain a valid pipeline dictionary.")

    return artifact
