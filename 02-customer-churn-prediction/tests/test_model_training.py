"""
Unit Tests for Model Training, Pipeline Composition, and Serialization.
"""

from pathlib import Path
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.data_loader import generate_churn_dataset
from src.data_preprocessing import clean_churn_data, prepare_churn_train_test_data
from src.model_training import (
    build_churn_model_pipeline,
    train_churn_classifiers,
    save_best_churn_model,
    load_saved_churn_model,
)


@pytest.fixture
def churn_train_data():
    """Generate minimal train-test data fixture."""
    raw = generate_churn_dataset(n_samples=250, seed=42)
    cleaned, _ = clean_churn_data(raw)
    return prepare_churn_train_test_data(cleaned, test_size=0.2, random_state=42)


def test_build_churn_model_pipeline():
    """Verify composition of ColumnTransformer and classifier into Pipeline."""
    clf = LogisticRegression()
    pipe = build_churn_model_pipeline(clf)
    assert isinstance(pipe, Pipeline)
    assert "preprocessor" in pipe.named_steps
    assert "classifier" in pipe.named_steps


def test_train_churn_classifiers(churn_train_data):
    """Verify training of all 3 candidate classification models."""
    X_train, _, y_train, _ = churn_train_data
    models = train_churn_classifiers(X_train, y_train, random_state=42)
    assert "Logistic Regression" in models
    assert "Decision Tree" in models
    assert "Random Forest" in models

    for name, pipe in models.items():
        assert hasattr(pipe, "predict")
        assert hasattr(pipe, "predict_proba")


def test_save_and_load_churn_model(churn_train_data, tmp_path: Path):
    """Verify serialization and deserialization of trained pipeline and metadata."""
    X_train, _, y_train, _ = churn_train_data
    clf = LogisticRegression()
    pipe = build_churn_model_pipeline(clf)
    pipe.fit(X_train, y_train)

    metrics = {"accuracy": 0.82, "roc_auc": 0.84, "f1_score": 0.55}
    saved_path = save_best_churn_model(
        pipeline=pipe,
        model_name="Logistic Regression",
        metrics=metrics,
        output_dir=tmp_path,
    )
    assert saved_path.exists()

    loaded = load_saved_churn_model(saved_path)
    assert loaded["model_name"] == "Logistic Regression"
    assert loaded["metrics"]["roc_auc"] == 0.84
    assert isinstance(loaded["pipeline"], Pipeline)


def test_load_saved_churn_model_missing_file_raises_error(tmp_path: Path):
    """Verify FileNotFoundError when loading non-existent model file."""
    fake_path = tmp_path / "does_not_exist.joblib"
    with pytest.raises(FileNotFoundError):
        load_saved_churn_model(fake_path)
