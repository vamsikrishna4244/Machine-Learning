"""
Unit Tests for Model Training, Pipeline Construction, and Artifact Persistence.
"""

from pathlib import Path
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.data_loader import generate_credit_card_data
from src.data_preprocessing import preprocess_fraud_data, prepare_stratified_train_test
from src.model_training import (
    build_fraud_pipeline,
    train_fraud_classifiers,
    save_best_fraud_model,
    load_saved_fraud_model,
)


@pytest.fixture
def minimal_fraud_train_data():
    """Minimal dataset fixture for model testing."""
    raw = generate_credit_card_data(n_samples=500, fraud_rate=0.04, seed=42)
    cleaned, _ = preprocess_fraud_data(raw)
    return prepare_stratified_train_test(cleaned, test_size=0.20, random_state=42)


def test_build_fraud_pipeline():
    """Verify composition of RobustScaler ColumnTransformer and estimator into Pipeline."""
    clf = LogisticRegression()
    pipe = build_fraud_pipeline(clf)
    assert isinstance(pipe, Pipeline)
    assert "preprocessor" in pipe.named_steps
    assert "classifier" in pipe.named_steps


def test_train_fraud_classifiers(minimal_fraud_train_data):
    """Verify training all candidate models across balancing regimes."""
    X_train, _, y_train, _ = minimal_fraud_train_data
    models = train_fraud_classifiers(X_train, y_train, random_state=42)
    assert len(models) == 5
    assert "Baseline Logistic Regression" in models
    assert "Weighted Logistic Regression" in models
    assert "Undersampled Logistic Regression" in models
    assert "Balanced Decision Tree" in models
    assert "Balanced Random Forest" in models

    for pipe in models.values():
        assert hasattr(pipe, "predict")
        assert hasattr(pipe, "predict_proba")


def test_save_and_load_fraud_model(minimal_fraud_train_data, tmp_path: Path):
    """Verify model artifact serialization and deserialization."""
    X_train, _, y_train, _ = minimal_fraud_train_data
    pipe = build_fraud_pipeline(LogisticRegression())
    pipe.fit(X_train, y_train)

    metrics = {"accuracy": 0.99, "pr_auc": 0.88, "f1_score": 0.85}
    saved_path = save_best_fraud_model(
        pipeline=pipe,
        model_name="Weighted Logistic Regression",
        metrics=metrics,
        output_dir=tmp_path,
    )
    assert saved_path.exists()

    loaded = load_saved_fraud_model(saved_path)
    assert loaded["model_name"] == "Weighted Logistic Regression"
    assert loaded["metrics"]["pr_auc"] == 0.88
    assert isinstance(loaded["pipeline"], Pipeline)


def test_load_saved_fraud_model_missing(tmp_path: Path):
    """Verify FileNotFoundError when model path does not exist."""
    fake_path = tmp_path / "missing_model.joblib"
    with pytest.raises(FileNotFoundError):
        load_saved_fraud_model(fake_path)
