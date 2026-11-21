"""
Unit Tests for Model Training, Pipeline Building, and Artifact Serialization.
"""

from pathlib import Path
import pytest
from sklearn.pipeline import Pipeline
from src.data_loader import generate_spam_dataset
from src.text_preprocessing import preprocess_corpus, split_email_dataset
from src.model_training import (
    build_spam_pipeline,
    train_spam_classifiers,
    save_best_spam_pipeline,
    load_saved_spam_pipeline,
)


@pytest.fixture
def minimal_train_data():
    """Minimal text fixture for pipeline training."""
    raw = generate_spam_dataset(n_samples=120, seed=42)
    cleaned, _ = preprocess_corpus(raw)
    return split_email_dataset(cleaned, test_size=0.25, random_state=42)


def test_build_spam_pipeline_combinations():
    """Verify building various valid pipeline configurations."""
    pipe1 = build_spam_pipeline("bow", "multinomial_nb")
    assert isinstance(pipe1, Pipeline)
    assert "vectorizer" in pipe1.named_steps
    assert "classifier" in pipe1.named_steps

    pipe2 = build_spam_pipeline("tfidf", "logistic_regression")
    assert isinstance(pipe2, Pipeline)

    pipe3 = build_spam_pipeline("tfidf", "svm")
    assert isinstance(pipe3, Pipeline)


def test_build_spam_pipeline_invalid_inputs():
    """Verify ValueError on unsupported vectorizer or classifier strings."""
    with pytest.raises(ValueError, match="Unsupported vectorizer_type"):
        build_spam_pipeline(vectorizer_type="word2vec")

    with pytest.raises(ValueError, match="Unsupported classifier_type"):
        build_spam_pipeline(classifier_type="transformer")


def test_train_spam_classifiers(minimal_train_data):
    """Verify training all 6 candidate model pipelines."""
    X_train, _, y_train, _ = minimal_train_data
    models = train_spam_classifiers(X_train, y_train, random_state=42)
    assert len(models) == 6
    for pipe in models.values():
        assert hasattr(pipe, "predict")


def test_save_and_load_spam_pipeline(minimal_train_data, tmp_path: Path):
    """Verify pipeline persistence and deserialization."""
    X_train, _, y_train, _ = minimal_train_data
    pipe = build_spam_pipeline("tfidf", "logistic_regression")
    pipe.fit(X_train, y_train)

    metrics = {"accuracy": 0.98, "f1_score": 0.95}
    saved_path = save_best_spam_pipeline(
        pipeline=pipe,
        model_name="TF-IDF + Logistic Regression",
        metrics=metrics,
        output_dir=tmp_path,
    )
    assert saved_path.exists()

    loaded = load_saved_spam_pipeline(saved_path)
    assert loaded["model_name"] == "TF-IDF + Logistic Regression"
    assert loaded["metrics"]["f1_score"] == 0.95
    assert isinstance(loaded["pipeline"], Pipeline)


def test_load_saved_spam_pipeline_missing(tmp_path: Path):
    """Verify FileNotFoundError on missing file."""
    fake_path = tmp_path / "non_existent.joblib"
    with pytest.raises(FileNotFoundError):
        load_saved_spam_pipeline(fake_path)
