"""Unit tests for employee attrition model training, tuning, and artifact serialization."""

import pytest
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.data_loader import generate_attrition_dataset
from src.data_preprocessing import clean_attrition_data, prepare_attrition_train_test
from src.model_training import (
    build_attrition_pipeline,
    train_baseline_models,
    tune_best_model_grid_search,
    save_attrition_pipeline,
    load_attrition_pipeline,
)


@pytest.fixture
def split_data():
    raw = generate_attrition_dataset(n_samples=250, seed=42)
    cleaned, _ = clean_attrition_data(raw)
    return prepare_attrition_train_test(cleaned, test_size=0.20, random_state=42)


def test_build_attrition_pipeline():
    clf = LogisticRegression()
    pipeline = build_attrition_pipeline(clf)
    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_train_baseline_models(split_data):
    X_train, _, y_train, _ = split_data
    trained_pipes, cv_df = train_baseline_models(X_train, y_train, cv_folds=2, random_state=42)

    assert isinstance(trained_pipes, dict)
    assert "Logistic Regression" in trained_pipes
    assert "Decision Tree" in trained_pipes
    assert "Random Forest" in trained_pipes
    assert len(cv_df) == 3
    assert "CV_Mean_ROC_AUC" in cv_df.columns
    assert cv_df.iloc[0]["CV_Mean_ROC_AUC"] >= 0.50


def test_tune_best_model_grid_search(split_data):
    X_train, _, y_train, _ = split_data
    best_pipe, best_params, best_score = tune_best_model_grid_search(
        X_train, y_train, cv_folds=2, random_state=42
    )

    assert isinstance(best_pipe, Pipeline)
    assert isinstance(best_params, dict)
    assert "classifier__n_estimators" in best_params
    assert best_score >= 0.50


def test_save_and_load_attrition_pipeline(split_data, tmp_path: Path):
    X_train, _, y_train, _ = split_data
    clf = LogisticRegression()
    pipe = build_attrition_pipeline(clf)
    pipe.fit(X_train, y_train)

    metrics = {"accuracy": 0.85, "roc_auc": 0.88}
    saved_path = save_attrition_pipeline(
        pipeline=pipe,
        model_name="Test Logistic Regression",
        metrics=metrics,
        best_params={"C": 1.0},
        output_dir=tmp_path,
    )

    assert saved_path.exists()
    loaded_artifact = load_attrition_pipeline(saved_path)
    assert loaded_artifact["model_name"] == "Test Logistic Regression"
    assert "pipeline" in loaded_artifact
    assert loaded_artifact["metrics"]["roc_auc"] == 0.88


def test_load_attrition_pipeline_invalid(tmp_path: Path):
    non_existent = tmp_path / "missing.joblib"
    with pytest.raises(FileNotFoundError, match="Attrition model not found"):
        load_attrition_pipeline(non_existent)

    # Corrupt artifact
    bad_file = tmp_path / "bad.joblib"
    import joblib
    joblib.dump({"invalid": 123}, bad_file)
    with pytest.raises(ValueError, match="does not contain a valid pipeline"):
        load_attrition_pipeline(bad_file)
