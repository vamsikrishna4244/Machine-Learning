"""
Unit tests for model training and persistence module.
"""

from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from src.data_loader import generate_housing_dataset
from src.data_preprocessing import clean_housing_data, prepare_train_test_data
from src.model_training import (
    build_model_pipeline,
    train_regression_models,
    save_best_model,
    load_saved_model,
)


@pytest.fixture
def split_data():
    raw = generate_housing_dataset(n_samples=120, seed=99)
    cleaned, _ = clean_housing_data(raw)
    return prepare_train_test_data(cleaned, test_size=0.25, random_state=42)


def test_build_model_pipeline():
    pipe = build_model_pipeline(LinearRegression())
    assert isinstance(pipe, Pipeline)
    assert "preprocessor" in pipe.named_steps
    assert "regressor" in pipe.named_steps


def test_train_regression_models(split_data):
    X_train, X_test, y_train, y_test = split_data
    models = train_regression_models(X_train, y_train, random_state=42)
    assert "Linear Regression" in models
    assert "Decision Tree" in models
    assert "Random Forest" in models

    for name, pipe in models.items():
        preds = pipe.predict(X_test)
        assert len(preds) == len(X_test)


def test_save_and_load_model(tmp_path: Path, split_data):
    X_train, X_test, y_train, y_test = split_data
    models = train_regression_models(X_train, y_train, random_state=42)
    pipeline = models["Linear Regression"]
    metrics = {"r2_score": 0.95, "rmse": 25000.0}

    saved_file = save_best_model(pipeline, "Linear Regression", metrics, output_dir=tmp_path)
    assert saved_file.exists()

    loaded = load_saved_model(saved_file)
    assert loaded["model_name"] == "Linear Regression"
    assert "pipeline" in loaded
    assert loaded["metrics"]["r2_score"] == 0.95


def test_load_saved_model_missing_file(tmp_path: Path):
    missing_path = tmp_path / "not_there.joblib"
    with pytest.raises(FileNotFoundError):
        load_saved_model(missing_path)
