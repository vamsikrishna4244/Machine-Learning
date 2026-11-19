"""
Unit tests for model evaluation and visualization module.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.data_loader import generate_housing_dataset
from src.data_preprocessing import clean_housing_data, prepare_train_test_data
from src.model_training import train_regression_models
from src.model_evaluation import (
    compute_regression_metrics,
    evaluate_models,
    compare_models,
    plot_model_evaluations,
)


def test_compute_regression_metrics():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([105.0, 195.0, 310.0, 390.0])
    m = compute_regression_metrics(y_true, y_pred)
    assert m["mae"] == 7.5
    assert m["r2_score"] > 0.99
    assert m["rmse"] > 0.0
    assert m["mse"] > 0.0
    assert m["mape_pct"] > 0.0


def test_evaluate_and_compare_models():
    raw = generate_housing_dataset(n_samples=100, seed=77)
    cleaned, _ = clean_housing_data(raw)
    X_train, X_test, y_train, y_test = prepare_train_test_data(cleaned, test_size=0.2, random_state=42)

    models = train_regression_models(X_train, y_train, random_state=42)
    summary_df, predictions = evaluate_models(models, X_test, y_test)

    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) == 3
    assert "R² Score" in summary_df.columns
    assert "RMSE ($)" in summary_df.columns

    best = compare_models(summary_df)
    assert best in models
    assert len(predictions[best]) == len(y_test)


def test_plot_model_evaluations(tmp_path: Path):
    raw = generate_housing_dataset(n_samples=80, seed=77)
    cleaned, _ = clean_housing_data(raw)
    X_train, X_test, y_train, y_test = prepare_train_test_data(cleaned, test_size=0.25, random_state=42)

    models = train_regression_models(X_train, y_train, random_state=42)
    summary_df, predictions = evaluate_models(models, X_test, y_test)
    best_name = compare_models(summary_df)

    plots = plot_model_evaluations(
        y_test, predictions, summary_df, models[best_name], output_dir=tmp_path
    )
    assert "actual_vs_predicted" in plots
    assert "residual_analysis" in plots
    assert "model_comparison" in plots
    for path in plots.values():
        assert path.exists()
