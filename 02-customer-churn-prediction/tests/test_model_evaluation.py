"""
Unit Tests for Model Evaluation, Metrics Benchmarking, and Diagnostics.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.data_loader import generate_churn_dataset
from src.data_preprocessing import clean_churn_data, prepare_churn_train_test_data
from src.model_training import train_churn_classifiers
from src.model_evaluation import (
    compute_classification_metrics,
    evaluate_churn_models,
    compare_churn_models,
    plot_churn_evaluations,
)


def test_compute_classification_metrics():
    """Verify metric calculations for balanced binary classification."""
    y_true = np.array([0, 1, 0, 1, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.6, 0.85, 0.95])

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_evaluate_churn_models_and_selection():
    """Verify benchmarking across multiple models and best model selection."""
    raw = generate_churn_dataset(n_samples=250, seed=42)
    cleaned, _ = clean_churn_data(raw)
    X_train, X_test, y_train, y_test = prepare_churn_train_test_data(cleaned, test_size=0.25)

    models = train_churn_classifiers(X_train, y_train, random_state=42)
    summary_df, predictions, probabilities = evaluate_churn_models(models, X_test, y_test)

    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) == 3
    assert "ROC-AUC" in summary_df.columns
    assert "F1-Score" in summary_df.columns
    assert len(predictions) == 3
    assert len(probabilities) == 3

    best_model_name = compare_churn_models(summary_df)
    assert best_model_name in models


def test_plot_churn_evaluations(tmp_path: Path):
    """Verify diagnostic plot generation and file creation."""
    raw = generate_churn_dataset(n_samples=200, seed=42)
    cleaned, _ = clean_churn_data(raw)
    X_train, X_test, y_train, y_test = prepare_churn_train_test_data(cleaned, test_size=0.25)

    models = train_churn_classifiers(X_train, y_train, random_state=42)
    summary_df, predictions, probabilities = evaluate_churn_models(models, X_test, y_test)
    best_name = compare_churn_models(summary_df)
    best_pipe = models[best_name]

    plots = plot_churn_evaluations(
        y_test=y_test,
        predictions=predictions,
        probabilities=probabilities,
        summary_df=summary_df,
        best_pipeline=best_pipe,
        output_dir=tmp_path,
    )

    assert "confusion_matrices" in plots
    assert "roc_curves" in plots
    assert "precision_recall_curves" in plots
    assert "feature_importance" in plots
    for p in plots.values():
        assert p.exists()
