"""
Unit Tests for Imbalanced Metrics, Financial Cost Calculations, and Diagnostics.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from src.data_loader import generate_credit_card_data
from src.data_preprocessing import preprocess_fraud_data, prepare_stratified_train_test
from src.model_training import train_fraud_classifiers
from src.model_evaluation import (
    compute_imbalanced_metrics,
    calculate_business_cost_impact,
    evaluate_fraud_models,
    plot_fraud_evaluations,
)


def test_compute_imbalanced_metrics():
    """Verify precision, recall, f1, and PR-AUC metric calculations."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 0])
    y_prob = np.array([0.05, 0.1, 0.15, 0.6, 0.2, 0.85, 0.9, 0.4])

    metrics = compute_imbalanced_metrics(y_true, y_pred, y_prob)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "pr_auc" in metrics
    assert "roc_auc" in metrics


def test_calculate_business_cost_impact():
    """Verify operational financial loss matrix computation."""
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1])  # 1 FP, 1 FN

    cost = calculate_business_cost_impact(y_true, y_pred, cost_fp=15.0, cost_fn=500.0)
    assert cost["false_positives"] == 1
    assert cost["false_negatives"] == 1
    assert cost["fp_investigation_cost"] == 15.0
    assert cost["fn_unrecovered_fraud_loss"] == 500.0
    assert cost["total_financial_loss"] == 515.0


def test_evaluate_fraud_models_and_plot(tmp_path: Path):
    """Verify benchmarking across pipelines and visualization artifact generation."""
    raw = generate_credit_card_data(n_samples=500, fraud_rate=0.04, seed=42)
    cleaned, _ = preprocess_fraud_data(raw)
    X_train, X_test, y_train, y_test = prepare_stratified_train_test(cleaned, test_size=0.20, random_state=42)

    models = train_fraud_classifiers(X_train, y_train, random_state=42)
    summary_df, predictions, probabilities = evaluate_fraud_models(models, X_test, y_test)

    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) == 5
    assert "PR-AUC" in summary_df.columns
    assert "Total Cost ($)" in summary_df.columns

    best_name = summary_df.iloc[0]["Model"]
    best_pipe = models[best_name]

    plots = plot_fraud_evaluations(
        y_test=y_test,
        predictions=predictions,
        probabilities=probabilities,
        summary_df=summary_df,
        best_pipeline=best_pipe,
        output_dir=tmp_path,
    )

    assert "pr_curves" in plots
    assert "roc_curves" in plots
    assert "confusion_matrices" in plots
    assert "cost_tradeoff" in plots
    for p in plots.values():
        assert p.exists()
