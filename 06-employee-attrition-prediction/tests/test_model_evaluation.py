"""Unit tests for employee attrition model evaluation, interpretability, fairness, and visualization."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.data_loader import generate_attrition_dataset
from src.data_preprocessing import clean_attrition_data, prepare_attrition_train_test
from src.model_training import build_attrition_pipeline
from src.model_evaluation import (
    compute_attrition_metrics,
    evaluate_attrition_models,
    extract_top_feature_importances,
    conduct_demographic_fairness_audit,
    plot_attrition_evaluations,
)


@pytest.fixture
def split_data():
    raw = generate_attrition_dataset(n_samples=250, seed=42)
    cleaned, _ = clean_attrition_data(raw)
    return prepare_attrition_train_test(cleaned, test_size=0.20, random_state=42)


def test_compute_attrition_metrics_with_and_without_prob():
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.85])

    m_no_prob = compute_attrition_metrics(y_true, y_pred)
    assert "accuracy" in m_no_prob
    assert "precision" in m_no_prob
    assert "recall" in m_no_prob
    assert "f1_score" in m_no_prob
    assert "roc_auc" not in m_no_prob

    m_prob = compute_attrition_metrics(y_true, y_pred, y_prob)
    assert "roc_auc" in m_prob
    assert "pr_auc" in m_prob
    assert 0.0 <= m_prob["roc_auc"] <= 1.0


def test_evaluate_attrition_models(split_data):
    X_train, X_test, y_train, y_test = split_data
    pipe_lr = build_attrition_pipeline(LogisticRegression())
    pipe_rf = build_attrition_pipeline(RandomForestClassifier(n_estimators=10, random_state=42))

    pipe_lr.fit(X_train, y_train)
    pipe_rf.fit(X_train, y_train)

    models = {"Logistic Regression": pipe_lr, "Random Forest": pipe_rf}
    summary_df, preds, probs = evaluate_attrition_models(models, X_test, y_test)

    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) == 2
    assert "ROC-AUC" in summary_df.columns
    assert "Logistic Regression" in preds
    assert "Random Forest" in probs


def test_extract_top_feature_importances(split_data):
    X_train, _, y_train, _ = split_data
    pipe_rf = build_attrition_pipeline(RandomForestClassifier(n_estimators=10, random_state=42))
    pipe_rf.fit(X_train, y_train)

    feat_rf = extract_top_feature_importances(pipe_rf, top_n=5)
    assert len(feat_rf) == 5
    assert "Feature" in feat_rf.columns
    assert "Importance" in feat_rf.columns

    pipe_lr = build_attrition_pipeline(LogisticRegression())
    pipe_lr.fit(X_train, y_train)
    feat_lr = extract_top_feature_importances(pipe_lr, top_n=5)
    assert len(feat_lr) == 5
    assert "Sign" in feat_lr.columns

    # Test edge case with unitted or empty pipeline
    empty_df = extract_top_feature_importances(None)
    assert empty_df.empty


def test_conduct_demographic_fairness_audit(split_data):
    X_train, X_test, y_train, _ = split_data
    pipe = build_attrition_pipeline(LogisticRegression())
    pipe.fit(X_train, y_train)

    audit_df = conduct_demographic_fairness_audit(pipe, X_test, sensitive_col="Gender")
    assert not audit_df.empty
    assert "Demographic_Group" in audit_df.columns
    assert "Disparate_Impact_Ratio" in audit_df.columns
    assert "Passes_80_Pct_Rule" in audit_df.columns

    # Missing column returns empty DataFrame
    empty_audit = conduct_demographic_fairness_audit(pipe, X_test, sensitive_col="NonExistent")
    assert empty_audit.empty


def test_plot_attrition_evaluations(split_data, tmp_path: Path):
    X_train, X_test, y_train, y_test = split_data
    pipe = build_attrition_pipeline(RandomForestClassifier(n_estimators=10, random_state=42))
    pipe.fit(X_train, y_train)

    summary_df, preds, probs = evaluate_attrition_models({"RF": pipe}, X_test, y_test)
    plots = plot_attrition_evaluations(
        y_test=y_test,
        predictions=preds,
        probabilities=probs,
        summary_df=summary_df,
        best_pipeline=pipe,
        output_dir=tmp_path,
    )

    assert "model_comparison" in plots
    assert "confusion_matrices" in plots
    assert "roc_pr_curves" in plots
    assert "feature_importance" in plots
    for p in plots.values():
        assert p.exists()
