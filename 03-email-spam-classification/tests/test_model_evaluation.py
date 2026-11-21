"""
Unit Tests for Model Evaluation and Diagnostics.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from src.data_loader import generate_spam_dataset
from src.text_preprocessing import preprocess_corpus, split_email_dataset
from src.model_training import train_spam_classifiers
from src.model_evaluation import (
    compute_spam_metrics,
    evaluate_spam_models,
    compare_vectorization_methods,
    plot_spam_evaluations,
)


def test_compute_spam_metrics():
    """Verify classification metric calculations."""
    y_true = np.array([0, 1, 0, 1, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.7, 0.85, 0.95])

    metrics = compute_spam_metrics(y_true, y_pred, y_prob)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert 0.0 <= metrics["f1_score"] <= 1.0


def test_evaluate_and_compare_spam_models():
    """Verify model benchmarking and vectorizer comparison."""
    raw = generate_spam_dataset(n_samples=120, seed=42)
    cleaned, _ = preprocess_corpus(raw)
    X_train, X_test, y_train, y_test = split_email_dataset(cleaned, test_size=0.25, random_state=42)

    models = train_spam_classifiers(X_train, y_train, random_state=42)
    summary_df, predictions, probabilities = evaluate_spam_models(models, X_test, y_test)

    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) == 6
    assert "F1-Score" in summary_df.columns
    assert len(predictions) == 6
    assert len(probabilities) == 6

    vec_comp = compare_vectorization_methods(summary_df)
    assert len(vec_comp) == 2  # BoW and TF-IDF
    assert "Vectorizer" in vec_comp.columns


def test_plot_spam_evaluations(tmp_path: Path):
    """Verify diagnostic plot saving."""
    raw = generate_spam_dataset(n_samples=120, seed=42)
    cleaned, _ = preprocess_corpus(raw)
    X_train, X_test, y_train, y_test = split_email_dataset(cleaned, test_size=0.25, random_state=42)

    models = train_spam_classifiers(X_train, y_train, random_state=42)
    summary_df, predictions, _ = evaluate_spam_models(models, X_test, y_test)
    best_pipe = models[summary_df.iloc[0]["Configuration"]]

    plots = plot_spam_evaluations(
        y_test=y_test,
        predictions=predictions,
        summary_df=summary_df,
        best_pipeline=best_pipe,
        output_dir=tmp_path,
    )

    assert "vectorizer_comparison" in plots
    assert "confusion_matrices" in plots
    for p in plots.values():
        assert p.exists()
