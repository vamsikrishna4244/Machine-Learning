"""
Evaluation, Benchmarking, Vectorizer Comparison, and Visualization for Spam Classification.

Computes Precision, Recall, F1-Score, Accuracy, and Confusion Matrices across all
Bag-of-Words and TF-IDF model pipelines, and renders diagnostic charts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


def compute_spam_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_prob: Optional[pd.Series | np.ndarray] = None,
) -> Dict[str, float]:
    """Calculate core classification performance metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0 = Ham, 1 = Spam).
    y_pred : array-like
        Predicted binary labels.
    y_prob : array-like, optional
        Predicted spam probabilities.

    Returns
    -------
    Dict[str, float]
        Dictionary with accuracy, precision, recall, f1_score, and optional roc_auc.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
    }

    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob, dtype=float)
        metrics["roc_auc"] = round(float(roc_auc_score(y_t, y_prob_arr)), 4)

    return metrics


def evaluate_spam_models(
    models: Dict[str, Pipeline],
    X_test: pd.Series,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Benchmark all trained spam classification pipelines on held-out test data.

    Parameters
    ----------
    models : Dict[str, Pipeline]
        Candidate pipelines (BoW & TF-IDF variants).
    X_test : pd.Series
        Test email text messages.
    y_test : pd.Series
        Test ground truth labels.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]
        Comparison summary dataframe, dictionary of predictions, and dictionary of probabilities.
    """
    records: List[Dict[str, Any]] = []
    predictions: Dict[str, np.ndarray] = {}
    probabilities: Dict[str, np.ndarray] = {}

    for name, pipeline in models.items():
        y_pred = pipeline.predict(X_test)
        predictions[name] = y_pred

        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            # Normalize decision function via sigmoid
            df_vals = pipeline.decision_function(X_test)
            y_prob = 1.0 / (1.0 + np.exp(-df_vals))
        else:
            y_prob = y_pred.astype(float)
        probabilities[name] = y_prob

        metrics = compute_spam_metrics(y_test, y_pred, y_prob)

        # Parse Vectorizer and Algorithm tags
        parts = name.split(" + ")
        vec_tech = parts[0]
        algo = parts[1] if len(parts) > 1 else name

        records.append({
            "Configuration": name,
            "Vectorizer": vec_tech,
            "Algorithm": algo,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-Score": metrics["f1_score"],
            "ROC-AUC": metrics.get("roc_auc", 0.0),
        })

    summary_df = pd.DataFrame(records).sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
    return summary_df, predictions, probabilities


def compare_vectorization_methods(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metric performance by vectorization technique (BoW vs TF-IDF)."""
    return (
        summary_df.groupby("Vectorizer")[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]]
        .mean()
        .round(4)
        .reset_index()
    )


def plot_spam_evaluations(
    y_test: pd.Series,
    predictions: Dict[str, np.ndarray],
    summary_df: pd.DataFrame,
    best_pipeline: Pipeline,
    output_dir: Optional[Path | str] = None,
) -> Dict[str, Path]:
    """Generate diagnostic visual charts for email spam classification.

    Generates:
    1. BoW vs TF-IDF Vectorizer Comparison Bar Chart
    2. Side-by-Side Confusion Matrix Grid
    3. Top Indicative Spam vs Ham Features

    Parameters
    ----------
    y_test : pd.Series
        Test ground truth.
    predictions : Dict[str, np.ndarray]
        Model prediction dictionary.
    summary_df : pd.DataFrame
        Performance benchmark table.
    best_pipeline : Pipeline
        Top-scoring pipeline.
    output_dir : Path or str, optional
        Output folder. Defaults to `visualizations/`.

    Returns
    -------
    Dict[str, Path]
        Paths of generated visualization files.
    """
    if output_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root / "visualizations"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_plots: Dict[str, Path] = {}
    y_t = np.asarray(y_test, dtype=int)

    # 1. Vectorizer & Algorithm F1-Score Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = summary_df.sort_values(by="F1-Score", ascending=True)
    colors = ["#3498db" if "TF-IDF" in c else "#e67e22" for c in plot_df["Configuration"]]
    bars = ax.barh(plot_df["Configuration"], plot_df["F1-Score"], color=colors, height=0.6)
    ax.set_xlim(0, 1.05)
    ax.set_title("Model & Vectorizer Performance Benchmark (F1-Score)", fontsize=12, fontweight="bold")
    ax.set_xlabel("F1-Score (Harmonic Mean of Precision & Recall)")
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=9)

    p1 = output_dir / "vectorizer_model_comparison.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=300)
    plt.close(fig)
    saved_plots["vectorizer_comparison"] = p1

    # 2. Confusion Matrices for Top 3 Configurations
    top_3_configs = summary_df.head(3)["Configuration"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for idx, cfg_name in enumerate(top_3_configs):
        ax = axes[idx]
        cm = confusion_matrix(y_t, predictions[cfg_name])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Purples",
            cbar=False,
            ax=ax,
            xticklabels=["Ham", "Spam"],
            yticklabels=["Ham", "Spam"],
        )
        ax.set_title(f"{cfg_name}\nConfusion Matrix", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class" if idx == 0 else "")

    p2 = output_dir / "confusion_matrices.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=300)
    plt.close(fig)
    saved_plots["confusion_matrices"] = p2

    # 3. Top Discriminating Feature Importance / Log-Odds
    clf = best_pipeline.named_steps.get("classifier")
    vec = best_pipeline.named_steps.get("vectorizer")
    if clf is not None and vec is not None:
        vocab = np.array(vec.get_feature_names_out())
        if hasattr(clf, "coef_"):
            coefs = clf.coef_[0]
            metric_label = "Log-Odds Coefficient (β)"
        elif hasattr(clf, "feature_log_prob_"):
            coefs = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
            metric_label = "Log-Likelihood Ratio (Spam - Ham)"
        else:
            coefs = None

        if coefs is not None and len(vocab) > 0:
            top_spam_idx = coefs.argsort()[::-1][:12]
            top_ham_idx = coefs.argsort()[:12]

            fig, (ax_spam, ax_ham) = plt.subplots(1, 2, figsize=(14, 5))

            # Top Spam tokens (positive coefficients)
            ax_spam.barh(vocab[top_spam_idx][::-1], coefs[top_spam_idx][::-1], color="#e74c3c")
            ax_spam.set_title("Top 12 Indicative SPAM Tokens", fontsize=11, fontweight="bold")
            ax_spam.set_xlabel(metric_label)
            ax_spam.grid(axis="x", linestyle=":", alpha=0.5)

            # Top Ham tokens (negative coefficients)
            ax_ham.barh(vocab[top_ham_idx], np.abs(coefs[top_ham_idx]), color="#27ae60")
            ax_ham.set_title("Top 12 Indicative HAM Tokens", fontsize=11, fontweight="bold")
            ax_ham.set_xlabel(f"Absolute {metric_label}")
            ax_ham.grid(axis="x", linestyle=":", alpha=0.5)

            p3 = output_dir / "top_spam_keywords.png"
            plt.tight_layout()
            plt.savefig(p3, dpi=300)
            plt.close(fig)
            saved_plots["top_spam_keywords"] = p3

    return saved_plots
