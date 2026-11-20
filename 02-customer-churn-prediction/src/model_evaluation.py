"""
Model Evaluation, Metric Benchmarking, and Confusion Matrix Diagnostics for Customer Churn.

Computes Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and visualizes
confusion matrices, ROC/PR curves, and feature importances.
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
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline


def compute_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_prob: Optional[pd.Series | np.ndarray] = None,
) -> Dict[str, float]:
    """Calculate comprehensive classification metrics including balanced trade-offs.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0 or 1).
    y_pred : array-like
        Hard predicted binary labels (0 or 1).
    y_prob : array-like, optional
        Predicted positive class probabilities.

    Returns
    -------
    Dict[str, float]
        Dictionary with accuracy, precision, recall, f1, and optional roc_auc and pr_auc.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))

    metrics: Dict[str, float] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
    }

    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob, dtype=float)
        metrics["roc_auc"] = round(float(roc_auc_score(y_t, y_prob_arr)), 4)
        metrics["pr_auc"] = round(float(average_precision_score(y_t, y_prob_arr)), 4)

    return metrics


def evaluate_churn_models(
    models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Evaluate candidate classifiers on held-out test data.

    Parameters
    ----------
    models : Dict[str, Pipeline]
        Candidate pipelines.
    X_test : pd.DataFrame
        Test features.
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
        else:
            y_prob = y_pred.astype(float)
        probabilities[name] = y_prob

        metrics = compute_classification_metrics(y_test, y_pred, y_prob)
        records.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-Score": metrics["f1_score"],
            "ROC-AUC": metrics.get("roc_auc", 0.0),
            "PR-AUC": metrics.get("pr_auc", 0.0),
        })

    summary_df = pd.DataFrame(records).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
    return summary_df, predictions, probabilities


def compare_churn_models(summary_df: pd.DataFrame) -> str:
    """Select the best classifier based primarily on ROC-AUC and F1-Score."""
    best_row = summary_df.iloc[0]
    return str(best_row["Model"])


def plot_churn_evaluations(
    y_test: pd.Series,
    predictions: Dict[str, np.ndarray],
    probabilities: Dict[str, np.ndarray],
    summary_df: pd.DataFrame,
    best_pipeline: Pipeline,
    output_dir: Optional[Path | str] = None,
) -> Dict[str, Path]:
    """Generate and save publication-grade classification evaluation charts.

    Parameters
    ----------
    y_test : pd.Series
        Ground truth test labels.
    predictions : Dict[str, np.ndarray]
        Binary predictions for each model.
    probabilities : Dict[str, np.ndarray]
        Positive class probabilities.
    summary_df : pd.DataFrame
        Model comparison table.
    best_pipeline : Pipeline
        Best performing pipeline.
    output_dir : Path or str, optional
        Target directory for saved figures. Defaults to `visualizations/`.

    Returns
    -------
    Dict[str, Path]
        Paths to generated visualization images.
    """
    if output_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root / "visualizations"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_plots: Dict[str, Path] = {}
    y_t = y_test.to_numpy()

    # 1. Confusion Matrix Comparison (Side-by-Side)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    model_names = list(predictions.keys())
    for idx, name in enumerate(model_names):
        ax = axes[idx]
        cm = confusion_matrix(y_t, predictions[name])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Retained", "Churned"],
            yticklabels=["Retained", "Churned"],
        )
        ax.set_title(f"{name}\nConfusion Matrix", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label" if idx == 0 else "")

    p1 = output_dir / "confusion_matrices.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=300)
    plt.close(fig)
    saved_plots["confusion_matrices"] = p1

    # 2. ROC Curves Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2980b9", "#27ae60", "#8e44ad"]
    for idx, (name, y_prob) in enumerate(probabilities.items()):
        fpr, tpr, _ = roc_curve(y_t, y_prob)
        auc_val = roc_auc_score(y_t, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=colors[idx % len(colors)], linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.500)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.legend(frameon=True, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)

    p2 = output_dir / "roc_curves.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=300)
    plt.close(fig)
    saved_plots["roc_curves"] = p2

    # 3. Precision-Recall Curves (Critical for Imbalanced Data)
    fig, ax = plt.subplots(figsize=(8, 6))
    baseline_pr = float(np.mean(y_t))
    ax.axhline(baseline_pr, color="red", linestyle="--", alpha=0.6, label=f"No Skill (Prevalence = {baseline_pr:.1%})")

    for idx, (name, y_prob) in enumerate(probabilities.items()):
        prec_arr, rec_arr, _ = precision_recall_curve(y_t, y_prob)
        pr_auc_val = average_precision_score(y_t, y_prob)
        ax.plot(rec_arr, prec_arr, label=f"{name} (PR-AUC = {pr_auc_val:.3f})", color=colors[idx % len(colors)], linewidth=2)

    ax.set_title("Precision-Recall Curves (Class Imbalance Robust)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall (Fraction of Churners Detected)")
    ax.set_ylabel("Precision (Accuracy of Churn Warnings)")
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

    p3 = output_dir / "precision_recall_curves.png"
    plt.tight_layout()
    plt.savefig(p3, dpi=300)
    plt.close(fig)
    saved_plots["precision_recall_curves"] = p3

    # 4. Feature Importance / Logistic Odds Ratios
    best_name = compare_churn_models(summary_df)
    clf = best_pipeline.named_steps.get("classifier")
    prep = best_pipeline.named_steps.get("preprocessor")

    if prep is not None and clf is not None:
        feat_names = prep.get_feature_names_out()
        clean_names = [f.replace("num__", "").replace("cat__", "") for f in feat_names]

        if hasattr(clf, "feature_importances_"):
            weights = clf.feature_importances_
            xlabel = "Relative Gini Feature Importance"
            title = f"Top 12 Predictive Features ({best_name})"
        elif hasattr(clf, "coef_"):
            weights = np.abs(clf.coef_[0])
            xlabel = "Standardized Coefficient Magnitude (|β|)"
            title = f"Top 12 Influential Churn Drivers ({best_name})"
        else:
            weights = None

        if weights is not None:
            feat_df = pd.DataFrame({"Feature": clean_names, "Weight": weights}).sort_values(by="Weight", ascending=True)
            fig, ax = plt.subplots(figsize=(9, 7))
            ax.barh(feat_df["Feature"].tail(12), feat_df["Weight"].tail(12), color="#2c3e50")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel(xlabel)
            ax.grid(axis="x", linestyle=":", alpha=0.5)

            p4 = output_dir / "feature_importance.png"
            plt.tight_layout()
            plt.savefig(p4, dpi=300)
            plt.close(fig)
            saved_plots["feature_importance"] = p4

    return saved_plots
