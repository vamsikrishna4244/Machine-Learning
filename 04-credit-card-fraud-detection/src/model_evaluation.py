"""
Model Evaluation, Imbalanced Classification Metrics, Cost Analysis, and Diagnostics.

Computes Precision, Recall, F1-Score, PR-AUC, ROC-AUC, Confusion Matrices, and
asymmetric financial loss curves across all candidate fraud detection pipelines.
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


def compute_imbalanced_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_prob: Optional[pd.Series | np.ndarray] = None,
) -> Dict[str, float]:
    """Calculate comprehensive classification metrics specialized for severe class imbalance.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 = Legit, 1 = Fraud).
    y_pred : array-like
        Predicted binary labels.
    y_prob : array-like, optional
        Predicted fraud probabilities.

    Returns
    -------
    Dict[str, float]
        Dictionary with accuracy, precision, recall, f1_score, roc_auc, and pr_auc.
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


def calculate_business_cost_impact(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fp: float = 10.0,
    cost_fn: float = 400.0,
) -> Dict[str, float]:
    """Compute financial loss matrix based on asymmetric operational error costs.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels.
    cost_fp : float, default=10.0
        Cost of customer friction / manual alert review ($10).
    cost_fn : float, default=400.0
        Average unrecovered loss per missed fraudulent chargeback ($400).

    Returns
    -------
    Dict[str, float]
        Financial loss summary dictionary.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fp_loss = float(fp * cost_fp)
    fn_loss = float(fn * cost_fn)
    total_loss = fp_loss + fn_loss

    return {
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "fp_investigation_cost": fp_loss,
        "fn_unrecovered_fraud_loss": fn_loss,
        "total_financial_loss": total_loss,
    }


def evaluate_fraud_models(
    models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Benchmark all candidate fraud detection pipelines on held-out test data.

    Parameters
    ----------
    models : Dict[str, Pipeline]
        Trained model pipelines.
    X_test : pd.DataFrame
        Held-out test features.
    y_test : pd.Series
        Held-out test labels.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]
        Summary table, dictionary of predictions, and dictionary of probabilities.
    """
    records: List[Dict[str, Any]] = []
    predictions: Dict[str, np.ndarray] = {}
    probabilities: Dict[str, np.ndarray] = {}
    y_t = np.asarray(y_test, dtype=int)

    for name, pipeline in models.items():
        y_pred = pipeline.predict(X_test)
        predictions[name] = y_pred

        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            df_vals = pipeline.decision_function(X_test)
            y_prob = 1.0 / (1.0 + np.exp(-df_vals))
        else:
            y_prob = y_pred.astype(float)
        probabilities[name] = y_prob

        metrics = compute_imbalanced_metrics(y_t, y_pred, y_prob)
        cost_impact = calculate_business_cost_impact(y_t, y_pred)

        records.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-Score": metrics["f1_score"],
            "ROC-AUC": metrics.get("roc_auc", 0.0),
            "PR-AUC": metrics.get("pr_auc", 0.0),
            "Total Cost ($)": cost_impact["total_financial_loss"],
        })

    summary_df = pd.DataFrame(records).sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)
    return summary_df, predictions, probabilities


def plot_fraud_evaluations(
    y_test: pd.Series,
    predictions: Dict[str, np.ndarray],
    probabilities: Dict[str, np.ndarray],
    summary_df: pd.DataFrame,
    best_pipeline: Pipeline,
    output_dir: Optional[Path | str] = None,
) -> Dict[str, Path]:
    """Generate diagnostic visual charts for credit card fraud detection.

    Generates:
    1. Precision-Recall Curves (with baseline prevalence)
    2. ROC Curves
    3. Confusion Matrix Grid
    4. Cost-vs-Threshold Trade-off Curve

    Parameters
    ----------
    y_test : pd.Series
        Test ground truth.
    predictions : Dict[str, np.ndarray]
        Binary predictions for each model.
    probabilities : Dict[str, np.ndarray]
        Positive fraud probabilities.
    summary_df : pd.DataFrame
        Benchmark summary table.
    best_pipeline : Pipeline
        Top-performing pipeline.
    output_dir : Path or str, optional
        Target directory. Defaults to `visualizations/`.

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
    baseline_prev = float(np.mean(y_t))

    colors = ["#2980b9", "#27ae60", "#e67e22", "#8e44ad", "#c0392b"]

    # 1. Precision-Recall Curves (Crucial for Imbalanced Data)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(
        baseline_prev,
        color="red",
        linestyle="--",
        label=f"Baseline Prevalence ({baseline_prev:.2%})",
    )
    for idx, (name, y_prob) in enumerate(probabilities.items()):
        prec, rec, _ = precision_recall_curve(y_t, y_prob)
        pr_auc = average_precision_score(y_t, y_prob)
        ax.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.3f})", color=colors[idx % len(colors)], lw=2)

    ax.set_title("Precision-Recall Curves (Class Imbalance Robust)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Recall (Detection Rate of Stolen Cards)")
    ax.set_ylabel("Precision (Accuracy of Fraud Flags)")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    p1 = output_dir / "pr_curves.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=300)
    plt.close(fig)
    saved_plots["pr_curves"] = p1

    # 2. ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.50)")
    for idx, (name, y_prob) in enumerate(probabilities.items()):
        fpr, tpr, _ = roc_curve(y_t, y_prob)
        auc_score = roc_auc_score(y_t, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (ROC-AUC = {auc_score:.3f})", color=colors[idx % len(colors)], lw=2)

    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    p2 = output_dir / "roc_curves.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=300)
    plt.close(fig)
    saved_plots["roc_curves"] = p2

    # 3. Confusion Matrix Grid for Top 3 Models
    top_models = summary_df.head(3)["Model"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, name in enumerate(top_models):
        ax = axes[idx]
        cm = confusion_matrix(y_t, predictions[name])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Reds",
            cbar=False,
            ax=ax,
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"],
        )
        ax.set_title(f"{name}\nConfusion Matrix", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label" if idx == 0 else "")

    p3 = output_dir / "confusion_matrices.png"
    plt.tight_layout()
    plt.savefig(p3, dpi=300)
    plt.close(fig)
    saved_plots["confusion_matrices"] = p3

    # 4. Financial Cost Trade-off Curve across Thresholds
    best_model_name = summary_df.iloc[0]["Model"]
    best_prob = probabilities[best_model_name]
    thresholds = np.linspace(0.05, 0.95, 37)
    total_costs = []

    for th in thresholds:
        pred_th = (best_prob >= th).astype(int)
        loss_dict = calculate_business_cost_impact(y_t, pred_th)
        total_costs.append(loss_dict["total_financial_loss"])

    opt_idx = int(np.argmin(total_costs))
    opt_th = thresholds[opt_idx]
    opt_cost = total_costs[opt_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, total_costs, color="#c0392b", lw=2.5, label="Total Business Loss ($)")
    ax.scatter([opt_th], [opt_cost], color="#27ae60", s=100, zorder=5, label=f"Optimal Threshold ({opt_th:.2f}: ${opt_cost:,.0f})")
    ax.set_title(f"Operational Loss vs. Decision Threshold ({best_model_name})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Total Financial Loss ($)")
    ax.legend(frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    p4 = output_dir / "cost_tradeoff_curve.png"
    plt.tight_layout()
    plt.savefig(p4, dpi=300)
    plt.close(fig)
    saved_plots["cost_tradeoff"] = p4

    return saved_plots
