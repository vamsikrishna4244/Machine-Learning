"""
Model Evaluation, Feature Importance Interpretability, Demographic Fairness, and Visualizations.

Computes classification metrics, extracts Gini feature importances, performs demographic
disparate impact fairness checks, and generates publication-grade diagnostic charts.
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


def compute_attrition_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_prob: Optional[pd.Series | np.ndarray] = None,
) -> Dict[str, float]:
    """Calculate core classification metrics for attrition evaluation.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels (0 = Stayed, 1 = Left).
    y_pred : array-like
        Predicted binary labels.
    y_prob : array-like, optional
        Predicted positive attrition probabilities.

    Returns
    -------
    Dict[str, float]
        Dictionary with accuracy, precision, recall, f1_score, and optional roc_auc, pr_auc.
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


def evaluate_attrition_models(
    models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Benchmark all candidate attrition models on held-out test data.

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
        Benchmark summary dataframe, dictionary of predictions, and dictionary of probabilities.
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

        metrics = compute_attrition_metrics(y_t, y_pred, y_prob)

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


def extract_top_feature_importances(
    pipeline: Pipeline,
    top_n: int = 15,
) -> pd.DataFrame:
    """Extract and rank the top predictive features from the fitted pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        Trained Scikit-learn Pipeline with preprocessor and classifier.
    top_n : int, default=15
        Number of top features to return.

    Returns
    -------
    pd.DataFrame
        Table with 'Feature', 'Importance', and 'Sign'.
    """
    if pipeline is None or not hasattr(pipeline, "named_steps"):
        return pd.DataFrame(columns=["Feature", "Importance", "Sign"])

    preprocessor = pipeline.named_steps.get("preprocessor")
    classifier = pipeline.named_steps.get("classifier")

    if preprocessor is None or classifier is None:
        return pd.DataFrame(columns=["Feature", "Importance", "Sign"])

    # Get transformed column names
    feat_names = preprocessor.get_feature_names_out()
    clean_names = [
        f.replace("num__", "").replace("cat__", "") for f in feat_names
    ]

    if hasattr(classifier, "feature_importances_"):
        weights = classifier.feature_importances_
        sign = ["Positive"] * len(weights)
    elif hasattr(classifier, "coef_"):
        raw_coefs = classifier.coef_[0]
        weights = np.abs(raw_coefs)
        sign = ["Positive (Risk)" if c > 0 else "Negative (Protective)" for c in raw_coefs]
    else:
        return pd.DataFrame(columns=["Feature", "Importance", "Sign"])

    df_imp = pd.DataFrame({
        "Feature": clean_names,
        "Importance": weights,
        "Sign": sign,
    }).sort_values(by="Importance", ascending=False).head(top_n).reset_index(drop=True)

    return df_imp


def conduct_demographic_fairness_audit(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    sensitive_col: str = "Gender",
) -> pd.DataFrame:
    """Perform demographic parity check to evaluate potential model bias across groups.

    Computes predicted positive attrition rates and disparate impact ratios (Four-Fifths / 80% rule).

    Parameters
    ----------
    pipeline : Pipeline
        Fitted model pipeline.
    X_test : pd.DataFrame
        Test features containing demographic attributes.
    sensitive_col : str, default='Gender'
        Protected demographic attribute.

    Returns
    -------
    pd.DataFrame
        Audit table showing predicted positive rates and disparate impact ratios.
    """
    if sensitive_col not in X_test.columns:
        return pd.DataFrame()

    preds = pipeline.predict(X_test)
    df_eval = pd.DataFrame({sensitive_col: X_test[sensitive_col].to_numpy(), "Predicted_Attrition": preds})

    audit_records = []
    groups = df_eval[sensitive_col].unique()
    max_rate = 0.0

    group_rates = {}
    for g in groups:
        sub = df_eval[df_eval[sensitive_col] == g]
        rate = float(sub["Predicted_Attrition"].mean())
        group_rates[g] = rate
        if rate > max_rate:
            max_rate = rate

    for g in groups:
        sub = df_eval[df_eval[sensitive_col] == g]
        rate = group_rates[g]
        disparate_impact_ratio = round(rate / max_rate, 3) if max_rate > 0 else 1.0
        eeoc_pass = bool(disparate_impact_ratio >= 0.80)

        audit_records.append({
            "Demographic_Group": g,
            "Sample_Count": len(sub),
            "Predicted_Attrition_Rate": round(rate, 4),
            "Disparate_Impact_Ratio": disparate_impact_ratio,
            "Passes_80_Pct_Rule": eeoc_pass,
        })

    return pd.DataFrame(audit_records)


def plot_attrition_evaluations(
    y_test: pd.Series,
    predictions: Dict[str, np.ndarray],
    probabilities: Dict[str, np.ndarray],
    summary_df: pd.DataFrame,
    best_pipeline: Pipeline,
    output_dir: Optional[Path | str] = None,
) -> Dict[str, Path]:
    """Generate diagnostic visual charts for employee attrition prediction.

    Generates:
    1. Multi-metric model comparison bar chart
    2. Side-by-side confusion matrix grid
    3. ROC and Precision-Recall Curves
    4. Top 15 Feature Importances

    Parameters
    ----------
    y_test : pd.Series
        Test ground truth.
    predictions : Dict[str, np.ndarray]
        Model prediction dictionary.
    probabilities : Dict[str, np.ndarray]
        Positive attrition probabilities.
    summary_df : pd.DataFrame
        Benchmark table.
    best_pipeline : Pipeline
        Top tuned pipeline.
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

    # 1. Multi-Metric Benchmark Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_melted = summary_df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "Recall", "F1-Score", "ROC-AUC"],
        var_name="Metric",
        value_name="Score",
    )
    sns.barplot(data=metrics_melted, x="Model", y="Score", hue="Metric", palette="Blues_r", ax=ax)
    ax.set_ylim(0, 1.1)
    ax.set_title("Employee Attrition Model Comparison across Metrics", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    p1 = output_dir / "model_comparison_benchmark.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=300)
    plt.close(fig)
    saved_plots["model_comparison"] = p1

    # 2. Confusion Matrices for All Candidates
    fig, axes = plt.subplots(1, len(predictions), figsize=(5 * len(predictions), 4.2))
    if len(predictions) == 1:
        axes = [axes]
    for idx, (name, y_pred) in enumerate(predictions.items()):
        ax = axes[idx]
        cm = confusion_matrix(y_t, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Stayed", "Left"],
            yticklabels=["Stayed", "Left"],
        )
        ax.set_title(f"{name}\nConfusion Matrix", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual" if idx == 0 else "")

    p2 = output_dir / "confusion_matrices.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=300)
    plt.close(fig)
    saved_plots["confusion_matrices"] = p2

    # 3. Combined ROC and Precision-Recall Curves
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = ["#2980b9", "#27ae60", "#e67e22", "#8e44ad"]

    ax_roc.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.50)")
    baseline_pr = float(np.mean(y_t))
    ax_pr.axhline(baseline_pr, color="red", linestyle="--", label=f"Prevalence ({baseline_pr:.1%})")

    for idx, (name, y_prob) in enumerate(probabilities.items()):
        fpr, tpr, _ = roc_curve(y_t, y_prob)
        auc_score = roc_auc_score(y_t, y_prob)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})", color=colors[idx % len(colors)], lw=2)

        prec, rec, _ = precision_recall_curve(y_t, y_prob)
        pr_score = average_precision_score(y_t, y_prob)
        ax_pr.plot(rec, prec, label=f"{name} (PR-AUC = {pr_score:.3f})", color=colors[idx % len(colors)], lw=2)

    ax_roc.set_title("Receiver Operating Characteristic (ROC)", fontsize=11, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate (Recall)")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, linestyle=":", alpha=0.5)

    ax_pr.set_title("Precision-Recall Curve", fontsize=11, fontweight="bold")
    ax_pr.set_xlabel("Recall (Attrition Detection Rate)")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="upper right")
    ax_pr.grid(True, linestyle=":", alpha=0.5)

    p3 = output_dir / "roc_pr_curves.png"
    plt.tight_layout()
    plt.savefig(p3, dpi=300)
    plt.close(fig)
    saved_plots["roc_pr_curves"] = p3

    # 4. Top 15 Feature Importances
    feat_df = extract_top_feature_importances(best_pipeline, top_n=15)
    if not feat_df.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(feat_df["Feature"][::-1], feat_df["Importance"][::-1], color="#2c3e50")
        best_name = summary_df.iloc[0]["Model"]
        ax.set_title(f"Top 15 Most Predictive Attrition Drivers ({best_name})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Relative Feature Importance")
        ax.grid(axis="x", linestyle=":", alpha=0.5)

        p4 = output_dir / "feature_importance_top15.png"
        plt.tight_layout()
        plt.savefig(p4, dpi=300)
        plt.close(fig)
        saved_plots["feature_importance"] = p4

    return saved_plots
