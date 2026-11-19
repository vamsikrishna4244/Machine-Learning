"""
Model Evaluation, Metric Computation, and Residual Analysis Module.

Computes MAE, MSE, RMSE, R² scores, generates comparative evaluation tables,
and renders publication-grade residual and feature importance visualizations.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


def compute_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> Dict[str, float]:
    """Compute standard regression performance metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth target values.
    y_pred : array-like
        Predicted values from estimator.

    Returns
    -------
    Dict[str, float]
        Dictionary with MAE, MSE, RMSE, and R² score.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_t, y_p))
    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_t, y_p))

    # Mean Absolute Percentage Error (MAPE)
    non_zero = y_t != 0
    mape = float(np.mean(np.abs((y_t[non_zero] - y_p[non_zero]) / y_t[non_zero])) * 100.0)

    return {
        "mae": round(mae, 2),
        "mse": round(mse, 2),
        "rmse": round(rmse, 2),
        "r2_score": round(r2, 4),
        "mape_pct": round(mape, 2),
    }


def evaluate_models(
    models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Evaluate a collection of trained models on the held-out test set.

    Parameters
    ----------
    models : Dict[str, Pipeline]
        Dictionary of trained model pipelines.
    X_test : pd.DataFrame
        Held-out test features.
    y_test : pd.Series
        Held-out ground truth target values.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, np.ndarray]]
        Comparison summary dataframe and dictionary of test predictions.
    """
    records: List[Dict[str, Any]] = []
    predictions: Dict[str, np.ndarray] = {}

    for name, pipeline in models.items():
        y_pred = pipeline.predict(X_test)
        predictions[name] = y_pred
        metrics = compute_regression_metrics(y_test, y_pred)
        records.append({
            "Model": name,
            "MAE ($)": metrics["mae"],
            "RMSE ($)": metrics["rmse"],
            "R² Score": metrics["r2_score"],
            "MAPE (%)": metrics["mape_pct"],
        })

    summary_df = pd.DataFrame(records).sort_values(by="R² Score", ascending=False).reset_index(drop=True)
    return summary_df, predictions


def compare_models(summary_df: pd.DataFrame) -> str:
    """Identify the top-performing model based on highest R² score and lowest RMSE."""
    best_row = summary_df.iloc[0]
    return str(best_row["Model"])


def plot_model_evaluations(
    y_test: pd.Series,
    predictions: Dict[str, np.ndarray],
    summary_df: pd.DataFrame,
    best_pipeline: Pipeline,
    output_dir: Optional[Path | str] = None,
) -> Dict[str, Path]:
    """Generate and save publication-grade visual evaluation figures.

    Parameters
    ----------
    y_test : pd.Series
        Ground truth test values.
    predictions : Dict[str, np.ndarray]
        Test predictions for each model.
    summary_df : pd.DataFrame
        Model comparison summary.
    best_pipeline : Pipeline
        Best model pipeline for feature importance extraction.
    output_dir : Path or str, optional
        Target directory for saved figures. Defaults to `visualizations/`.

    Returns
    -------
    Dict[str, Path]
        Paths to generated plot artifacts.
    """
    if output_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root / "visualizations"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_plots: Dict[str, Path] = {}

    best_model_name = compare_models(summary_df)
    y_best_pred = predictions[best_model_name]
    y_t = y_test.to_numpy()

    # 1. Actual vs Predicted Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_t / 1000.0, y_best_pred / 1000.0, alpha=0.5, color="#2980b9", edgecolors="none", s=25)
    min_val = min(y_t.min(), y_best_pred.min()) / 1000.0
    max_val = max(y_t.max(), y_best_pred.max()) / 1000.0
    ax.plot([min_val, max_val], [min_val, max_val], color="#e74c3c", linestyle="--", linewidth=2, label="Ideal Fit (y = x)")
    ax.set_title(f"Actual vs. Predicted House Prices ({best_model_name})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Actual Price ($ in Thousands)", fontsize=11)
    ax.set_ylabel("Predicted Price ($ in Thousands)", fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    p1 = output_dir / "actual_vs_predicted.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=300)
    plt.close(fig)
    saved_plots["actual_vs_predicted"] = p1

    # 2. Residual Distribution & Residual vs Predicted
    residuals = y_t - y_best_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(residuals / 1000.0, kde=True, ax=ax1, color="#8e44ad", bins=30)
    ax1.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax1.set_title("Residual Error Distribution", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Residual (Actual - Predicted, $ in Thousands)")
    ax1.set_ylabel("Density")
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2.scatter(y_best_pred / 1000.0, residuals / 1000.0, alpha=0.5, color="#16a085", s=25)
    ax2.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax2.set_title("Residuals vs. Predicted Values", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Predicted Price ($ in Thousands)")
    ax2.set_ylabel("Residual ($ in Thousands)")
    ax2.grid(True, linestyle=":", alpha=0.5)

    p2 = output_dir / "residual_analysis.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=300)
    plt.close(fig)
    saved_plots["residual_analysis"] = p2

    # 3. Model Comparison Bar Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#27ae60", "#2980b9", "#f39c12"]
    ax1.bar(summary_df["Model"], summary_df["R² Score"], color=colors[:len(summary_df)])
    ax1.set_title("R² Score Comparison (Higher is Better)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("R² Score")
    ax1.set_ylim(0, 1.0)
    for idx, v in enumerate(summary_df["R² Score"]):
        ax1.text(idx, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    ax2.bar(summary_df["Model"], summary_df["RMSE ($)"] / 1000.0, color=colors[:len(summary_df)])
    ax2.set_title("RMSE Comparison (Lower is Better)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("RMSE ($ in Thousands)")
    for idx, v in enumerate(summary_df["RMSE ($)"] / 1000.0):
        ax2.text(idx, v + 1.0, f"${v:.1f}k", ha="center", fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    p3 = output_dir / "model_comparison.png"
    plt.tight_layout()
    plt.savefig(p3, dpi=300)
    plt.close(fig)
    saved_plots["model_comparison"] = p3

    # 4. Feature Importance / Model Coefficients
    regressor = best_pipeline.named_steps.get("regressor")
    preprocessor = best_pipeline.named_steps.get("preprocessor")
    if preprocessor is not None:
        feature_names = preprocessor.get_feature_names_out()
        clean_names = [f.replace("num__", "").replace("cat__neighborhood_", "Neighborhood: ") for f in feature_names]

        if hasattr(regressor, "feature_importances_"):
            importances = regressor.feature_importances_
            feat_df = pd.DataFrame({"Feature": clean_names, "Importance": importances}).sort_values(by="Importance", ascending=True)
            title = f"Top 12 Predictive Features ({best_model_name})"
            xlabel = "Relative Gini Feature Importance"
        elif hasattr(regressor, "coef_"):
            coefs = regressor.coef_
            feat_df = pd.DataFrame({"Feature": clean_names, "Importance": np.abs(coefs)}).sort_values(by="Importance", ascending=True)
            title = f"Top 12 Absolute Feature Coefficients ({best_model_name})"
            xlabel = "Standardized Coefficient Magnitude (|β|)"
        else:
            feat_df = None

        if feat_df is not None:
            fig, ax = plt.subplots(figsize=(9, 7))
            ax.barh(feat_df["Feature"].tail(12), feat_df["Importance"].tail(12), color="#2c3e50")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel(xlabel)
            ax.grid(axis="x", linestyle=":", alpha=0.5)
            p4 = output_dir / "feature_importance.png"
            plt.tight_layout()
            plt.savefig(p4, dpi=300)
            plt.close(fig)
            saved_plots["feature_importance"] = p4

    return saved_plots
