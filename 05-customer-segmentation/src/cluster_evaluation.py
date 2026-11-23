"""
Cluster Evaluation, Optimal K Selection, Diagnostic Visualizations, and Persona Profiling.

Computes Inertia (Elbow Method), Silhouette Scores, Davies-Bouldin, and Calinski-Harabasz indices,
profiles clusters into retail customer personas, and renders diagnostic charts.
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
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from .data_loader import CUSTOMER_FEATURES


PERSONA_TEMPLATES: Dict[str, Dict[str, str]] = {
    "Affluent VIP Champions": {
        "description": "High income, highest spending score, high purchase frequency, and large basket size.",
        "marketing_strategy": "Concierge VIP service, exclusive early product drops, premium loyalty tiers, and personalized gifts.",
    },
    "Affluent Cautious Savers": {
        "description": "High income, conservative spending score, low order frequency, and selective high-ticket purchases.",
        "marketing_strategy": "Quality assurance messaging, long-term warranty highlights, financial value retention, and premium essentials.",
    },
    "Young Impulse Trendsetters": {
        "description": "Younger demographics, low-to-mid income, high spending score, and high frequency of trendy impulse buys.",
        "marketing_strategy": "Flash sales, viral social media marketing, Buy-Now-Pay-Later (BNPL) options, and limited edition launches.",
    },
    "Frugal Budget Planners": {
        "description": "Low income, low spending score, low frequency, and highly price-sensitive baskets.",
        "marketing_strategy": "Clearance discount alerts, bulk savings vouchers, price-match guarantees, and free delivery thresholds.",
    },
    "Core Mainstream Shoppers": {
        "description": "Moderate income, moderate spending score, and consistent steady purchasing cadence.",
        "marketing_strategy": "Seasonal cross-selling, recurring replenishment subscriptions, and customer loyalty point rewards.",
    },
}


def compute_elbow_and_silhouette_sweep(
    X_scaled: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> pd.DataFrame:
    """Calculate Inertia and Silhouette scores across a range of cluster counts (k).

    Parameters
    ----------
    X_scaled : np.ndarray
        Standardized feature matrix.
    k_range : range, default=range(2, 11)
        Range of k values to sweep.
    random_state : int, default=42
        Reproducible random seed.

    Returns
    -------
    pd.DataFrame
        Table with 'k', 'Inertia', and 'Silhouette_Score'.
    """
    records = []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        labels = km.fit_predict(X_scaled)
        inertia = float(km.inertia_)
        sil = float(silhouette_score(X_scaled, labels))
        records.append({
            "k": k,
            "Inertia": round(inertia, 2),
            "Silhouette_Score": round(sil, 4),
        })

    return pd.DataFrame(records)


def evaluate_clustering_metrics(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """Compute Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics for assigned clusters.

    Parameters
    ----------
    X_scaled : np.ndarray
        Standardized feature matrix.
    labels : np.ndarray
        Assigned cluster labels.
    model_name : str
        Algorithm name.

    Returns
    -------
    Dict[str, Any]
        Metric evaluation dictionary.
    """
    unique_labels = np.unique(labels)
    # Filter noise points (-1 in DBSCAN) for metric computation if needed
    valid_mask = labels != -1
    n_valid_clusters = len(set(unique_labels) - {-1})

    if n_valid_clusters < 2:
        return {
            "Model": model_name,
            "Clusters": n_valid_clusters,
            "Silhouette": 0.0,
            "Davies_Bouldin": 99.0,
            "Calinski_Harabasz": 0.0,
            "Noise_Points": int((labels == -1).sum()),
        }

    X_eval = X_scaled[valid_mask]
    labels_eval = labels[valid_mask]

    sil = float(silhouette_score(X_eval, labels_eval))
    db = float(davies_bouldin_score(X_eval, labels_eval))
    ch = float(calinski_harabasz_score(X_eval, labels_eval))

    return {
        "Model": model_name,
        "Clusters": n_valid_clusters,
        "Silhouette": round(sil, 4),
        "Davies_Bouldin": round(db, 4),
        "Calinski_Harabasz": round(ch, 2),
        "Noise_Points": int((labels == -1).sum()),
    }


def profile_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    features: Optional[List[str]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Synthesize descriptive statistics and marketing personas for each cluster.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned customer records.
    labels : np.ndarray
        Assigned cluster IDs.
    features : List[str], optional
        Feature list. Defaults to CUSTOMER_FEATURES.

    Returns
    -------
    Dict[int, Dict[str, Any]]
        Cluster profiling dictionary.
    """
    if features is None:
        features = CUSTOMER_FEATURES

    df_clustered = df.copy()
    df_clustered["Cluster"] = labels

    profiles: Dict[int, Dict[str, Any]] = {}
    unique_clusters = sorted([c for c in np.unique(labels) if c != -1])

    for cid in unique_clusters:
        sub = df_clustered[df_clustered["Cluster"] == cid]
        means = sub[features].mean().round(2).to_dict()
        medians = sub[features].median().round(2).to_dict()
        size = len(sub)
        pct = round(size / len(df_clustered) * 100.0, 1)

        # Classify persona by income and spending score
        inc = means.get("Annual_Income_k", 50.0)
        spend = means.get("Spending_Score", 50.0)

        if inc >= 75.0 and spend >= 65.0:
            p_name = "Affluent VIP Champions"
        elif inc >= 75.0 and spend < 45.0:
            p_name = "Affluent Cautious Savers"
        elif inc < 50.0 and spend >= 65.0:
            p_name = "Young Impulse Trendsetters"
        elif inc < 50.0 and spend < 45.0:
            p_name = "Frugal Budget Planners"
        else:
            p_name = "Core Mainstream Shoppers"

        info = PERSONA_TEMPLATES[p_name]

        profiles[int(cid)] = {
            "cluster_id": int(cid),
            "persona_name": p_name,
            "customer_count": size,
            "percentage": pct,
            "mean_attributes": means,
            "median_attributes": medians,
            "description": info["description"],
            "marketing_strategy": info["marketing_strategy"],
        }

    return profiles


def plot_clustering_diagnostics(
    X_scaled: np.ndarray,
    X_pca: np.ndarray,
    labels_dict: Dict[str, np.ndarray],
    sweep_df: pd.DataFrame,
    profiles: Dict[int, Dict[str, Any]],
    output_dir: Optional[Path | str] = None,
) -> Dict[str, Path]:
    """Generate diagnostic visual charts for customer segmentation.

    Generates:
    1. Elbow & Silhouette Analysis Curve
    2. PCA 2D Cluster Scatterplot with Centroids
    3. Standardized Feature Heatmap by Cluster Profile
    4. Multi-Algorithm Clustering Comparison (K-Means vs Hierarchical vs DBSCAN)

    Parameters
    ----------
    X_scaled : np.ndarray
        Standardized feature matrix.
    X_pca : np.ndarray
        2D PCA projection coordinates.
    labels_dict : Dict[str, np.ndarray]
        Labels for each algorithm.
    sweep_df : pd.DataFrame
        Elbow/Silhouette sweep dataframe.
    profiles : Dict[int, Dict[str, Any]]
        Persona profiling dictionary.
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

    # 1. Elbow & Silhouette Curve (Dual-Axis)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "#2980b9"
    color2 = "#27ae60"

    ax1.set_xlabel("Number of Clusters (k)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)", color=color1, fontsize=11, fontweight="bold")
    ax1.plot(sweep_df["k"], sweep_df["Inertia"], marker="o", color=color1, lw=2.5, label="Inertia (Elbow)")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Silhouette Score", color=color2, fontsize=11, fontweight="bold")
    ax2.plot(sweep_df["k"], sweep_df["Silhouette_Score"], marker="s", color=color2, lw=2.5, linestyle="--", label="Silhouette")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("Elbow Method & Silhouette Score vs. Number of Clusters (k)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    p1 = output_dir / "elbow_silhouette_analysis.png"
    plt.savefig(p1, dpi=300)
    plt.close(fig)
    saved_plots["elbow_silhouette"] = p1

    # 2. PCA 2D Cluster Scatterplot
    fig, ax = plt.subplots(figsize=(9, 6))
    km_labels = labels_dict.get("K-Means", list(labels_dict.values())[0])
    palette = sns.color_palette("Set2", len(np.unique(km_labels)))

    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=km_labels,
        cmap="tab10",
        alpha=0.65,
        s=40,
        edgecolor="none",
    )

    # Plot Cluster Centroids in PCA space
    for cid in np.unique(km_labels):
        if cid == -1:
            continue
        mask = km_labels == cid
        center = X_pca[mask].mean(axis=0)
        p_name = profiles.get(cid, {}).get("persona_name", f"Cluster {cid}")
        ax.scatter(center[0], center[1], c="black", marker="X", s=180, zorder=10)
        ax.annotate(
            f"C{cid}: {p_name}",
            (center[0], center[1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontweight="bold",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.7),
        )

    ax.set_title("Customer Segments in 2D PCA Latent Space", fontsize=13, fontweight="bold")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.grid(True, linestyle=":", alpha=0.5)

    p2 = output_dir / "cluster_scatterplot_pca.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=300)
    plt.close(fig)
    saved_plots["pca_scatterplot"] = p2

    # 3. Cluster Profile Standardized Feature Heatmap
    profile_rows = []
    cluster_names = []
    for cid, p_info in sorted(profiles.items()):
        means = p_info["mean_attributes"]
        profile_rows.append(means)
        cluster_names.append(f"C{cid}: {p_info['persona_name']}")

    heat_df = pd.DataFrame(profile_rows, index=cluster_names)
    # Standardize across clusters for heatmap color contrast
    heat_norm = (heat_df - heat_df.mean()) / heat_df.std()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        heat_norm,
        annot=heat_df.round(1),
        fmt="g",
        cmap="Blues",
        cbar_kws={"label": "Normalized Z-Score Across Segments"},
        ax=ax,
    )
    ax.set_title("Customer Persona Feature Profiles (Mean Values & Relative Z-Scores)", fontsize=12, fontweight="bold")
    p3 = output_dir / "cluster_profiles_heatmap.png"
    plt.tight_layout()
    plt.savefig(p3, dpi=300)
    plt.close(fig)
    saved_plots["profile_heatmap"] = p3

    # 4. Multi-Algorithm Clustering Comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    algo_names = list(labels_dict.keys())

    for idx, name in enumerate(algo_names):
        ax = axes[idx]
        lbls = labels_dict[name]
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=lbls, cmap="tab10", s=25, alpha=0.6)
        n_c = len(set(lbls) - {-1})
        ax.set_title(f"{name}\n({n_c} Clusters)", fontsize=11, fontweight="bold")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2" if idx == 0 else "")
        ax.grid(True, linestyle=":", alpha=0.5)

    p4 = output_dir / "algorithm_comparison.png"
    plt.tight_layout()
    plt.savefig(p4, dpi=300)
    plt.close(fig)
    saved_plots["algorithm_comparison"] = p4

    return saved_plots
