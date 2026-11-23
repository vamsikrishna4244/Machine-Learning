"""
Unit Tests for Cluster Evaluation, Optimal K Sweeps, and Diagnostics.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from src.data_loader import generate_customer_data
from src.data_preprocessing import preprocess_customer_data, scale_customer_features, apply_pca_reduction
from src.cluster_models import train_kmeans
from src.cluster_evaluation import (
    compute_elbow_and_silhouette_sweep,
    evaluate_clustering_metrics,
    profile_clusters,
    plot_clustering_diagnostics,
)


@pytest.fixture
def prepared_cluster_data():
    """Fixture providing cleaned dataframe, scaled array, and PCA coordinates."""
    raw = generate_customer_data(n_samples=250, seed=42)
    cleaned, _ = preprocess_customer_data(raw)
    X_scaled, _ = scale_customer_features(cleaned)
    X_pca, _, _ = apply_pca_reduction(X_scaled)
    return cleaned, X_scaled, X_pca


def test_compute_elbow_and_silhouette_sweep(prepared_cluster_data):
    """Verify sweeping k in [2, 5] computes Inertia and Silhouette scores."""
    _, X_scaled, _ = prepared_cluster_data
    sweep_df = compute_elbow_and_silhouette_sweep(X_scaled, k_range=range(2, 6))

    assert isinstance(sweep_df, pd.DataFrame)
    assert len(sweep_df) == 4
    assert "Inertia" in sweep_df.columns
    assert "Silhouette_Score" in sweep_df.columns
    # Check that inertia drops monotonically
    inertias = sweep_df["Inertia"].tolist()
    assert all(inertias[i] >= inertias[i + 1] for i in range(len(inertias) - 1))


def test_evaluate_clustering_metrics(prepared_cluster_data):
    """Verify calculation of Silhouette, Davies-Bouldin, and Calinski-Harabasz."""
    _, X_scaled, _ = prepared_cluster_data
    _, labels = train_kmeans(X_scaled, n_clusters=4, random_state=42)

    metrics = evaluate_clustering_metrics(X_scaled, labels, "K-Means")
    assert "Silhouette" in metrics
    assert "Davies_Bouldin" in metrics
    assert "Calinski_Harabasz" in metrics
    assert -1.0 <= metrics["Silhouette"] <= 1.0
    assert metrics["Davies_Bouldin"] >= 0.0


def test_profile_clusters(prepared_cluster_data):
    """Verify cluster profiling into descriptive statistics and retail personas."""
    cleaned, X_scaled, _ = prepared_cluster_data
    _, labels = train_kmeans(X_scaled, n_clusters=5, random_state=42)

    profiles = profile_clusters(cleaned, labels)
    assert len(profiles) == 5
    for cid, p in profiles.items():
        assert "persona_name" in p
        assert "mean_attributes" in p
        assert "description" in p
        assert "marketing_strategy" in p


def test_plot_clustering_diagnostics(prepared_cluster_data, tmp_path: Path):
    """Verify generation of all 4 diagnostic clustering visualizations."""
    cleaned, X_scaled, X_pca = prepared_cluster_data
    _, labels = train_kmeans(X_scaled, n_clusters=5, random_state=42)
    sweep_df = compute_elbow_and_silhouette_sweep(X_scaled, k_range=range(2, 5))
    profiles = profile_clusters(cleaned, labels)

    plots = plot_clustering_diagnostics(
        X_scaled=X_scaled,
        X_pca=X_pca,
        labels_dict={"K-Means": labels, "Hierarchical": labels, "DBSCAN": labels},
        sweep_df=sweep_df,
        profiles=profiles,
        output_dir=tmp_path,
    )

    assert "elbow_silhouette" in plots
    assert "pca_scatterplot" in plots
    assert "profile_heatmap" in plots
    assert "algorithm_comparison" in plots
    for p in plots.values():
        assert p.exists()
