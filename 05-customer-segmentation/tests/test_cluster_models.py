"""
Unit Tests for Clustering Algorithms, Model Training, and Artifact Persistence.
"""

from pathlib import Path
import pytest
import numpy as np
from src.data_loader import generate_customer_data
from src.data_preprocessing import preprocess_customer_data, scale_customer_features
from src.cluster_models import (
    train_kmeans,
    train_hierarchical,
    train_dbscan,
    train_all_clustering_models,
    save_clustering_artifact,
    load_clustering_artifact,
)


@pytest.fixture
def scaled_customer_data():
    """Fixture providing scaled customer behavioral matrix."""
    raw = generate_customer_data(n_samples=250, seed=42)
    cleaned, _ = preprocess_customer_data(raw)
    X_scaled, scaler = scale_customer_features(cleaned)
    return X_scaled, scaler


def test_train_kmeans(scaled_customer_data):
    """Verify KMeans model training and cluster label assignment."""
    X_scaled, _ = scaled_customer_data
    km, labels = train_kmeans(X_scaled, n_clusters=4, random_state=42)
    assert len(labels) == len(X_scaled)
    assert len(np.unique(labels)) == 4
    assert hasattr(km, "cluster_centers_")
    assert hasattr(km, "inertia_")


def test_train_hierarchical(scaled_customer_data):
    """Verify Agglomerative Hierarchical clustering."""
    X_scaled, _ = scaled_customer_data
    h_model, labels = train_hierarchical(X_scaled, n_clusters=4)
    assert len(labels) == len(X_scaled)
    assert len(np.unique(labels)) == 4


def test_train_dbscan(scaled_customer_data):
    """Verify DBSCAN density clustering."""
    X_scaled, _ = scaled_customer_data
    db_model, labels = train_dbscan(X_scaled, eps=0.8, min_samples=10)
    assert len(labels) == len(X_scaled)
    assert isinstance(labels, np.ndarray)


def test_train_all_clustering_models(scaled_customer_data):
    """Verify training all 3 clustering algorithms."""
    X_scaled, _ = scaled_customer_data
    models_dict = train_all_clustering_models(X_scaled, n_clusters=5)
    assert "K-Means" in models_dict
    assert "Hierarchical (Ward)" in models_dict
    assert "DBSCAN" in models_dict


def test_save_and_load_clustering_artifact(scaled_customer_data, tmp_path: Path):
    """Verify saving and loading clustering artifact and profiles."""
    X_scaled, scaler = scaled_customer_data
    km, labels = train_kmeans(X_scaled, n_clusters=3, random_state=42)

    profiles = {
        0: {"persona_name": "VIP", "percentage": 50.0, "mean_attributes": {}},
        1: {"persona_name": "Saver", "percentage": 50.0, "mean_attributes": {}},
    }
    metrics = {"silhouette": 0.52}

    saved_path = save_clustering_artifact(
        model=km,
        scaler=scaler,
        algorithm_name="K-Means",
        n_clusters=3,
        metrics=metrics,
        cluster_profiles=profiles,
        output_dir=tmp_path,
    )
    assert saved_path.exists()

    loaded = load_clustering_artifact(saved_path)
    assert loaded["algorithm_name"] == "K-Means"
    assert loaded["n_clusters"] == 3
    assert loaded["metrics"]["silhouette"] == 0.52
    assert "model" in loaded
    assert "scaler" in loaded


def test_load_clustering_artifact_missing(tmp_path: Path):
    """Verify FileNotFoundError when artifact path does not exist."""
    fake_path = tmp_path / "missing_cluster.joblib"
    with pytest.raises(FileNotFoundError):
        load_clustering_artifact(fake_path)
