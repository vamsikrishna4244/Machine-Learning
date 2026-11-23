"""
Unsupervised Clustering Algorithms and Model Artifact Serialization.

Implements K-Means, Agglomerative Hierarchical Clustering, and DBSCAN,
providing model fitting, cluster labeling, and Joblib artifact persistence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler


def train_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42,
) -> Tuple[KMeans, np.ndarray]:
    """Train K-Means clustering algorithm with k-means++ centroid initialization.

    Parameters
    ----------
    X_scaled : np.ndarray
        Standardized feature matrix.
    n_clusters : int, default=5
        Number of clusters to partition.
    random_state : int, default=42
        Reproducible random seed.

    Returns
    -------
    Tuple[KMeans, np.ndarray]
        Fitted KMeans estimator and assigned cluster labels.
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=random_state,
    )
    labels = kmeans.fit_predict(X_scaled)
    return kmeans, labels


def train_hierarchical(
    X_scaled: np.ndarray,
    n_clusters: int = 5,
    linkage: str = "ward",
) -> Tuple[AgglomerativeClustering, np.ndarray]:
    """Train Agglomerative Hierarchical Clustering using Ward variance minimization.

    Parameters
    ----------
    X_scaled : np.ndarray
        Standardized feature matrix.
    n_clusters : int, default=5
        Target cluster count.
    linkage : {'ward', 'complete', 'average'}, default='ward'
        Linkage distance criterion.

    Returns
    -------
    Tuple[AgglomerativeClustering, np.ndarray]
        Fitted AgglomerativeClustering estimator and assigned cluster labels.
    """
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
    )
    labels = hierarchical.fit_predict(X_scaled)
    return hierarchical, labels


def train_dbscan(
    X_scaled: np.ndarray,
    eps: float = 0.85,
    min_samples: int = 15,
) -> Tuple[DBSCAN, np.ndarray]:
    """Train Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    Parameters
    ----------
    X_scaled : np.ndarray
        Standardized feature matrix.
    eps : float, default=0.85
        Maximum distance between two samples for one to be considered in neighborhood.
    min_samples : int, default=15
        Number of samples in neighborhood for a point to be considered core.

    Returns
    -------
    Tuple[DBSCAN, np.ndarray]
        Fitted DBSCAN estimator and cluster labels (-1 denotes noise).
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    return dbscan, labels


def train_all_clustering_models(
    X_scaled: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42,
) -> Dict[str, Tuple[Any, np.ndarray]]:
    """Train K-Means, Hierarchical, and DBSCAN under identical standardized features.

    Parameters
    ----------
    X_scaled : np.ndarray
        Standardized feature array.
    n_clusters : int, default=5
        Cluster count for partition-based algorithms.
    random_state : int, default=42
        Reproducible random seed.

    Returns
    -------
    Dict[str, Tuple[Any, np.ndarray]]
        Dictionary mapping model names to (estimator, labels).
    """
    km_model, km_labels = train_kmeans(X_scaled, n_clusters=n_clusters, random_state=random_state)
    h_model, h_labels = train_hierarchical(X_scaled, n_clusters=n_clusters)
    db_model, db_labels = train_dbscan(X_scaled)

    return {
        "K-Means": (km_model, km_labels),
        "Hierarchical (Ward)": (h_model, h_labels),
        "DBSCAN": (db_model, db_labels),
    }


def save_clustering_artifact(
    model: Any,
    scaler: StandardScaler,
    algorithm_name: str,
    n_clusters: int,
    metrics: Dict[str, float],
    cluster_profiles: Dict[int, Dict[str, Any]],
    output_dir: Optional[Path | str] = None,
) -> Path:
    """Serialize the top clustering model, scaler, and persona profiles.

    Parameters
    ----------
    model : estimator
        Fitted clustering estimator.
    scaler : StandardScaler
        Fitted feature scaler.
    algorithm_name : str
        Algorithm title.
    n_clusters : int
        Number of clusters.
    metrics : Dict[str, float]
        Evaluation metric dictionary.
    cluster_profiles : Dict[int, Dict[str, Any]]
        Persona profiling dictionary.
    output_dir : Path or str, optional
        Target directory. Defaults to `models/`.

    Returns
    -------
    Path
        Path to persisted Joblib artifact.
    """
    if output_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root / "models"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "best_clustering_model.joblib"

    artifact = {
        "algorithm_name": algorithm_name,
        "model": model,
        "scaler": scaler,
        "n_clusters": n_clusters,
        "metrics": metrics,
        "cluster_profiles": cluster_profiles,
    }
    joblib.dump(artifact, target_path)
    return target_path


def load_clustering_artifact(model_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Load serialized clustering artifact and persona profiles from disk.

    Parameters
    ----------
    model_path : Path or str, optional
        Path to artifact. Defaults to `models/best_clustering_model.joblib`.

    Returns
    -------
    Dict[str, Any]
        Loaded artifact dictionary.
    """
    if model_path is None:
        project_root = Path(__file__).resolve().parent.parent
        model_path = project_root / "models" / "best_clustering_model.joblib"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Clustering model artifact not found at: {model_path.resolve()}")

    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or "model" not in artifact or "scaler" not in artifact:
        raise ValueError("Loaded artifact is corrupted or does not contain a valid clustering model.")

    return artifact
