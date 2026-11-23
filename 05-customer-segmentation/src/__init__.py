"""
Customer Segmentation Using Unsupervised Clustering Package.

Provides K-Means, Hierarchical/Agglomerative Clustering, and DBSCAN algorithms,
along with Elbow & Silhouette evaluation, PCA projections, and customer persona profiling.
"""

from .data_loader import generate_customer_data, load_customer_data
from .data_preprocessing import preprocess_customer_data, build_scaling_pipeline, apply_pca_reduction
from .cluster_models import (
    train_kmeans,
    train_hierarchical,
    train_dbscan,
    train_all_clustering_models,
    save_clustering_artifact,
    load_clustering_artifact,
)
from .cluster_evaluation import (
    evaluate_clustering_metrics,
    compute_elbow_and_silhouette_sweep,
    profile_clusters,
    plot_clustering_diagnostics,
)
from .segmenter import CustomerSegmenter

__all__ = [
    "generate_customer_data",
    "load_customer_data",
    "preprocess_customer_data",
    "build_scaling_pipeline",
    "apply_pca_reduction",
    "train_kmeans",
    "train_hierarchical",
    "train_dbscan",
    "train_all_clustering_models",
    "save_clustering_artifact",
    "load_clustering_artifact",
    "evaluate_clustering_metrics",
    "compute_elbow_and_silhouette_sweep",
    "profile_clusters",
    "plot_clustering_diagnostics",
    "CustomerSegmenter",
]
