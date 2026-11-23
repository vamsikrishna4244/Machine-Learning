"""
Production Inference and Customer Segmentation Scoring Engine.

Loads persisted clustering artifacts and assigns new customer profiles to clusters,
computing distance to centroids, persona descriptions, and personalized marketing playbooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .cluster_models import load_clustering_artifact
from .data_loader import CUSTOMER_FEATURES


class CustomerSegmenter:
    """Production customer segmentation engine for retail persona profiling."""

    def __init__(self, model_path: Optional[Path | str] = None) -> None:
        """Initialize segmentation engine with serialized clustering artifact."""
        artifact = load_clustering_artifact(model_path)
        self.algorithm_name: str = artifact["algorithm_name"]
        self.model: KMeans = artifact["model"]
        self.scaler: StandardScaler = artifact["scaler"]
        self.n_clusters: int = artifact["n_clusters"]
        self.metrics: Dict[str, float] = artifact["metrics"]
        self.profiles: Dict[int, Dict[str, Any]] = artifact["cluster_profiles"]

    def _prepare_inputs(
        self,
        customer: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    ) -> pd.DataFrame:
        """Sanitize customer inputs and ensure all standard features exist."""
        if isinstance(customer, dict):
            df = pd.DataFrame([customer])
        elif isinstance(customer, list):
            df = pd.DataFrame(customer)
        elif isinstance(customer, pd.DataFrame):
            df = customer.copy()
        else:
            raise TypeError("Customer data must be a dict, list of dicts, or pandas DataFrame.")

        # Reindex features, filling missing attributes with column medians from profiles
        for col in CUSTOMER_FEATURES:
            if col not in df.columns:
                # Default to overall median
                df[col] = float(np.mean([p["mean_attributes"][col] for p in self.profiles.values()]))

        return df[CUSTOMER_FEATURES].fillna(0.0)

    def segment_customers(
        self,
        customer_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    ) -> np.ndarray:
        """Assign cluster IDs for one or more customer records.

        Parameters
        ----------
        customer_data : dict, list of dicts, or DataFrame
            Customer attributes.

        Returns
        -------
        np.ndarray
            Assigned cluster IDs.
        """
        df = self._prepare_inputs(customer_data)
        X_scaled = self.scaler.transform(df.to_numpy(dtype=float))
        return self.model.predict(X_scaled)

    def score_single_customer(
        self,
        customer_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate comprehensive customer persona profiling and marketing recommendations.

        Parameters
        ----------
        customer_record : Dict[str, Any]
            Customer attribute dictionary.

        Returns
        -------
        Dict[str, Any]
            Cluster assignment, persona name, distance to centroid, description, and marketing playbook.
        """
        df = self._prepare_inputs(customer_record)
        X_scaled = self.scaler.transform(df.to_numpy(dtype=float))
        cluster_id = int(self.model.predict(X_scaled)[0])

        # Distance to centroid (measure of how archetypal the customer is)
        centroid = self.model.cluster_centers_[cluster_id]
        dist_to_centroid = float(np.linalg.norm(X_scaled[0] - centroid))

        profile_info = self.profiles.get(cluster_id, {})
        persona = profile_info.get("persona_name", f"Segment {cluster_id}")
        desc = profile_info.get("description", "Standard retail shopper.")
        strategy = profile_info.get("marketing_strategy", "Standard seasonal engagement.")

        return {
            "cluster_id": cluster_id,
            "persona_name": persona,
            "prototypical_distance": round(dist_to_centroid, 3),
            "customer_share_pct": profile_info.get("percentage", 20.0),
            "description": desc,
            "marketing_strategy": strategy,
            "algorithm_used": self.algorithm_name,
            "silhouette_score": self.metrics.get("silhouette", 0.0),
        }
