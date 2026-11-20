"""
Production Inference and Churn Risk Scoring Engine.

Loads persisted model pipeline and generates churn probabilities, risk tiers,
and retention intervention recommendations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .model_training import load_saved_churn_model


class ChurnPredictor:
    """Production churn risk scoring and customer retention decision engine."""

    def __init__(self, model_path: Optional[Path | str] = None) -> None:
        """Initialize scoring engine with serialized model artifact."""
        artifact = load_saved_churn_model(model_path)
        self.model_name: str = artifact["model_name"]
        self.pipeline: Pipeline = artifact["pipeline"]
        self.metrics: Dict[str, float] = artifact["metrics"]

    def _prepare_input_dataframe(
        self,
        features: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    ) -> pd.DataFrame:
        """Sanitize raw customer input dictionaries and compute derived features."""
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        elif isinstance(features, list):
            df = pd.DataFrame(features)
        elif isinstance(features, pd.DataFrame):
            df = features.copy()
        else:
            raise TypeError("Features must be a dictionary, list of dicts, or pandas DataFrame.")

        # Compute derived features if omitted
        if "tenure_months" in df.columns:
            if "tenure_years" not in df.columns:
                df["tenure_years"] = np.round(df["tenure_months"] / 12.0, 2)
            if "total_charges" in df.columns and "monthly_charges" in df.columns:
                if "monthly_to_total_ratio" not in df.columns:
                    df["monthly_to_total_ratio"] = np.round(
                        df["monthly_charges"] / (df["total_charges"] + 1.0), 4
                    )
            elif "monthly_charges" in df.columns:
                df["total_charges"] = df["monthly_charges"] * df["tenure_months"]
                df["monthly_to_total_ratio"] = np.round(
                    df["monthly_charges"] / (df["total_charges"] + 1.0), 4
                )

        if "services_count" not in df.columns:
            service_cols = [
                "phone_service",
                "multiple_lines",
                "online_security",
                "online_backup",
                "device_protection",
                "tech_support",
                "streaming_tv",
                "streaming_movies",
            ]
            df["services_count"] = 0
            for col in service_cols:
                if col in df.columns:
                    df["services_count"] += (df[col] == "Yes").astype(int)

        if "has_tech_and_security" not in df.columns:
            has_sec = df.get("online_security", pd.Series(["No"] * len(df))) == "Yes"
            has_tech = df.get("tech_support", pd.Series(["No"] * len(df))) == "Yes"
            df["has_tech_and_security"] = np.where(has_sec & has_tech, "Yes", "No")

        return df

    def predict_proba(
        self,
        features: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    ) -> np.ndarray:
        """Predict continuous churn probability (between 0.0 and 1.0).

        Parameters
        ----------
        features : dict, list of dicts, or DataFrame
            Customer attributes.

        Returns
        -------
        np.ndarray
            Array of churn probabilities.
        """
        df = self._prepare_input_dataframe(features)
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(df)[:, 1]
        else:
            probs = self.pipeline.predict(df).astype(float)
        return np.round(probs, 4)

    def predict(
        self,
        features: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Predict binary churn classification under a customizable probability threshold.

        Parameters
        ----------
        features : dict, list of dicts, or DataFrame
            Customer features.
        threshold : float, default=0.5
            Decision threshold (lower thresholds prioritize high recall for retention).

        Returns
        -------
        np.ndarray
            Binary predictions (0 = Retained, 1 = Churned).
        """
        probs = self.predict_proba(features)
        return (probs >= threshold).astype(int)

    def score_customer(
        self,
        customer_record: Dict[str, Any],
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Generate comprehensive churn risk profile and retention recommendations for a customer.

        Parameters
        ----------
        customer_record : Dict[str, Any]
            Single customer attributes dictionary.
        threshold : float, default=0.5
            Decision threshold.

        Returns
        -------
        Dict[str, Any]
            Churn probability, risk tier, action recommendation, and model metrics.
        """
        prob = float(self.predict_proba(customer_record)[0])
        is_churn = bool(prob >= threshold)

        if prob >= 0.65:
            risk_tier = "High Risk"
            recommendation = "Immediate Intervention: Proactive outreach with 1-year contract discount and free Tech Support bundle."
        elif prob >= 0.35:
            risk_tier = "Medium Risk"
            recommendation = "Nurture Campaign: Offer self-service optimization, device protection review, and automated billing incentives."
        else:
            risk_tier = "Low Risk"
            recommendation = "Standard Operations: Maintain engagement via loyalty rewards and cross-sell streaming addons."

        return {
            "churn_probability": prob,
            "churn_probability_pct": f"{prob * 100.0:.1f}%",
            "predicted_churn": is_churn,
            "risk_tier": risk_tier,
            "recommended_action": recommendation,
            "model_used": self.model_name,
            "model_roc_auc": self.metrics.get("roc_auc", 0.0),
            "model_f1_score": self.metrics.get("f1_score", 0.0),
        }
