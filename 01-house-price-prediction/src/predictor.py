"""
Production Inference and House Price Prediction Interface.

Provides a robust, validated prediction wrapper around the persisted model artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .model_training import load_saved_model


class HousePricePredictor:
    """Production valuation engine for real estate pricing."""

    def __init__(self, model_path: Optional[Path | str] = None) -> None:
        """Initialize predictor by loading the trained model artifact."""
        artifact = load_saved_model(model_path)
        self.model_name: str = artifact["model_name"]
        self.pipeline: Pipeline = artifact["pipeline"]
        self.metrics: Dict[str, float] = artifact["metrics"]

    def _prepare_input_dataframe(
        self,
        features: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    ) -> pd.DataFrame:
        """Sanitize and format raw property input records."""
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        elif isinstance(features, list):
            df = pd.DataFrame(features)
        elif isinstance(features, pd.DataFrame):
            df = features.copy()
        else:
            raise TypeError("Features must be a dictionary, list of dictionaries, or DataFrame.")

        # Compute derived features if not present
        if "yr_built" in df.columns and "house_age" not in df.columns:
            df["house_age"] = (2024 - df["yr_built"]).clip(lower=0)

        if "yr_renovated" in df.columns:
            df["yr_renovated"] = df["yr_renovated"].fillna(0.0)
            if "is_renovated" not in df.columns:
                df["is_renovated"] = (df["yr_renovated"] > 0).astype(int)
        else:
            if "is_renovated" not in df.columns:
                df["is_renovated"] = 0

        if "sqft_living" in df.columns and "bedrooms" in df.columns:
            if "sqft_per_bedroom" not in df.columns:
                df["sqft_per_bedroom"] = (
                    df["sqft_living"] / df["bedrooms"].clip(lower=1)
                ).round(2)
            if "bath_to_bed_ratio" not in df.columns and "bathrooms" in df.columns:
                df["bath_to_bed_ratio"] = (
                    df["bathrooms"] / df["bedrooms"].clip(lower=1)
                ).round(2)

        return df

    def predict(
        self,
        features: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    ) -> np.ndarray:
        """Generate point price predictions for one or more properties.

        Parameters
        ----------
        features : dict, list of dicts, or DataFrame
            Property attributes.

        Returns
        -------
        np.ndarray
            Predicted house prices in USD.
        """
        df = self._prepare_input_dataframe(features)
        predictions = self.pipeline.predict(df)
        return np.round(predictions, -2)

    def predict_single(self, property_features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed valuation summary for an individual residential property.

        Parameters
        ----------
        property_features : Dict[str, Any]
            Property characteristics.

        Returns
        -------
        Dict[str, Any]
            Valuation estimate, expected error margin, and model metadata.
        """
        predicted_price = float(self.predict(property_features)[0])
        rmse_margin = self.metrics.get("rmse", 30_000.0)

        return {
            "estimated_price": predicted_price,
            "estimated_range_lower": max(50_000.0, round(predicted_price - rmse_margin, -2)),
            "estimated_range_upper": round(predicted_price + rmse_margin, -2),
            "model_used": self.model_name,
            "model_r2_score": self.metrics.get("r2_score", 0.0),
            "model_mae": self.metrics.get("mae", 0.0),
        }
