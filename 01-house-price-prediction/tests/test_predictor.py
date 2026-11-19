"""
Unit tests for HousePricePredictor inference interface.
"""

from __future__ import annotations

import pandas as pd
import pytest
from src.predictor import HousePricePredictor


@pytest.fixture
def predictor() -> HousePricePredictor:
    return HousePricePredictor()


def test_predictor_initialization(predictor: HousePricePredictor):
    assert predictor.pipeline is not None
    assert isinstance(predictor.model_name, str)
    assert "r2_score" in predictor.metrics


def test_predictor_single_property(predictor: HousePricePredictor):
    sample = {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 2000.0,
        "sqft_lot": 5500.0,
        "floors": 1.0,
        "waterfront": 0,
        "view_quality": 1,
        "condition": 3,
        "grade": 7,
        "yr_built": 1995,
        "yr_renovated": 0,
        "neighborhood": "Suburbs",
    }
    val = predictor.predict_single(sample)
    assert isinstance(val, dict)
    assert val["estimated_price"] > 100_000.0
    assert val["estimated_range_lower"] < val["estimated_price"]
    assert val["estimated_range_upper"] > val["estimated_price"]
    assert val["model_r2_score"] > 0.90


def test_predictor_batch_prediction(predictor: HousePricePredictor):
    batch = [
        {
            "bedrooms": 2,
            "bathrooms": 1.5,
            "sqft_living": 1200.0,
            "sqft_lot": 3000.0,
            "floors": 1.0,
            "waterfront": 0,
            "view_quality": 0,
            "condition": 3,
            "grade": 6,
            "yr_built": 1980,
            "yr_renovated": 0,
            "neighborhood": "Greenwood",
        },
        {
            "bedrooms": 5,
            "bathrooms": 3.5,
            "sqft_living": 3800.0,
            "sqft_lot": 12000.0,
            "floors": 2.0,
            "waterfront": 1,
            "view_quality": 4,
            "condition": 5,
            "grade": 11,
            "yr_built": 2015,
            "yr_renovated": 2020,
            "neighborhood": "Lakeside",
        },
    ]
    preds = predictor.predict(batch)
    assert len(preds) == 2
    assert preds[1] > preds[0]  # Luxury waterfront home should be priced significantly higher


def test_predictor_invalid_type(predictor: HousePricePredictor):
    with pytest.raises(TypeError):
        predictor.predict("invalid_input_string")
