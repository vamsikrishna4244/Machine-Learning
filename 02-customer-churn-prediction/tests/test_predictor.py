"""
Unit Tests for Churn Predictor and Inference Engine.
"""

from pathlib import Path
import pytest
import pandas as pd
from src.predictor import ChurnPredictor


@pytest.fixture
def sample_customer_dict():
    """Fixture for single customer input."""
    return {
        "gender": "Female",
        "senior_citizen": "No",
        "partner": "No",
        "dependents": "No",
        "tenure_months": 3,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 85.0,
        "total_charges": 255.0,
    }


def test_predictor_score_customer(sample_customer_dict):
    """Verify ChurnPredictor outputs probability, risk tier, and retention recommendation."""
    predictor = ChurnPredictor()
    score = predictor.score_customer(sample_customer_dict, threshold=0.5)

    assert "churn_probability" in score
    assert "risk_tier" in score
    assert "predicted_churn" in score
    assert "recommended_action" in score
    assert 0.0 <= score["churn_probability"] <= 1.0
    assert score["risk_tier"] in ["High Risk", "Medium Risk", "Low Risk"]


def test_predictor_threshold_tuning(sample_customer_dict):
    """Verify threshold modulation alters binary prediction."""
    predictor = ChurnPredictor()
    pred_low_thresh = predictor.predict(sample_customer_dict, threshold=0.01)
    pred_high_thresh = predictor.predict(sample_customer_dict, threshold=0.99)

    assert pred_low_thresh[0] == 1
    assert pred_high_thresh[0] == 0


def test_predictor_batch_dataframe(sample_customer_dict):
    """Verify batch inference over a pandas DataFrame."""
    predictor = ChurnPredictor()
    df = pd.DataFrame([sample_customer_dict, sample_customer_dict])
    probs = predictor.predict_proba(df)

    assert len(probs) == 2
    assert probs[0] == probs[1]


def test_predictor_invalid_type_raises_error():
    """Verify TypeError when invalid data format is provided."""
    predictor = ChurnPredictor()
    with pytest.raises(TypeError):
        predictor.predict("invalid_string_input")  # type: ignore


def test_predictor_list_of_dicts_and_auto_total_charges(sample_customer_dict):
    """Verify inference with a list of dicts and omitted total_charges."""
    predictor = ChurnPredictor()
    cust_copy = sample_customer_dict.copy()
    del cust_copy["total_charges"]  # should auto compute total_charges

    probs = predictor.predict_proba([cust_copy, cust_copy])
    assert len(probs) == 2
    assert 0.0 <= probs[0] <= 1.0


def test_predictor_low_risk_tier():
    """Verify scoring a low-risk subscriber triggers Low Risk tier."""
    predictor = ChurnPredictor()
    loyal_cust = {
        "gender": "Male",
        "senior_citizen": "No",
        "partner": "Yes",
        "dependents": "Yes",
        "tenure_months": 70,
        "phone_service": "Yes",
        "multiple_lines": "Yes",
        "internet_service": "DSL",
        "online_security": "Yes",
        "online_backup": "Yes",
        "device_protection": "Yes",
        "tech_support": "Yes",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Two year",
        "paperless_billing": "No",
        "payment_method": "Credit card (automatic)",
        "monthly_charges": 25.0,
        "total_charges": 1750.0,
    }
    score = predictor.score_customer(loyal_cust)
    assert score["risk_tier"] == "Low Risk"
    assert "Standard Operations" in score["recommended_action"]
