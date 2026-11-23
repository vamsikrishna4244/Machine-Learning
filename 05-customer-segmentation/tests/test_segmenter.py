"""
Unit Tests for Customer Segmenter and Production Inference Engine.
"""

from pathlib import Path
import pytest
import pandas as pd
from src.segmenter import CustomerSegmenter


def test_score_single_customer():
    """Verify customer scoring outputs cluster assignment, persona, and marketing strategy."""
    segmenter = CustomerSegmenter()

    vip_cust = {
        "Age": 38,
        "Annual_Income_k": 95.0,
        "Spending_Score": 85.0,
        "Purchase_Frequency": 32.0,
        "Average_Basket_Value": 210.0,
    }
    score = segmenter.score_single_customer(vip_cust)
    assert "cluster_id" in score
    assert "persona_name" in score
    assert "prototypical_distance" in score
    assert "marketing_strategy" in score
    assert score["cluster_id"] in range(segmenter.n_clusters)
    assert score["prototypical_distance"] >= 0.0


def test_segment_customers_batch():
    """Verify batch segmentation over a DataFrame and list of dicts."""
    segmenter = CustomerSegmenter()
    customers = [
        {"Age": 25, "Annual_Income_k": 30.0, "Spending_Score": 80.0},
        {"Age": 55, "Annual_Income_k": 90.0, "Spending_Score": 20.0},
    ]

    preds_list = segmenter.segment_customers(customers)
    assert len(preds_list) == 2

    df = pd.DataFrame(customers)
    preds_df = segmenter.segment_customers(df)
    assert len(preds_df) == 2
    assert preds_list[0] == preds_df[0]


def test_segmenter_invalid_type_raises_error():
    """Verify TypeError when invalid data type is passed."""
    segmenter = CustomerSegmenter()
    with pytest.raises(TypeError):
        segmenter.segment_customers("invalid_string_input")  # type: ignore
