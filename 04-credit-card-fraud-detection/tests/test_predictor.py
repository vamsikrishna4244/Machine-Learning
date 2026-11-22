"""
Unit Tests for Fraud Predictor and Transaction Scoring Engine.
"""

from pathlib import Path
import pytest
import pandas as pd
from src.predictor import FraudPredictor


def test_score_transaction_legit_and_fraud():
    """Verify transaction scoring across decision tiers and action playbooks."""
    predictor = FraudPredictor()

    # Normal coffee purchase
    legit_txn = {"Time": 25000.0, "Amount": 4.50, "V14": 0.2, "V12": 0.1, "V4": 0.0}
    score_legit = predictor.score_transaction(legit_txn)
    assert score_legit["decision"] == "INSTANT APPROVAL"
    assert score_legit["risk_tier"] == "Low Risk"
    assert not score_legit["is_fraud"]

    # Fraudulent drain
    fraud_txn = {"Time": 70000.0, "Amount": 1500.00, "V14": -6.5, "V12": -5.0, "V4": 4.5}
    score_fraud = predictor.score_transaction(fraud_txn)
    assert score_fraud["decision"] == "DECLINE & FREEZE"
    assert score_fraud["risk_tier"] == "High Risk"
    assert score_fraud["is_fraud"]
    assert len(score_fraud["flagged_signals"]) > 0


def test_predict_threshold_tuning():
    """Verify threshold variation alters binary fraud prediction output."""
    predictor = FraudPredictor()
    txn = {"Amount": 200.00, "V14": -2.0}

    pred_zero = predictor.predict(txn, threshold=0.0)
    pred_max = predictor.predict(txn, threshold=1.0)
    assert pred_zero[0] == 1
    assert pred_max[0] == 0


def test_predict_batch_dataframe():
    """Verify batch inference over pandas DataFrame and list of dicts."""
    predictor = FraudPredictor()
    txns = [
        {"Time": 1000.0, "Amount": 25.0},
        {"Time": 2000.0, "Amount": 1250.0, "V14": -5.0},
    ]

    probs_list = predictor.predict_proba(txns)
    assert len(probs_list) == 2

    df = pd.DataFrame(txns)
    probs_df = predictor.predict_proba(df)
    assert len(probs_df) == 2


def test_predict_invalid_type_raises_error():
    """Verify TypeError when invalid format is provided."""
    predictor = FraudPredictor()
    with pytest.raises(TypeError):
        predictor.predict("invalid_string_input")  # type: ignore


def test_score_transaction_decision_tiers(monkeypatch):
    """Verify moderate risk and elevated caution branches."""
    predictor = FraudPredictor()

    # Step-up 2FA Challenge (prob between threshold and 0.70)
    monkeypatch.setattr(predictor, "predict_proba", lambda t: [0.60])
    res_2fa = predictor.score_transaction({"Amount": 100.0}, threshold=0.50)
    assert "2FA" in res_2fa["decision"]

    # Elevated Caution (prob between 0.15 and threshold)
    monkeypatch.setattr(predictor, "predict_proba", lambda t: [0.25])
    res_caution = predictor.score_transaction({"Amount": 100.0}, threshold=0.50)
    assert "PASSIVE MONITORING" in res_caution["decision"]
