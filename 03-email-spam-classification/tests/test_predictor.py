"""
Unit Tests for Spam Predictor and Inference Engine.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from src.predictor import SpamPredictor


def test_predictor_score_email():
    """Verify scoring of legitimate and spam email strings."""
    predictor = SpamPredictor()

    ham_msg = "Hi team, let us meet at 2 PM to review the updated sprint architecture."
    res_ham = predictor.score_email(ham_msg)
    assert res_ham["classification"] == "HAM"
    assert "spam_probability" in res_ham
    assert "confidence_tier" in res_ham
    assert "recommended_action" in res_ham

    spam_msg = "Congratulations winner! Claim your $5,000,000 cash prize lottery now at http://scam.xyz"
    res_spam = predictor.score_email(spam_msg)
    assert res_spam["classification"] == "SPAM"
    assert res_spam["spam_probability"] > 0.50
    assert len(res_spam["trigger_keywords"]) > 0


def test_predictor_predict_threshold():
    """Verify threshold variation alters binary prediction output."""
    predictor = SpamPredictor()
    ham_msg = "Weekly team sprint review sync and status notes."

    # At threshold 0.0, everything is positive (spam)
    pred_zero = predictor.predict(ham_msg, threshold=0.0)
    # At standard threshold, ham is 0
    pred_normal = predictor.predict(ham_msg, threshold=0.50)

    assert pred_zero[0] == 1
    assert pred_normal[0] == 0


def test_predictor_list_and_batch_series():
    """Verify scoring a list of strings and a pandas Series."""
    predictor = SpamPredictor()
    msg_list = [
        "Meeting reschedule update",
        "Urgent claim free lottery reward money",
    ]
    probs_list = predictor.predict_proba(msg_list)
    assert len(probs_list) == 2

    series = pd.Series(msg_list)
    probs_series = predictor.predict_proba(series)
    assert len(probs_series) == 2


def test_predictor_invalid_input_type():
    """Verify TypeError when invalid format is passed."""
    predictor = SpamPredictor()
    with pytest.raises(TypeError):
        predictor.predict(12345)  # type: ignore


def test_predictor_confidence_tiers(monkeypatch):
    """Verify all confidence tiers in score_email."""
    predictor = SpamPredictor()

    # Moderate Confidence Spam (prob between 0.50 and 0.80)
    monkeypatch.setattr(predictor, "predict_proba", lambda msg: [0.65])
    res_mod = predictor.score_email("Sample test message", threshold=0.50)
    assert res_mod["confidence_tier"] == "Moderate Confidence Spam"

    # Likely Legitimate (prob between 0.20 and 0.50)
    monkeypatch.setattr(predictor, "predict_proba", lambda msg: [0.35])
    res_legit = predictor.score_email("Sample test message", threshold=0.50)
    assert res_legit["confidence_tier"] == "Likely Legitimate"


def test_predictor_decision_function_fallback(monkeypatch):
    """Verify probability computation when pipeline uses decision_function."""
    predictor = SpamPredictor()

    class MockPipeline:
        def __init__(self):
            self.named_steps = {}
        def decision_function(self, X):
            return np.array([2.5, -2.5])

    monkeypatch.setattr(predictor, "pipeline", MockPipeline())
    probs = predictor.predict_proba(["spam message", "ham message"])
    assert len(probs) == 2
    assert probs[0] > 0.5
    assert probs[1] < 0.5
