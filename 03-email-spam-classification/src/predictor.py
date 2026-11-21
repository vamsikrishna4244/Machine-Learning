"""
Production Inference and Email Spam Scoring Engine.

Loads persisted spam classification pipeline and classifies raw email strings,
evaluates spam risk probabilities, and extracts triggering spam tokens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .model_training import load_saved_spam_pipeline
from .text_preprocessing import clean_email_text


class SpamPredictor:
    """Production email spam classification and risk evaluation engine."""

    def __init__(self, model_path: Optional[Path | str] = None) -> None:
        """Initialize scoring engine with serialized model artifact."""
        artifact = load_saved_spam_pipeline(model_path)
        self.model_name: str = artifact["model_name"]
        self.pipeline: Pipeline = artifact["pipeline"]
        self.metrics: Dict[str, float] = artifact["metrics"]

    def _preprocess_inputs(
        self,
        messages: Union[str, List[str], pd.Series],
    ) -> List[str]:
        """Sanitize raw email messages into cleaned tokens."""
        if isinstance(messages, str):
            raw_list = [messages]
        elif isinstance(messages, list):
            raw_list = messages
        elif isinstance(messages, pd.Series):
            raw_list = messages.tolist()
        else:
            raise TypeError("Messages must be a string, list of strings, or pandas Series.")

        return [clean_email_text(msg) for msg in raw_list]

    def predict_proba(
        self,
        messages: Union[str, List[str], pd.Series],
    ) -> np.ndarray:
        """Calculate continuous spam probabilities (between 0.0 and 1.0).

        Parameters
        ----------
        messages : str, list of str, or Series
            Raw email text messages.

        Returns
        -------
        np.ndarray
            Array of spam probabilities.
        """
        cleaned = self._preprocess_inputs(messages)
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(cleaned)[:, 1]
        elif hasattr(self.pipeline, "decision_function"):
            df_vals = self.pipeline.decision_function(cleaned)
            probs = 1.0 / (1.0 + np.exp(-df_vals))
        else:
            preds = self.pipeline.predict(cleaned)
            probs = preds.astype(float)
        return probs

    def predict(
        self,
        messages: Union[str, List[str], pd.Series],
        threshold: float = 0.50,
    ) -> np.ndarray:
        """Predict binary classification labels (0 = Ham, 1 = Spam).

        Parameters
        ----------
        messages : str, list of str, or Series
            Raw email text messages.
        threshold : float, default=0.50
            Classification decision threshold.

        Returns
        -------
        np.ndarray
            Binary predictions (0 or 1).
        """
        probs = self.predict_proba(messages)
        return (probs >= threshold).astype(int)

    def extract_trigger_keywords(self, raw_message: str, top_k: int = 5) -> List[str]:
        """Detect vocabulary keywords present in the message that correlate with spam.

        Parameters
        ----------
        raw_message : str
            Email message text.
        top_k : int, default=5
            Maximum trigger tokens to return.

        Returns
        -------
        List[str]
            Identified trigger keywords.
        """
        cleaned = clean_email_text(raw_message)
        tokens = set(cleaned.split())
        vec = self.pipeline.named_steps.get("vectorizer")
        clf = self.pipeline.named_steps.get("classifier")

        if vec is not None and clf is not None:
            vocab = vec.vocabulary_
            if hasattr(clf, "coef_"):
                weights = clf.coef_[0]
            elif hasattr(clf, "feature_log_prob_"):
                weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
            else:
                weights = None

            if weights is not None:
                matched = []
                for token in tokens:
                    if token in vocab:
                        idx = vocab[token]
                        w = weights[idx]
                        if w > 0:  # Positively correlates with spam
                            matched.append((token, w))
                matched.sort(key=lambda x: x[1], reverse=True)
                return [tok for tok, _ in matched[:top_k]]

        return []

    def score_email(
        self,
        raw_message: str,
        threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """Generate a complete spam classification assessment for an incoming email.

        Parameters
        ----------
        raw_message : str
            Raw email text string.
        threshold : float, default=0.50
            Decision threshold.

        Returns
        -------
        Dict[str, Any]
            Spam probability, verdict, confidence tier, and detected trigger keywords.
        """
        prob = float(self.predict_proba(raw_message)[0])
        is_spam = bool(prob >= threshold)
        triggers = self.extract_trigger_keywords(raw_message)

        if prob >= 0.80:
            confidence_tier = "High Confidence Spam"
            recommendation = "Quarantine: Route directly to Spam folder and block sender domain."
        elif prob >= threshold:
            confidence_tier = "Moderate Confidence Spam"
            recommendation = "Warning Banner: Deliver to inbox with [SUSPICIOUS] subject tag and disabled hyperlinks."
        elif prob >= 0.20:
            confidence_tier = "Likely Legitimate"
            recommendation = "Deliver: Standard inbox delivery with background SPF/DKIM authentication."
        else:
            confidence_tier = "High Confidence Ham"
            recommendation = "Safe: Deliver directly to primary inbox."

        return {
            "spam_probability": prob,
            "spam_probability_pct": f"{prob * 100.0:.1f}%",
            "is_spam": is_spam,
            "classification": "SPAM" if is_spam else "HAM",
            "confidence_tier": confidence_tier,
            "recommended_action": recommendation,
            "trigger_keywords": triggers,
            "model_used": self.model_name,
            "model_f1_score": self.metrics.get("f1_score", 0.0),
        }
