"""
Production Inference and Fraud Risk Scoring Engine.

Loads persisted credit card fraud pipeline and generates transaction fraud probabilities,
decision actions (Approve, Challenge with 2FA, Decline), and risk tier profiling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .model_training import load_saved_fraud_model


FEATURE_NAMES = [f"V{i}" for i in range(1, 29)]
ALL_COLUMNS = ["Time", "Amount"] + FEATURE_NAMES


class FraudPredictor:
    """Production transaction scoring and fraud decisioning engine."""

    def __init__(self, model_path: Optional[Path | str] = None) -> None:
        """Initialize scoring engine with serialized model artifact."""
        artifact = load_saved_fraud_model(model_path)
        self.model_name: str = artifact["model_name"]
        self.pipeline: Pipeline = artifact["pipeline"]
        self.metrics: Dict[str, float] = artifact["metrics"]

    def _prepare_inputs(
        self,
        transaction: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    ) -> pd.DataFrame:
        """Sanitize raw transaction input into standard DataFrame matching training schema."""
        if isinstance(transaction, dict):
            df = pd.DataFrame([transaction])
        elif isinstance(transaction, list):
            df = pd.DataFrame(transaction)
        elif isinstance(transaction, pd.DataFrame):
            df = transaction.copy()
        else:
            raise TypeError("Transaction must be a dict, list of dicts, or pandas DataFrame.")

        # Ensure all columns exist and replace any missing/NaN values with 0.0
        df = df.reindex(columns=ALL_COLUMNS, fill_value=0.0).fillna(0.0)

        return df

    def predict_proba(
        self,
        transaction: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    ) -> np.ndarray:
        """Compute continuous fraud probabilities (between 0.0 and 1.0).

        Parameters
        ----------
        transaction : dict, list of dicts, or DataFrame
            Transaction attributes.

        Returns
        -------
        np.ndarray
            Array of predicted fraud probabilities.
        """
        df = self._prepare_inputs(transaction)
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(df)[:, 1]
        elif hasattr(self.pipeline, "decision_function"):
            df_vals = self.pipeline.decision_function(df)
            probs = 1.0 / (1.0 + np.exp(-df_vals))
        else:
            probs = self.pipeline.predict(df).astype(float)
        return probs

    def predict(
        self,
        transaction: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
        threshold: float = 0.50,
    ) -> np.ndarray:
        """Generate binary fraud predictions (0 = Legit, 1 = Fraud).

        Parameters
        ----------
        transaction : dict, list of dicts, or DataFrame
            Transaction attributes.
        threshold : float, default=0.50
            Classification decision threshold.

        Returns
        -------
        np.ndarray
            Binary predictions (0 or 1).
        """
        probs = self.predict_proba(transaction)
        return (probs >= threshold).astype(int)

    def score_transaction(
        self,
        transaction_record: Dict[str, Any],
        threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """Generate comprehensive fraud risk profile and decision playbook for a single transaction.

        Parameters
        ----------
        transaction_record : Dict[str, Any]
            Transaction features.
        threshold : float, default=0.50
            Decision threshold.

        Returns
        -------
        Dict[str, Any]
            Fraud probability, verdict, risk tier, action playbook, and influential features.
        """
        prob = float(self.predict_proba(transaction_record)[0])
        is_fraud = bool(prob >= threshold)
        amount = float(transaction_record.get("Amount", 0.0))

        # Decision Tiers
        if prob >= 0.70:
            risk_tier = "High Risk"
            decision = "DECLINE & FREEZE"
            action = "Immediate Action: Block payment authorization, send fraud SMS alert, and place temporary hold on card."
        elif prob >= threshold:
            risk_tier = "Moderate Risk"
            decision = "STEP-UP 2FA CHALLENGE"
            action = "Authentication Challenge: Prompt cardholder for biometrics or One-Time Passcode (OTP) before completing authorization."
        elif prob >= 0.15:
            risk_tier = "Elevated Caution"
            decision = "APPROVE WITH PASSIVE MONITORING"
            action = "Approve transaction; log velocity telemetry and queue for post-settlement behavioral surveillance."
        else:
            risk_tier = "Low Risk"
            decision = "INSTANT APPROVAL"
            action = "Approve: Frictionless checkout authorized within normal limits."

        # Detect anomalous feature signals
        flagged_signals = []
        if float(transaction_record.get("V14", 0.0)) < -3.0:
            flagged_signals.append("Severe negative shift in V14 (high fraud signature).")
        if float(transaction_record.get("V12", 0.0)) < -3.0:
            flagged_signals.append("Depressed V12 component indicating anomalous device/geo mismatch.")
        if float(transaction_record.get("V4", 0.0)) > 2.5:
            flagged_signals.append("Elevated V4 component signaling card-not-present velocity.")
        if amount > 1000.0:
            flagged_signals.append(f"High transaction value (${amount:,.2f}) above 95th percentile.")

        return {
            "fraud_probability": prob,
            "fraud_probability_pct": f"{prob * 100.0:.2f}%",
            "is_fraud": is_fraud,
            "decision": decision,
            "risk_tier": risk_tier,
            "recommended_action": action,
            "flagged_signals": flagged_signals,
            "model_used": self.model_name,
            "model_pr_auc": self.metrics.get("pr_auc", 0.0),
            "model_roc_auc": self.metrics.get("roc_auc", 0.0),
        }
