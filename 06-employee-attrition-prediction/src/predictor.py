"""
Production Inference and Employee Flight Risk Decision Engine.

Generates individual employee attrition probabilities, risk tiers, key flight risk drivers,
and tailored HR talent retention playbooks while enforcing responsible AI compliance guidelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .model_training import load_attrition_pipeline
from .data_preprocessing import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


ALL_MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


class AttritionPredictor:
    """Enterprise employee attrition scoring and retention decisioning engine."""

    ETHICAL_DISCLAIMER: str = (
        "COMPLIANCE NOTICE: This score is an advisory predictive signal intended exclusively "
        "for proactive employee engagement, wellness support, and talent retention. It must NEVER "
        "be used as an automated decision-maker for termination, compensation cuts, or adverse employment actions."
    )

    def __init__(self, model_path: Optional[Path | str] = None) -> None:
        """Initialize scoring engine with serialized model artifact."""
        artifact = load_attrition_pipeline(model_path)
        self.model_name: str = artifact["model_name"]
        self.pipeline: Pipeline = artifact["pipeline"]
        self.metrics: Dict[str, float] = artifact["metrics"]
        self.best_params: Dict[str, Any] = artifact.get("best_params", {})

    def _prepare_inputs(
        self,
        employee_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    ) -> pd.DataFrame:
        """Sanitize raw employee inputs and align feature columns."""
        if isinstance(employee_data, dict):
            df = pd.DataFrame([employee_data])
        elif isinstance(employee_data, list):
            df = pd.DataFrame(employee_data)
        elif isinstance(employee_data, pd.DataFrame):
            df = employee_data.copy()
        else:
            raise TypeError("Employee data must be a dict, list of dicts, or pandas DataFrame.")

        # Ensure all required features exist with sensible default values
        for col in NUMERICAL_FEATURES:
            if col not in df.columns:
                df[col] = 0.0
        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                df[col] = "Missing"

        return df[ALL_MODEL_FEATURES]

    def predict_proba(
        self,
        employee_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    ) -> np.ndarray:
        """Calculate continuous attrition probability (between 0.0 and 1.0).

        Parameters
        ----------
        employee_data : dict, list of dicts, or DataFrame
            Employee attributes.

        Returns
        -------
        np.ndarray
            Array of attrition probabilities.
        """
        df = self._prepare_inputs(employee_data)
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
        employee_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
        threshold: float = 0.50,
    ) -> np.ndarray:
        """Generate binary attrition prediction (0 = Stayed, 1 = Left).

        Parameters
        ----------
        employee_data : dict, list of dicts, or DataFrame
            Employee attributes.
        threshold : float, default=0.50
            Classification decision threshold.

        Returns
        -------
        np.ndarray
            Binary predictions.
        """
        probs = self.predict_proba(employee_data)
        return (probs >= threshold).astype(int)

    def extract_risk_factors(self, employee_record: Dict[str, Any]) -> List[str]:
        """Identify key individual behavioral risk drivers for this employee."""
        factors = []
        if employee_record.get("OverTime") == "Yes":
            factors.append("Frequent Overtime: Elevated burnout risk without schedule relief.")
        if float(employee_record.get("WorkLifeBalance", 3)) <= 1:
            factors.append("Poor Work-Life Balance: Severe work-family equilibrium strain reported.")
        if float(employee_record.get("JobSatisfaction", 3)) <= 1:
            factors.append("Critically Low Job Satisfaction: Disengagement signal in role.")
        if float(employee_record.get("YearsSinceLastPromotion", 0)) >= 5:
            yrs = employee_record.get("YearsSinceLastPromotion")
            factors.append(f"Career Stagnation: {yrs} years since last promotion.")
        if float(employee_record.get("MonthlyIncome", 6000)) < 3500:
            factors.append("Compensation Sensitivity: Monthly income in lowest quartile for role.")
        if employee_record.get("BusinessTravel") == "Travel_Frequently":
            factors.append("Travel Fatigue: High frequency business travel schedule.")
        if float(employee_record.get("StockOptionLevel", 1)) == 0:
            factors.append("Unvested Equity Gap: Zero stock option ownership reduces retention lock.")

        return factors

    def score_employee(
        self,
        employee_record: Dict[str, Any],
        threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """Generate comprehensive attrition flight risk profile and tailored HR retention playbook.

        Parameters
        ----------
        employee_record : Dict[str, Any]
            Employee attribute dictionary.
        threshold : float, default=0.50
            Decision threshold.

        Returns
        -------
        Dict[str, Any]
            Flight probability, risk tier, risk drivers, and retention recommendations.
        """
        prob = float(self.predict_proba(employee_record)[0])
        is_attrition = bool(prob >= threshold)
        factors = self.extract_risk_factors(employee_record)

        if prob >= 0.65:
            risk_tier = "High Flight Risk"
            verdict = "ELEVATED ATTRITION RISK"
            action = (
                "Priority 1-on-1 Retention Sync: Schedule confidential executive check-in. "
                "Review workload/overtime limits, evaluate compensation adjustment, and establish career path milestones."
            )
        elif prob >= threshold:
            risk_tier = "Moderate Flight Risk"
            verdict = "MONITORED ATTRITION RISK"
            action = (
                "Manager Touchpoint: Review project workload distribution, explore internal mobility opportunities, "
                "and ensure recognition for recent contributions."
            )
        elif prob >= 0.20:
            risk_tier = "Low Flight Risk"
            verdict = "STABLE ENGAGEMENT"
            action = "Standard Cadence: Maintain regular bi-weekly check-ins and support continuous skill development."
        else:
            risk_tier = "Minimal Flight Risk"
            verdict = "HIGHLY ENGAGED"
            action = "Growth Sponsorship: Identify high-impact leadership or mentorship opportunities to preserve long-term loyalty."

        return {
            "attrition_probability": prob,
            "attrition_probability_pct": f"{prob * 100.0:.1f}%",
            "is_attrition_predicted": is_attrition,
            "verdict": verdict,
            "risk_tier": risk_tier,
            "risk_factors": factors,
            "recommended_action": action,
            "model_used": self.model_name,
            "model_roc_auc": self.metrics.get("roc_auc", 0.0),
            "ethical_disclaimer": self.ETHICAL_DISCLAIMER,
        }
