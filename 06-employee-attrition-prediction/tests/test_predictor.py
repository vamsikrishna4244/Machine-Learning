"""Unit tests for production AttritionPredictor decisioning engine."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from src.data_loader import generate_attrition_dataset
from src.data_preprocessing import clean_attrition_data, prepare_attrition_train_test
from src.model_training import build_attrition_pipeline, save_attrition_pipeline
from src.predictor import AttritionPredictor


@pytest.fixture
def trained_model_path(tmp_path: Path) -> Path:
    raw = generate_attrition_dataset(n_samples=250, seed=42)
    cleaned, _ = clean_attrition_data(raw)
    X_train, _, y_train, _ = prepare_attrition_train_test(cleaned, test_size=0.20, random_state=42)

    pipe = build_attrition_pipeline(RandomForestClassifier(n_estimators=10, random_state=42))
    pipe.fit(X_train, y_train)

    return save_attrition_pipeline(
        pipeline=pipe,
        model_name="Trained RF Test",
        metrics={"roc_auc": 0.85},
        best_params={"n_estimators": 10},
        output_dir=tmp_path,
    )


def test_predictor_initialization_and_metadata(trained_model_path: Path):
    predictor = AttritionPredictor(model_path=trained_model_path)
    assert predictor.model_name == "Trained RF Test"
    assert predictor.metrics["roc_auc"] == 0.85
    assert "COMPLIANCE NOTICE" in predictor.ETHICAL_DISCLAIMER


def test_predictor_inputs_preparation(trained_model_path: Path):
    predictor = AttritionPredictor(model_path=trained_model_path)

    # Dict input with missing columns (should be auto-filled)
    d_input = {"Age": 35, "OverTime": "Yes"}
    df_prepared = predictor._prepare_inputs(d_input)
    assert len(df_prepared) == 1
    assert "MonthlyIncome" in df_prepared.columns

    # List input
    list_input = [{"Age": 25}, {"Age": 45}]
    df_list = predictor._prepare_inputs(list_input)
    assert len(df_list) == 2

    # DataFrame input
    df_in = pd.DataFrame([{"Age": 30}])
    df_out = predictor._prepare_inputs(df_in)
    assert len(df_out) == 1

    # Invalid type
    with pytest.raises(TypeError, match="Employee data must be a dict"):
        predictor._prepare_inputs(12345)


def test_predict_and_predict_proba(trained_model_path: Path):
    predictor = AttritionPredictor(model_path=trained_model_path)
    emp = {
        "Age": 28,
        "Gender": "Male",
        "MaritalStatus": "Single",
        "Department": "Sales",
        "JobRole": "Sales Representative",
        "JobLevel": 1,
        "MonthlyIncome": 2500,
        "TotalWorkingYears": 3,
        "YearsAtCompany": 1,
        "YearsInCurrentRole": 1,
        "YearsSinceLastPromotion": 0,
        "YearsWithCurrManager": 1,
        "DistanceFromHome": 25,
        "NumCompaniesWorked": 3,
        "JobSatisfaction": 1,
        "EnvironmentSatisfaction": 1,
        "WorkLifeBalance": 1,
        "JobInvolvement": 2,
        "OverTime": "Yes",
        "BusinessTravel": "Travel_Frequently",
        "StockOptionLevel": 0,
    }

    probs = predictor.predict_proba(emp)
    assert isinstance(probs, np.ndarray)
    assert 0.0 <= probs[0] <= 1.0

    preds = predictor.predict(emp, threshold=0.50)
    assert preds[0] in [0, 1]


def test_extract_risk_factors(trained_model_path: Path):
    predictor = AttritionPredictor(model_path=trained_model_path)

    high_risk_profile = {
        "OverTime": "Yes",
        "WorkLifeBalance": 1,
        "JobSatisfaction": 1,
        "YearsSinceLastPromotion": 6,
        "MonthlyIncome": 2900,
        "BusinessTravel": "Travel_Frequently",
        "StockOptionLevel": 0,
    }
    factors = predictor.extract_risk_factors(high_risk_profile)
    assert len(factors) >= 5
    assert any("Overtime" in f for f in factors)
    assert any("Promotion" in f or "Stagnation" in f for f in factors)

    low_risk_profile = {
        "OverTime": "No",
        "WorkLifeBalance": 4,
        "JobSatisfaction": 4,
        "YearsSinceLastPromotion": 1,
        "MonthlyIncome": 12000,
        "BusinessTravel": "Non-Travel",
        "StockOptionLevel": 2,
    }
    factors_low = predictor.extract_risk_factors(low_risk_profile)
    assert len(factors_low) == 0


def test_score_employee_tiers(trained_model_path: Path, monkeypatch):
    predictor = AttritionPredictor(model_path=trained_model_path)

    sample = {"Age": 30, "OverTime": "No", "JobSatisfaction": 3}

    # Test High Flight Risk (>= 0.65)
    monkeypatch.setattr(predictor, "predict_proba", lambda x: np.array([0.80]))
    high_rep = predictor.score_employee(sample)
    assert high_rep["risk_tier"] == "High Flight Risk"
    assert "ELEVATED" in high_rep["verdict"]

    # Test Moderate Flight Risk (>= 0.50)
    monkeypatch.setattr(predictor, "predict_proba", lambda x: np.array([0.55]))
    mod_rep = predictor.score_employee(sample)
    assert mod_rep["risk_tier"] == "Moderate Flight Risk"
    assert "MONITORED" in mod_rep["verdict"]

    # Test Low Flight Risk (>= 0.20)
    monkeypatch.setattr(predictor, "predict_proba", lambda x: np.array([0.30]))
    low_rep = predictor.score_employee(sample)
    assert low_rep["risk_tier"] == "Low Flight Risk"
    assert "STABLE" in low_rep["verdict"]

    # Test Minimal Flight Risk (< 0.20)
    monkeypatch.setattr(predictor, "predict_proba", lambda x: np.array([0.10]))
    min_rep = predictor.score_employee(sample)
    assert min_rep["risk_tier"] == "Minimal Flight Risk"
    assert "HIGHLY ENGAGED" in min_rep["verdict"]


def test_predictor_without_predict_proba(trained_model_path: Path):
    predictor = AttritionPredictor(model_path=trained_model_path)

    # Class with decision_function only
    class DummyDecision:
        def decision_function(self, df):
            return np.array([1.5])
    predictor.pipeline = DummyDecision()
    probs = predictor.predict_proba({"Age": 35})
    assert len(probs) == 1
    assert 0.0 <= probs[0] <= 1.0

    # Class with predict only
    class DummyPredict:
        def predict(self, df):
            return np.array([1])
    predictor.pipeline = DummyPredict()
    probs2 = predictor.predict_proba({"Age": 35})
    assert probs2[0] == 1.0
