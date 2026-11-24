"""
Enterprise Employee Attrition & Retention Intelligence Dashboard.

An interactive HR analytics decision-support console for flight risk assessment,
cross-validation benchmarking, hyperparameter tuning inspection, and ethical AI auditing.
"""

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="Employee Attrition & Retention Intelligence",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import AttritionPredictor
from src.data_loader import load_attrition_data
from src.data_preprocessing import clean_attrition_data, prepare_attrition_train_test
from src.model_training import (
    train_baseline_models,
    tune_best_model_grid_search,
    save_attrition_pipeline,
)
from src.model_evaluation import (
    evaluate_attrition_models,
    conduct_demographic_fairness_audit,
    plot_attrition_evaluations,
)


@st.cache_resource(show_spinner=False)
def load_or_train_attrition_model():
    """Load the serialized attrition pipeline or train one if missing."""
    model_path = PROJECT_ROOT / "models" / "best_attrition_model.joblib"
    if not model_path.exists():
        raw_df = load_attrition_data()
        clean_df, _ = clean_attrition_data(raw_df)
        X_train, X_test, y_train, y_test = prepare_attrition_train_test(clean_df)
        candidates, _ = train_baseline_models(X_train, y_train)
        tuned_pipe, best_params, _ = tune_best_model_grid_search(X_train, y_train)
        candidates["Tuned Random Forest"] = tuned_pipe
        summary_df, preds, probs = evaluate_attrition_models(candidates, X_test, y_test)
        best_name = summary_df.iloc[0]["Model"]
        best_pipe = candidates[best_name]
        metrics = summary_df.iloc[0].to_dict()
        save_attrition_pipeline(
            best_pipe,
            best_name,
            {
                "accuracy": metrics["Accuracy"],
                "precision": metrics["Precision"],
                "recall": metrics["Recall"],
                "f1_score": metrics["F1-Score"],
                "roc_auc": metrics["ROC-AUC"],
                "pr_auc": metrics["PR-AUC"],
            },
            best_params=best_params,
        )
        plot_attrition_evaluations(y_test, preds, probs, summary_df, best_pipe)

    return AttritionPredictor(model_path=model_path)


@st.cache_data(show_spinner=False)
def load_cached_benchmarks():
    """Load or generate cached cross-validation and evaluation benchmark summaries."""
    data_path = PROJECT_ROOT / "data" / "processed_attrition.csv"
    if data_path.exists():
        clean_df = pd.read_csv(data_path)
    else:
        raw_df = load_attrition_data()
        clean_df, _ = clean_attrition_data(raw_df)

    X_train, X_test, y_train, y_test = prepare_attrition_train_test(clean_df)
    candidates, cv_df = train_baseline_models(X_train, y_train)
    summary_df, _, _ = evaluate_attrition_models(candidates, X_test, y_test)
    audit_df = conduct_demographic_fairness_audit(candidates["Logistic Regression"], X_test, "Gender")
    return clean_df, cv_df, summary_df, audit_df


# Sidebar Configuration
st.sidebar.title("👥 Employee Profile Controls")
st.sidebar.markdown("Configure workforce parameters or select an employee preset.")

preset_choice = st.sidebar.radio(
    "Load Profile Preset:",
    [
        "Custom Parameters",
        "At-Risk Sales Rep (Heavy OverTime)",
        "Stagnant Mid-Level Engineer (No Promotion)",
        "Satisfied R&D Manager (High Retention)",
    ],
    index=0,
)

PRESETS = {
    "At-Risk Sales Rep (Heavy OverTime)": {
        "Age": 28, "Gender": "Male", "MaritalStatus": "Single", "Department": "Sales",
        "JobRole": "Sales Representative", "JobLevel": 1, "MonthlyIncome": 2800,
        "TotalWorkingYears": 3, "YearsAtCompany": 2, "YearsInCurrentRole": 1,
        "YearsSinceLastPromotion": 2, "YearsWithCurrManager": 1, "DistanceFromHome": 25,
        "NumCompaniesWorked": 2, "JobSatisfaction": 1, "EnvironmentSatisfaction": 1,
        "WorkLifeBalance": 1, "JobInvolvement": 2, "OverTime": "Yes",
        "BusinessTravel": "Travel_Frequently", "StockOptionLevel": 0,
    },
    "Stagnant Mid-Level Engineer (No Promotion)": {
        "Age": 36, "Gender": "Female", "MaritalStatus": "Married", "Department": "Research & Development",
        "JobRole": "Software Engineer", "JobLevel": 2, "MonthlyIncome": 5400,
        "TotalWorkingYears": 11, "YearsAtCompany": 8, "YearsInCurrentRole": 6,
        "YearsSinceLastPromotion": 6, "YearsWithCurrManager": 5, "DistanceFromHome": 14,
        "NumCompaniesWorked": 3, "JobSatisfaction": 2, "EnvironmentSatisfaction": 3,
        "WorkLifeBalance": 2, "JobInvolvement": 3, "OverTime": "Yes",
        "BusinessTravel": "Travel_Rarely", "StockOptionLevel": 0,
    },
    "Satisfied R&D Manager (High Retention)": {
        "Age": 48, "Gender": "Female", "MaritalStatus": "Married", "Department": "Research & Development",
        "JobRole": "Manager", "JobLevel": 4, "MonthlyIncome": 14200,
        "TotalWorkingYears": 24, "YearsAtCompany": 15, "YearsInCurrentRole": 8,
        "YearsSinceLastPromotion": 2, "YearsWithCurrManager": 7, "DistanceFromHome": 5,
        "NumCompaniesWorked": 2, "JobSatisfaction": 4, "EnvironmentSatisfaction": 4,
        "WorkLifeBalance": 3, "JobInvolvement": 4, "OverTime": "No",
        "BusinessTravel": "Travel_Rarely", "StockOptionLevel": 2,
    },
}

selected_preset = PRESETS.get(preset_choice, {})

st.sidebar.markdown("---")
threshold = st.sidebar.slider(
    "🎯 Flight Risk Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="Adjust threshold to balance proactive engagement reach (high recall) vs focused intervention resources.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("🏢 Role & Organization")

dept = st.sidebar.selectbox(
    "Department",
    ["Research & Development", "Sales", "Human Resources"],
    index=["Research & Development", "Sales", "Human Resources"].index(
        selected_preset.get("Department", "Research & Development")
    ),
)

role = st.sidebar.selectbox(
    "Job Role",
    [
        "Software Engineer", "Research Scientist", "Sales Executive", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Sales Representative",
        "Manager", "Human Resources Specialist",
    ],
    index=0,
)

job_level = st.sidebar.slider("Job Level (1 to 5)", 1, 5, int(selected_preset.get("JobLevel", 2)))
monthly_income = st.sidebar.number_input("Monthly Income ($)", 1800, 25000, int(selected_preset.get("MonthlyIncome", 5500)), 250)

st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Work Conditions & Satisfaction")

col_w1, col_w2 = st.sidebar.columns(2)
with col_w1:
    overtime = st.selectbox("OverTime", ["Yes", "No"], index=0 if selected_preset.get("OverTime") == "Yes" else 1)
    work_life = st.slider("Work-Life Balance (1-4)", 1, 4, int(selected_preset.get("WorkLifeBalance", 3)))
    job_sat = st.slider("Job Satisfaction (1-4)", 1, 4, int(selected_preset.get("JobSatisfaction", 3)))
with col_w2:
    travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"], index=0)
    env_sat = st.slider("Environment Sat (1-4)", 1, 4, int(selected_preset.get("EnvironmentSatisfaction", 3)))
    involvement = st.slider("Job Involvement (1-4)", 1, 4, int(selected_preset.get("JobInvolvement", 3)))

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Experience & Demographics")

col_e1, col_e2 = st.sidebar.columns(2)
with col_e1:
    age = st.slider("Age", 21, 60, int(selected_preset.get("Age", 34)))
    gender = st.selectbox("Gender", ["Female", "Male"], index=0)
    marital = st.selectbox("Marital Status", ["Married", "Single", "Divorced"], index=0)
    stock_opt = st.slider("Stock Option Level (0-3)", 0, 3, int(selected_preset.get("StockOptionLevel", 1)))
with col_e2:
    years_company = st.slider("Years at Company", 0, 30, int(selected_preset.get("YearsAtCompany", 5)))
    years_promo = st.slider("Years Since Last Promo", 0, 15, int(selected_preset.get("YearsSinceLastPromotion", 2)))
    distance = st.slider("Distance from Home (mi)", 1, 30, int(selected_preset.get("DistanceFromHome", 8)))
    total_working = st.slider("Total Working Years", 1, 40, int(selected_preset.get("TotalWorkingYears", 8)))

# Main Application Title
st.title("👥 Employee Attrition Prediction & Retention Intelligence")
st.markdown(
    "Enterprise decision-support system modeling workforce attrition hazards using Scikit-learn Pipelines, "
    "5-fold cross-validation, and GridSearchCV hyperparameter tuning. Built in adherence to responsible AI ethics."
)

predictor = load_or_train_attrition_model()

# Prepare employee record
employee_payload = {
    "Age": age,
    "Gender": gender,
    "MaritalStatus": marital,
    "Department": dept,
    "JobRole": role,
    "JobLevel": job_level,
    "MonthlyIncome": monthly_income,
    "TotalWorkingYears": total_working,
    "YearsAtCompany": years_company,
    "YearsInCurrentRole": min(years_company, 4),
    "YearsSinceLastPromotion": years_promo,
    "YearsWithCurrManager": min(years_company, 4),
    "DistanceFromHome": distance,
    "NumCompaniesWorked": 2,
    "JobSatisfaction": job_sat,
    "EnvironmentSatisfaction": env_sat,
    "WorkLifeBalance": work_life,
    "JobInvolvement": involvement,
    "OverTime": overtime,
    "BusinessTravel": travel,
    "StockOptionLevel": stock_opt,
}

score = predictor.score_employee(employee_payload, threshold=threshold)
prob = score["attrition_probability"]
risk_tier = score["risk_tier"]
verdict = score["verdict"]
action = score["recommended_action"]
factors = score["risk_factors"]

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Flight Risk Assessment",
    "📈 Model Comparison & CV Benchmarks",
    "🔍 Feature Importance & Drivers",
    "⚖️ Ethical AI & Fairness Audit",
    "📂 Workforce Data Explorer",
])

with tab1:
    st.subheader("Employee Flight Risk Assessment")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Attrition Probability",
            value=f"{prob:.1%}",
            delta=f"{'+' if prob >= threshold else '-'}{abs(prob - threshold):.1%} vs Threshold",
            delta_color="inverse",
        )
    with c2:
        tier_color = "red" if "High" in risk_tier else ("orange" if "Moderate" in risk_tier else "green")
        st.markdown("**Risk Tier**")
        st.markdown(f"<h3 style='color:{tier_color}; margin-top:-10px;'>{risk_tier}</h3>", unsafe_allow_html=True)
    with c3:
        st.markdown("**Advisory Verdict**")
        st.markdown(f"<h4 style='color:{tier_color}; margin-top:-10px;'>{verdict}</h4>", unsafe_allow_html=True)
    with c4:
        st.metric(label="Decision Threshold", value=f"{threshold:.2f}")

    # Progress bar
    st.progress(float(prob))

    # Action recommendation
    if "High" in risk_tier or "Moderate" in risk_tier:
        st.warning(f"**Recommended HR Retention Action:**\n\n{action}")
    else:
        st.success(f"**Recommended HR Retention Action:**\n\n{action}")

    st.markdown("---")
    st.subheader("Key Attrition Risk Signals Identified for this Employee")
    if factors:
        for f in factors:
            st.markdown(f"⚠️ `{f}`")
    else:
        st.markdown("✅ *No elevated flight risk factors identified for this profile.*")

    st.info(f"🛡️ **Responsible AI Policy:** {predictor.ETHICAL_DISCLAIMER}")

with tab2:
    st.subheader("Cross-Validation & Hyperparameter Tuning Benchmarks")
    clean_df, cv_df, summary_df, _ = load_cached_benchmarks()

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown("#### 5-Fold Stratified Cross-Validation (Training Partition)")
        st.dataframe(cv_df, width="stretch")
    with col_b2:
        st.markdown("#### Held-Out Test Set Benchmark ($N = 400$)")
        st.dataframe(summary_df, width="stretch")

    vis_dir = PROJECT_ROOT / "visualizations"
    p_comp = vis_dir / "model_comparison_benchmark.png"
    if p_comp.exists():
        st.image(str(p_comp), caption="Model Performance Across Accuracy, Recall, F1, and ROC-AUC", width="stretch")

with tab3:
    st.subheader("Model Interpretability & Feature Importance")
    st.markdown(
        "Feature importances reveal the organizational factors driving attrition. "
        "Top drivers include **OverTime**, **JobSatisfaction**, **MonthlyIncome**, and **WorkLifeBalance**."
    )

    c_f1, c_f2 = st.columns(2)
    p_feat = vis_dir / "feature_importance_top15.png"
    p_roc = vis_dir / "roc_pr_curves.png"
    if p_feat.exists():
        with c_f1:
            st.image(str(p_feat), caption="Top 15 Most Predictive Attrition Features", width="stretch")
    if p_roc.exists():
        with c_f2:
            st.image(str(p_roc), caption="ROC & Precision-Recall Curves", width="stretch")

with tab4:
    st.subheader("Ethical AI, Algorithmic Fairness & Responsible Use")
    st.markdown(
        """
        ### Responsible AI Governance Framework
        Predictive models in Human Resources must strictly comply with employment discrimination laws (e.g., Title VII, EEOC guidelines).
        
        > [!IMPORTANT]
        > **Ethical Principles:**
        > 1. **Advisory Signal Only**: Attrition scores must strictly serve as advisory telemetry for managerial pulse check-ins and wellness support.
        > 2. **Never for Adverse Decisions**: Models must **never** automate or justify termination, disciplinary action, salary reductions, or promotion withholding.
        > 3. **Four-Fifths (80%) Rule Compliance**: Selection and predicted attrition rates across protected demographic categories must not exhibit adverse disparate impact.
        """
    )

    _, _, _, audit_df = load_cached_benchmarks()
    st.markdown("#### Demographic Disparate Impact Audit (Gender)")
    st.dataframe(audit_df, width="stretch")

with tab5:
    st.subheader("Workforce Dataset Explorer")
    clean_df, _, _, _ = load_cached_benchmarks()
    st.markdown(f"**Total Workforce Records:** `{len(clean_df):,}` | **Overall Attrition:** `{clean_df['Attrition'].mean():.1%}`")
    st.dataframe(clean_df.head(50), width="stretch")
