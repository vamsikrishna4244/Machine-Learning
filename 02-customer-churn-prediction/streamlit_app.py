"""
Customer Churn Prediction and Retention Strategy Dashboard.

An interactive Streamlit application providing real-time churn risk scoring,
prescriptive customer retention playbooks, threshold sensitivity tuning,
and model performance benchmarking.
"""

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Customer Churn Intelligence Dashboard",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import ChurnPredictor
from src.data_loader import load_churn_data
from src.data_preprocessing import clean_churn_data, prepare_churn_train_test_data
from src.model_training import train_churn_classifiers, save_best_churn_model
from src.model_evaluation import evaluate_churn_models, compare_churn_models, plot_churn_evaluations


@st.cache_resource(show_spinner=False)
def load_churn_predictor():
    """Load the serialized best churn model artifact, or train one if not yet saved."""
    model_path = PROJECT_ROOT / "models" / "best_customer_churn_model.joblib"
    if not model_path.exists():
        raw_df = load_churn_data()
        cleaned_df, _ = clean_churn_data(raw_df)
        X_train, X_test, y_train, y_test = prepare_churn_train_test_data(cleaned_df)
        models = train_churn_classifiers(X_train, y_train)
        summary_df, preds, probs = evaluate_churn_models(models, X_test, y_test)
        best_name = compare_churn_models(summary_df)
        best_pipe = models[best_name]
        metrics = summary_df[summary_df["Model"] == best_name].iloc[0].to_dict()
        save_best_churn_model(
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
        )
        plot_churn_evaluations(y_test, preds, probs, summary_df, best_pipe)

    return ChurnPredictor(model_path=model_path)


@st.cache_data(show_spinner=False)
def load_benchmark_summary():
    """Load or generate cached classifier benchmark summary."""
    data_dir = PROJECT_ROOT / "data"
    proc_csv = data_dir / "processed_customer_churn.csv"
    if proc_csv.exists():
        df_bench = pd.read_csv(proc_csv)
    else:
        raw_df = load_churn_data()
        df_bench, _ = clean_churn_data(raw_df)
    X_train, X_test, y_train, y_test = prepare_churn_train_test_data(df_bench)
    trained_models = train_churn_classifiers(X_train, y_train)
    bench_summary, _, _ = evaluate_churn_models(trained_models, X_test, y_test)
    return bench_summary


# Sidebar controls
st.sidebar.title("🎛️ Account Parameters")
st.sidebar.markdown("Configure subscriber attributes to evaluate churn likelihood.")

# Contract & Account
contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"],
    index=0,
    help="Month-to-month contracts have historically higher churn hazard.",
)

tenure_months = st.sidebar.slider(
    "Tenure (Months)",
    min_value=1,
    max_value=72,
    value=6,
    step=1,
    help="Number of months customer has subscribed to the telecom provider.",
)

monthly_charges = st.sidebar.slider(
    "Monthly Charges ($)",
    min_value=18.0,
    max_value=120.0,
    value=78.50,
    step=0.50,
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
    index=0,
)

paperless_billing = st.sidebar.radio("Paperless Billing", ["Yes", "No"], index=0, horizontal=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📡 Internet & Add-on Services")

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["Fiber optic", "DSL", "No"],
    index=0,
)

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    online_security = st.selectbox("Online Security", ["No", "Yes"], index=0)
    device_protection = st.selectbox("Device Protection", ["No", "Yes"], index=0)
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"], index=0)
with col_s2:
    online_backup = st.selectbox("Online Backup", ["No", "Yes"], index=0)
    tech_support = st.selectbox("Tech Support", ["No", "Yes"], index=0)
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("👥 Demographics & Phone")

col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    gender = st.selectbox("Gender", ["Female", "Male"], index=0)
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"], index=0)
    partner = st.selectbox("Partner", ["No", "Yes"], index=0)
with col_d2:
    dependents = st.selectbox("Dependents", ["No", "Yes"], index=0)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0)
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"], index=0)

st.sidebar.markdown("---")
threshold = st.sidebar.slider(
    "🎯 Decision Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="Adjust threshold to balance Precision (avoid wasted retention budget) vs Recall (catch all potential churners).",
)

# Main Dashboard
st.title("🔄 Customer Churn Intelligence & Retention Dashboard")
st.markdown(
    "Predict subscriber churn probability, evaluate business retention trade-offs, "
    "and inspect diagnostic machine learning benchmarks across Logistic Regression, Decision Tree, and Random Forest."
)

predictor = load_churn_predictor()

# Prepare customer payload
total_charges = round(monthly_charges * tenure_months, 2)
customer_record = {
    "gender": gender,
    "senior_citizen": senior_citizen,
    "partner": partner,
    "dependents": dependents,
    "tenure_months": tenure_months,
    "phone_service": phone_service,
    "multiple_lines": multiple_lines,
    "internet_service": internet_service,
    "online_security": online_security,
    "online_backup": online_backup,
    "device_protection": device_protection,
    "tech_support": tech_support,
    "streaming_tv": streaming_tv,
    "streaming_movies": streaming_movies,
    "contract": contract,
    "paperless_billing": paperless_billing,
    "payment_method": payment_method,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
}

score = predictor.score_customer(customer_record, threshold=threshold)
churn_prob = score["churn_probability"]
risk_tier = score["risk_tier"]
is_churn = score["predicted_churn"]
recommendation = score["recommended_action"]

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Churn Risk Profiler",
    "📊 Model Performance & Benchmarks",
    "📈 Diagnostic Curves & Interpretability",
    "📂 Dataset Inspector",
])

with tab1:
    st.subheader("Subscriber Churn Risk Assessment")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Churn Probability",
            value=f"{churn_prob:.1%}",
            delta=f"{'+' if churn_prob >= threshold else '-'}{abs(churn_prob - threshold):.1%} vs Threshold",
            delta_color="inverse",
        )
    with c2:
        tier_color = "red" if risk_tier == "High Risk" else ("orange" if risk_tier == "Medium Risk" else "green")
        st.markdown(f"**Risk Tier**")
        st.markdown(f"<h3 style='color:{tier_color}; margin-top:-10px;'>{risk_tier}</h3>", unsafe_allow_html=True)
    with c3:
        prediction_label = "🚨 LIKELY TO CHURN" if is_churn else "✅ RETENTION SECURE"
        pred_color = "red" if is_churn else "green"
        st.markdown(f"**Classification Decision**")
        st.markdown(f"<h3 style='color:{pred_color}; margin-top:-10px;'>{prediction_label}</h3>", unsafe_allow_html=True)
    with c4:
        st.metric(label="Decision Threshold", value=f"{threshold:.2f}")

    # Risk progress bar
    st.progress(float(churn_prob))

    # Actionable playbook
    st.info(f"**Recommended Retention Playbook:**\n\n{recommendation}")

    st.markdown("---")
    st.subheader("Key Risk Drivers for this Subscriber")

    col_ra, col_rb = st.columns(2)
    with col_ra:
        st.markdown("#### Positive Retention Anchors (Lower Churn)")
        anchors = []
        if contract in ["One year", "Two year"]:
            anchors.append(f"✅ **Contract Stability:** Committed to a long-term `{contract}` contract.")
        if tenure_months >= 24:
            anchors.append(f"✅ **Loyal Account:** Long tenure ({tenure_months} months / {tenure_months//12} yrs).")
        if online_security == "Yes" and tech_support == "Yes":
            anchors.append("✅ **High-Retention Bundle:** Both Tech Support and Online Security active.")
        if payment_method in ["Bank transfer (automatic)", "Credit card (automatic)"]:
            anchors.append("✅ **Frictionless Billing:** Automated payment reduces accidental churn.")
        if not anchors:
            st.markdown("_No strong retention anchors detected for this profile._")
        else:
            for a in anchors:
                st.markdown(a)

    with col_rb:
        st.markdown("#### Elevated Churn Risk Signals (Higher Churn)")
        signals = []
        if contract == "Month-to-month":
            signals.append("⚠️ **Month-to-Month Hazard:** High flexibility allows immediate cancellation.")
        if tenure_months <= 6:
            signals.append("⚠️ **Onboarding Vulnerability:** Early tenure (< 6 months) experiences steepest dropout.")
        if internet_service == "Fiber optic" and monthly_charges > 70.0:
            signals.append(f"⚠️ **Premium Bill Sensitivity:** High monthly charge (${monthly_charges:.2f}) on Fiber.")
        if online_security == "No" and tech_support == "No":
            signals.append("⚠️ **Unassisted Experience:** Lacks Tech Support & Online Security support.")
        if payment_method == "Electronic check":
            signals.append("⚠️ **Payment Friction:** Electronic check correlates with higher payment lapses.")
        if not signals:
            st.markdown("_No major churn hazard flags identified._")
        else:
            for s in signals:
                st.markdown(s)

with tab2:
    st.subheader("Supervised Classification Benchmarks")
    st.markdown(
        "All candidate classifiers were trained under identical stratified cross-validation splits "
        "with `class_weight='balanced'` to actively counteract class imbalance without synthetic sampling distortion."
    )

    bench_summary = load_benchmark_summary()
    st.dataframe(bench_summary, width="stretch")

    st.markdown("---")
    st.subheader("Business Impact: False Positives vs. False Negatives")
    col_fp, col_fn = st.columns(2)
    with col_fp:
        st.markdown(
            """
            **False Positive (Type I Error)**
            - *Scenario*: Model flags a loyal subscriber as churning; team sends proactive discount.
            - *Cost*: Minor retention marketing discount cost ($10 - $25 voucher).
            - *Impact*: Safe failure mode; customer feels valued even if retention was unneeded.
            """
        )
    with col_fn:
        st.markdown(
            """
            **False Negative (Type II Error)**
            - *Scenario*: Model predicts customer will stay; subscriber abruptly cancels.
            - *Cost*: Complete Customer Lifetime Value (CLV) lost ($800 - $1,500+).
            - *Impact*: Severe business impact. A lower threshold (e.g. 0.40) is often preferred to reduce False Negatives.
            """
        )

with tab3:
    st.subheader("Model Diagnostic Charts")
    vis_dir = PROJECT_ROOT / "visualizations"

    c_img1, c_img2 = st.columns(2)
    cm_path = vis_dir / "confusion_matrices.png"
    roc_path = vis_dir / "roc_curves.png"
    if cm_path.exists():
        with c_img1:
            st.image(str(cm_path), caption="Confusion Matrices Across All Classifiers", width="stretch")
    if roc_path.exists():
        with c_img2:
            st.image(str(roc_path), caption="Receiver Operating Characteristic (ROC) Comparison", width="stretch")

    c_img3, c_img4 = st.columns(2)
    pr_path = vis_dir / "precision_recall_curves.png"
    fi_path = vis_dir / "feature_importance.png"
    if pr_path.exists():
        with c_img3:
            st.image(str(pr_path), caption="Precision-Recall Curves (Class Imbalance Robust)", width="stretch")
    if fi_path.exists():
        with c_img4:
            st.image(str(fi_path), caption="Predictive Feature Importances / Coefficients", width="stretch")

with tab4:
    st.subheader("Customer Churn Dataset Explorer")
    data_path = PROJECT_ROOT / "data" / "customer_churn.csv"
    if data_path.exists():
        sample_df = pd.read_csv(data_path)
        st.markdown(f"**Total Records:** `{len(sample_df):,}` rows | **Features:** `{sample_df.shape[1]}` columns")
        st.dataframe(sample_df.head(50), width="stretch")

        st.markdown("#### Churn Class Distribution")
        churn_dist = sample_df["churn"].value_counts(normalize=True).rename({0: "Retained (0)", 1: "Churned (1)"})
        st.bar_chart(churn_dist)
    else:
        st.info("Dataset file not found. Run `train.py` to generate or load data.")
