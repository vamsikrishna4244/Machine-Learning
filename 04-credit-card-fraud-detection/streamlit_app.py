"""
Credit Card Fraud Detection and Real-Time Transaction Decisioning Dashboard.

Provides real-time fraud scoring for payment transactions, cost-sensitive threshold
optimization, simulated bank authorization policies, and imbalanced model benchmarking.
"""

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Card Fraud Detection & Risk Engine",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import FraudPredictor
from src.data_loader import load_fraud_data
from src.data_preprocessing import preprocess_fraud_data, prepare_stratified_train_test
from src.model_training import train_fraud_classifiers, save_best_fraud_model
from src.model_evaluation import (
    evaluate_fraud_models,
    plot_fraud_evaluations,
    calculate_business_cost_impact,
)


@st.cache_resource(show_spinner=False)
def load_or_train_fraud_model():
    """Load the serialized fraud detection pipeline or train one if missing."""
    model_path = PROJECT_ROOT / "models" / "best_fraud_detector.joblib"
    if not model_path.exists():
        raw_df = load_fraud_data()
        clean_df, _ = preprocess_fraud_data(raw_df)
        X_train, X_test, y_train, y_test = prepare_stratified_train_test(clean_df)
        models = train_fraud_classifiers(X_train, y_train)
        summary_df, preds, probs = evaluate_fraud_models(models, X_test, y_test)
        best_name = summary_df.iloc[0]["Model"]
        best_pipe = models[best_name]
        metrics = summary_df.iloc[0].to_dict()
        save_best_fraud_model(
            best_pipe,
            best_name,
            {
                "accuracy": metrics["Accuracy"],
                "precision": metrics["Precision"],
                "recall": metrics["Recall"],
                "f1_score": metrics["F1-Score"],
                "pr_auc": metrics["PR-AUC"],
                "roc_auc": metrics["ROC-AUC"],
            },
        )
        plot_fraud_evaluations(y_test, preds, probs, summary_df, best_pipe)

    return FraudPredictor(model_path=model_path)


@st.cache_data(show_spinner=False)
def load_cached_benchmarks():
    """Load or generate cached classifier benchmark summary."""
    data_path = PROJECT_ROOT / "data" / "processed_transactions.csv"
    if data_path.exists():
        clean_df = pd.read_csv(data_path)
    else:
        raw_df = load_fraud_data()
        clean_df, _ = preprocess_fraud_data(raw_df)

    X_train, X_test, y_train, y_test = prepare_stratified_train_test(clean_df)
    models = train_fraud_classifiers(X_train, y_train)
    summary_df, _, _ = evaluate_fraud_models(models, X_test, y_test)
    return summary_df


# Sidebar Controls
st.sidebar.title("💳 Transaction Controls")
st.sidebar.markdown("Configure transaction attributes or load presets to evaluate fraud likelihood.")

threshold = st.sidebar.slider(
    "🎯 Fraud Decision Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help="Adjust decision threshold to balance False Positives (customer friction) vs False Negatives (unrecovered fraud).",
)

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Transaction Preset Profiles")

preset_choice = st.sidebar.radio(
    "Select Transaction Template:",
    [
        "Custom Parameters",
        "Coffee Shop Purchase ($4.50 - Normal)",
        "Supermarket Grocery ($85.00 - Normal)",
        "Out-of-Pattern High Value ($1,450.00 - Fraud)",
        "Micro Authorization Probe ($1.85 - Fraud)",
    ],
    index=0,
)

PRESETS = {
    "Coffee Shop Purchase ($4.50 - Normal)": {
        "Time": 36000.0, "Amount": 4.50, "V4": 0.1, "V10": 0.2, "V12": 0.4, "V14": 0.5, "V17": 0.1, "V11": -0.2
    },
    "Supermarket Grocery ($85.00 - Normal)": {
        "Time": 48200.0, "Amount": 85.00, "V4": 0.3, "V10": 0.1, "V12": 0.2, "V14": 0.3, "V17": -0.1, "V11": 0.1
    },
    "Out-of-Pattern High Value ($1,450.00 - Fraud)": {
        "Time": 82000.0, "Amount": 1450.00, "V4": 4.2, "V10": -4.1, "V12": -4.8, "V14": -6.5, "V17": -3.9, "V11": 3.1
    },
    "Micro Authorization Probe ($1.85 - Fraud)": {
        "Time": 12000.0, "Amount": 1.85, "V4": 3.8, "V10": -3.5, "V12": -4.2, "V14": -5.1, "V17": -3.2, "V11": 2.7
    },
}

selected_preset = PRESETS.get(preset_choice, {})

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Transaction Parameters")

amount = st.sidebar.number_input(
    "Transaction Amount ($)",
    min_value=0.50,
    max_value=10000.0,
    value=float(selected_preset.get("Amount", 75.00)),
    step=5.0,
)

time_sec = st.sidebar.slider(
    "Elapsed Time (Seconds)",
    min_value=0,
    max_value=172800,
    value=int(selected_preset.get("Time", 50000)),
    step=1000,
)

st.sidebar.markdown("---")
st.sidebar.subheader("🧬 Key Latent PCA Features")

col_v1, col_v2 = st.sidebar.columns(2)
with col_v1:
    v14 = st.slider("V14 (Strong Fraud Negative)", -10.0, 5.0, float(selected_preset.get("V14", 0.2)), 0.1)
    v12 = st.slider("V12 (Geo / Device Latent)", -10.0, 5.0, float(selected_preset.get("V12", 0.3)), 0.1)
    v10 = st.slider("V10 (Behavioral Latent)", -10.0, 5.0, float(selected_preset.get("V10", 0.1)), 0.1)
with col_v2:
    v4 = st.slider("V4 (Velocity Positive)", -5.0, 10.0, float(selected_preset.get("V4", 0.2)), 0.1)
    v11 = st.slider("V11 (Risk Positive)", -5.0, 10.0, float(selected_preset.get("V11", 0.0)), 0.1)
    v17 = st.slider("V17 (Account Latent)", -10.0, 5.0, float(selected_preset.get("V17", 0.0)), 0.1)

# Main Application Title
st.title("💳 Credit Card Fraud Detection & Risk Authorization Engine")
st.markdown(
    "Real-time transaction fraud scoring engineered to handle severe class imbalance (~0.8% fraud prevalence). "
    "Features cost-sensitive loss curves, precision-recall optimization, and automated bank authorization tiers."
)

predictor = load_or_train_fraud_model()

# Prepare transaction dictionary
transaction_payload = {
    "Time": float(time_sec),
    "Amount": float(amount),
    "V4": float(v4),
    "V10": float(v10),
    "V11": float(v11),
    "V12": float(v12),
    "V14": float(v14),
    "V17": float(v17),
}

score = predictor.score_transaction(transaction_payload, threshold=threshold)
prob = score["fraud_probability"]
is_fraud = score["is_fraud"]
decision = score["decision"]
risk_tier = score["risk_tier"]
action = score["recommended_action"]
signals = score["flagged_signals"]

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🛡️ Real-Time Authorization Engine",
    "📊 Imbalanced Machine Learning Benchmarks",
    "📈 Diagnostic Curves & Loss Economics",
    "📂 Transaction Dataset Explorer",
])

with tab1:
    st.subheader("Transaction Fraud Risk Assessment")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Fraud Likelihood",
            value=f"{prob:.2%}",
            delta=f"{'+' if prob >= threshold else '-'}{abs(prob - threshold):.2%} vs Threshold",
            delta_color="inverse",
        )
    with c2:
        tier_color = "red" if risk_tier == "High Risk" else ("orange" if "Risk" in risk_tier or "Caution" in risk_tier else "green")
        st.markdown("**Risk Tier**")
        st.markdown(f"<h3 style='color:{tier_color}; margin-top:-10px;'>{risk_tier}</h3>", unsafe_allow_html=True)
    with c3:
        dec_color = "red" if is_fraud else ("orange" if "2FA" in decision else "green")
        st.markdown("**Authorization Decision**")
        st.markdown(f"<h4 style='color:{dec_color}; margin-top:-10px;'>{decision}</h4>", unsafe_allow_html=True)
    with c4:
        st.metric(label="Decision Threshold", value=f"{threshold:.2f}")

    # Risk progress bar
    st.progress(float(prob))

    # Actionable guidance
    if is_fraud:
        st.error(f"**Authorization Policy Action:**\n\n{action}")
    elif "2FA" in decision:
        st.warning(f"**Authorization Policy Action:**\n\n{action}")
    else:
        st.success(f"**Authorization Policy Action:**\n\n{action}")

    st.markdown("---")
    st.subheader("Detected Behavioral & Latent Anomaly Signals")
    if signals:
        for s in signals:
            st.markdown(f"⚠️ `{s}`")
    else:
        st.markdown("✅ *No anomalous signals detected; all transaction metrics conform to standard cardholder patterns.*")

with tab2:
    st.subheader("Imbalanced Machine Learning Benchmarks")
    st.markdown(
        "Standard classification accuracy is fundamentally misleading on imbalanced datasets. "
        "A naive dummy classifier predicting 'Legitimate' for every transaction achieves **99.2% Accuracy** while capturing **0% of Fraud**. "
        "The models below are evaluated using **PR-AUC (Average Precision)**, **Recall**, and **Asymmetric Financial Loss**."
    )

    summary_df = load_cached_benchmarks()
    st.dataframe(summary_df, width="stretch")

    st.markdown("---")
    st.subheader("Asymmetric Cost Matrix: False Positives vs. False Negatives")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown(
            """
            **False Positive (Type I Error)**
            - *Event*: Legitimate cardholder flagged as fraud.
            - *Business Cost*: **$10** (Customer friction, SMS verification cost, potential cart abandonment).
            - *Strategy*: Acceptable at moderate volume to prevent catastrophic balance drainage.
            """
        )
    with col_c2:
        st.markdown(
            """
            **False Negative (Type II Error)**
            - *Event*: Fraudulent stolen card transaction authorized.
            - *Business Cost*: **$400+** (Direct chargeback liability, merchant dispute fee, loss of stolen goods).
            - *Strategy*: Extremely expensive; models must prioritize high Recall on fraud events.
            """
        )

with tab3:
    st.subheader("Model Diagnostic Charts")
    vis_dir = PROJECT_ROOT / "visualizations"

    c_g1, c_g2 = st.columns(2)
    p_pr = vis_dir / "pr_curves.png"
    p_roc = vis_dir / "roc_curves.png"
    if p_pr.exists():
        with c_g1:
            st.image(str(p_pr), caption="Precision-Recall Curves (Class Imbalance Robust)", width="stretch")
    if p_roc.exists():
        with c_g2:
            st.image(str(p_roc), caption="Receiver Operating Characteristic (ROC) Comparison", width="stretch")

    c_g3, c_g4 = st.columns(2)
    p_cm = vis_dir / "confusion_matrices.png"
    p_cost = vis_dir / "cost_tradeoff_curve.png"
    if p_cm.exists():
        with c_g3:
            st.image(str(p_cm), caption="Confusion Matrices Across Balancing Regimes", width="stretch")
    if p_cost.exists():
        with c_g4:
            st.image(str(p_cost), caption="Total Financial Loss vs. Decision Threshold", width="stretch")

with tab4:
    st.subheader("Credit Card Transaction Explorer")
    data_path = PROJECT_ROOT / "data" / "credit_card_fraud.csv"
    if data_path.exists():
        df_txn = pd.read_csv(data_path)
        st.markdown(f"**Total Transactions:** `{len(df_txn):,}` | **Fraud Rate:** `{df_txn['Class'].mean():.3%}`")
        st.dataframe(df_txn[["Time", "Amount", "V4", "V10", "V12", "V14", "Class"]].head(50), width="stretch")

        st.markdown("#### Transaction Class Distribution")
        dist = df_txn["Class"].value_counts().rename({0: "Legitimate (0)", 1: "Fraudulent (1)"})
        st.bar_chart(dist)
    else:
        st.info("Transaction dataset not found. Run `train.py` to generate.")
