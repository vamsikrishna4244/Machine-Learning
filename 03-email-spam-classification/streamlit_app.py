"""
Interactive Email Spam Classification and Threat Intelligence Dashboard.

Provides real-time spam scoring for incoming emails, keyword trigger attribution,
customizable classification thresholds, and comprehensive BoW vs. TF-IDF model benchmarks.
"""

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="Email Spam Classifier & Threat Intelligence",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import SpamPredictor
from src.data_loader import load_email_data
from src.text_preprocessing import preprocess_corpus, split_email_dataset
from src.model_training import train_spam_classifiers, save_best_spam_pipeline
from src.model_evaluation import evaluate_spam_models, compare_vectorization_methods, plot_spam_evaluations


@st.cache_resource(show_spinner=False)
def load_or_train_spam_model():
    """Load the serialized spam classifier or train one if not yet saved."""
    model_path = PROJECT_ROOT / "models" / "best_spam_classifier.joblib"
    if not model_path.exists():
        raw_df = load_email_data()
        clean_df, _ = preprocess_corpus(raw_df)
        X_train, X_test, y_train, y_test = split_email_dataset(clean_df)
        models = train_spam_classifiers(X_train, y_train)
        summary_df, preds, _ = evaluate_spam_models(models, X_test, y_test)
        best_cfg = summary_df.iloc[0]["Configuration"]
        best_pipe = models[best_cfg]
        metrics = summary_df.iloc[0].to_dict()
        save_best_spam_pipeline(
            best_pipe,
            best_cfg,
            {
                "accuracy": metrics["Accuracy"],
                "precision": metrics["Precision"],
                "recall": metrics["Recall"],
                "f1_score": metrics["F1-Score"],
                "roc_auc": metrics["ROC-AUC"],
            },
        )
        plot_spam_evaluations(y_test, preds, summary_df, best_pipe)

    return SpamPredictor(model_path=model_path)


@st.cache_data(show_spinner=False)
def load_cached_benchmarks():
    """Generate and cache multi-model and vectorizer performance comparison tables."""
    data_path = PROJECT_ROOT / "data" / "processed_emails.csv"
    if data_path.exists():
        clean_df = pd.read_csv(data_path)
    else:
        raw_df = load_email_data()
        clean_df, _ = preprocess_corpus(raw_df)

    X_train, X_test, y_train, y_test = split_email_dataset(clean_df)
    models = train_spam_classifiers(X_train, y_train)
    summary_df, _, _ = evaluate_spam_models(models, X_test, y_test)
    vec_summary = compare_vectorization_methods(summary_df)
    return summary_df, vec_summary


# Sidebar Configuration
st.sidebar.title("🛡️ Gateway Security Controls")
st.sidebar.markdown("Configure email security gateway sensitivity and load presets.")

threshold = st.sidebar.slider(
    "🎯 Spam Decision Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="Lower threshold to capture all suspicious emails (high recall); raise threshold to minimize false positives.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Preset Email Templates")

sample_choice = st.sidebar.radio(
    "Load Test Message:",
    [
        "Custom Email Input",
        "Prize & Lottery Giveaway (Spam)",
        "Urgent Account Suspension (Spam)",
        "Engineering Sprint Review (Ham)",
        "Vendor Invoice Clearance (Ham)",
    ],
    index=0,
)

PRESETS = {
    "Prize & Lottery Giveaway (Spam)": (
        "Congratulations! You won the official Apple iPhone 15 lottery giveaway. "
        "Claim your prize immediately at http://secure-verify-999.net/claim within 24 hours."
    ),
    "Urgent Account Suspension (Spam)": (
        "FINAL WARNING: Your bank account has been flagged for unauthorized login attempts. "
        "Click the link below immediately to verify your credentials and claim your refund: http://bank-update.xyz"
    ),
    "Engineering Sprint Review (Ham)": (
        "Hi team, please review the attached sprint burndown chart and let me know if there are any blockers "
        "before Friday's demo. All unit tests and CI pipelines are passing."
    ),
    "Vendor Invoice Clearance (Ham)": (
        "Hello all, please find attached the paid invoice for our Q3 cloud software subscription. "
        "Finance has already processed and cleared the vendor charge."
    ),
}

# Main Application Title
st.title("📧 Email Spam Classification & Threat Intelligence")
st.markdown(
    "Evaluate real-time spam risk using classical NLP vectorization (Bag-of-Words and TF-IDF) "
    "combined with Naive Bayes, Logistic Regression, and Support Vector Machines."
)

predictor = load_or_train_spam_model()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Real-Time Message Scanner",
    "📊 Vectorizer & Model Benchmarks",
    "📈 Diagnostic Visualizations",
    "📂 Corpus Explorer",
])

with tab1:
    st.subheader("Interactive Email Content Scanner")

    # Set initial text based on preset selection
    default_text = PRESETS.get(sample_choice, "")

    email_input = st.text_area(
        "Paste Email Subject & Body Below:",
        value=default_text,
        height=140,
        placeholder="Type or paste email subject and text here...",
    )

    if email_input.strip():
        score = predictor.score_email(email_input, threshold=threshold)
        prob = score["spam_probability"]
        is_spam = score["is_spam"]
        tier = score["confidence_tier"]
        action = score["recommended_action"]
        triggers = score["trigger_keywords"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="Spam Probability",
                value=f"{prob:.1%}",
                delta=f"{'+' if prob >= threshold else '-'}{abs(prob - threshold):.1%} vs Threshold",
                delta_color="inverse",
            )
        with col2:
            badge_color = "red" if is_spam else "green"
            verdict = "🚨 SPAM DETECTED" if is_spam else "✅ LEGITIMATE (HAM)"
            st.markdown("**Verdict**")
            st.markdown(f"<h3 style='color:{badge_color}; margin-top:-10px;'>{verdict}</h3>", unsafe_allow_html=True)
        with col3:
            st.markdown("**Confidence Tier**")
            st.markdown(f"<h4 style='margin-top:-10px;'>{tier}</h4>", unsafe_allow_html=True)
        with col4:
            st.metric(label="Active Threshold", value=f"{threshold:.2f}")

        # Progress bar
        st.progress(float(prob))

        # Action policy alert
        if is_spam:
            st.error(f"**Recommended Gateway Action:** {action}")
        else:
            st.success(f"**Recommended Gateway Action:** {action}")

        # Trigger tokens display
        st.markdown("#### Detected Keyword Attribution")
        if triggers:
            st.markdown("The following vocabulary tokens positively increased the spam likelihood score:")
            token_pills = " ".join([f"`{t}`" for t in triggers])
            st.markdown(f"🏷️ **Trigger Tokens:** {token_pills}")
        else:
            st.markdown("ℹ️ *No strong spam-inducing vocabulary tokens were detected.*")
    else:
        st.info("👈 Enter email text above or select a preset template from the sidebar to inspect spam score.")

with tab2:
    st.subheader("Vectorizer & Machine Learning Benchmarks")
    st.markdown(
        "Benchmarking all six permutations of **Bag-of-Words (CountVectorizer)** and **TF-IDF (TfidfVectorizer)** "
        "paired with **Multinomial Naive Bayes**, **Logistic Regression**, and **Support Vector Machines**."
    )

    summary_df, vec_summary = load_cached_benchmarks()

    st.markdown("#### All Model Configurations")
    st.dataframe(summary_df, width="stretch")

    st.markdown("#### Vectorization Technique Comparison (Mean Performance)")
    st.dataframe(vec_summary, width="stretch")

    st.markdown("---")
    st.subheader("Architectural Comparison: Bag of Words vs. TF-IDF")
    c_bow, c_tfidf = st.columns(2)
    with c_bow:
        st.markdown(
            """
            **Bag of Words (CountVectorizer)**
            - *Mechanism*: Simple unigram and bigram frequency counts.
            - *Advantage*: Captures raw token presence; highly effective for Naive Bayes generative probabilistic modeling.
            - *Limitation*: Can over-weight common document-level tokens if stopword filtering is incomplete.
            """
        )
    with c_tfidf:
        st.markdown(
            """
            **TF-IDF (TfidfVectorizer)**
            - *Mechanism*: Scales term frequency inversely with document frequency: $\\text{TF-IDF} = \\text{TF} \\times \\log(N / \\text{DF})$.
            - *Advantage*: Naturally dampens ubiquitous domain words and highlights rare, discriminative spam keywords.
            - *Limitation*: Slightly higher computational matrix construction cost during continuous streaming ingestion.
            """
        )

with tab3:
    st.subheader("Diagnostic Visualizations")
    vis_dir = PROJECT_ROOT / "visualizations"

    c_v1, c_v2 = st.columns(2)
    p_comp = vis_dir / "vectorizer_model_comparison.png"
    p_cm = vis_dir / "confusion_matrices.png"
    p_kw = vis_dir / "top_spam_keywords.png"

    if p_comp.exists():
        with c_v1:
            st.image(str(p_comp), caption="Model & Vectorizer Benchmark (F1-Score)", width="stretch")
    if p_cm.exists():
        with c_v2:
            st.image(str(p_cm), caption="Confusion Matrices Across Top Models", width="stretch")

    if p_kw.exists():
        st.image(str(p_kw), caption="Top Indicative Tokens for Spam vs Ham Classes", width="stretch")

with tab4:
    st.subheader("Email Corpus Explorer")
    data_path = PROJECT_ROOT / "data" / "spam_emails.csv"
    if data_path.exists():
        corpus_df = pd.read_csv(data_path)
        st.markdown(f"**Total Corpus Size:** `{len(corpus_df):,}` emails | **Spam Ratio:** `{corpus_df['label'].mean():.1%}`")
        st.dataframe(corpus_df[["email_id", "subject", "label_text", "message"]].head(50), width="stretch")

        st.markdown("#### Class Distribution")
        dist = corpus_df["label_text"].value_counts()
        st.bar_chart(dist)
    else:
        st.info("Corpus dataset not found. Run `train.py` to generate data.")
