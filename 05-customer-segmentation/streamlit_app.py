"""
Interactive Customer Segmentation and Retail Persona Intelligence Dashboard.

Provides real-time customer persona classification, marketing playbook generation,
Elbow & Silhouette cluster count optimization, and PCA latent space exploration.
"""

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Customer Segmentation & Persona Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmenter import CustomerSegmenter
from src.data_loader import load_customer_data
from src.data_preprocessing import preprocess_customer_data, scale_customer_features, apply_pca_reduction
from src.cluster_models import (
    train_kmeans,
    train_all_clustering_models,
    save_clustering_artifact,
)
from src.cluster_evaluation import (
    compute_elbow_and_silhouette_sweep,
    evaluate_clustering_metrics,
    profile_clusters,
    plot_clustering_diagnostics,
)


@st.cache_resource(show_spinner=False)
def load_or_train_segmenter():
    """Load the serialized clustering artifact or train one if not yet saved."""
    model_path = PROJECT_ROOT / "models" / "best_clustering_model.joblib"
    if not model_path.exists():
        raw_df = load_customer_data()
        clean_df, _ = preprocess_customer_data(raw_df)
        X_scaled, scaler = scale_customer_features(clean_df)
        X_pca, _, _ = apply_pca_reduction(X_scaled)
        sweep_df = compute_elbow_and_silhouette_sweep(X_scaled)
        optimal_k = int(sweep_df.sort_values(by="Silhouette_Score", ascending=False).iloc[0]["k"])
        models_dict = train_all_clustering_models(X_scaled, n_clusters=optimal_k)
        labels_dict = {name: res[1] for name, res in models_dict.items()}
        km_model = models_dict["K-Means"][0]
        profiles = profile_clusters(clean_df, labels_dict["K-Means"])
        km_metrics = evaluate_clustering_metrics(X_scaled, labels_dict["K-Means"], "K-Means")
        save_clustering_artifact(
            model=km_model,
            scaler=scaler,
            algorithm_name="K-Means",
            n_clusters=optimal_k,
            metrics={
                "silhouette": km_metrics["Silhouette"],
                "davies_bouldin": km_metrics["Davies_Bouldin"],
                "calinski_harabasz": km_metrics["Calinski_Harabasz"],
            },
            cluster_profiles=profiles,
        )
        plot_clustering_diagnostics(X_scaled, X_pca, labels_dict, sweep_df, profiles)

    return CustomerSegmenter(model_path=model_path)


@st.cache_data(show_spinner=False)
def load_cached_clustering_data():
    """Load cached dataset and compute evaluation summaries."""
    data_path = PROJECT_ROOT / "data" / "processed_customers.csv"
    if data_path.exists():
        clean_df = pd.read_csv(data_path)
    else:
        raw_df = load_customer_data()
        clean_df, _ = preprocess_customer_data(raw_df)

    X_scaled, scaler = scale_customer_features(clean_df)
    sweep_df = compute_elbow_and_silhouette_sweep(X_scaled)
    models_dict = train_all_clustering_models(X_scaled, n_clusters=5)
    eval_records = [evaluate_clustering_metrics(X_scaled, res[1], name) for name, res in models_dict.items()]
    eval_df = pd.DataFrame(eval_records)
    return clean_df, sweep_df, eval_df


# Sidebar Configuration
st.sidebar.title("🛍️ Customer Profile Inputs")
st.sidebar.markdown("Configure behavioral spending parameters or choose an archetype preset.")

preset_choice = st.sidebar.radio(
    "Load Customer Preset:",
    [
        "Custom Input",
        "Affluent VIP Executive",
        "Affluent Cautious Saver",
        "Young Trendsetter Impulse Shopper",
        "Frugal Budget Shopper",
        "Core Mainstream Family",
    ],
    index=0,
)

PRESETS = {
    "Affluent VIP Executive": {"Age": 38, "Annual_Income_k": 95.0, "Spending_Score": 85.0, "Purchase_Frequency": 32.0, "Average_Basket_Value": 210.0},
    "Affluent Cautious Saver": {"Age": 52, "Annual_Income_k": 92.0, "Spending_Score": 22.0, "Purchase_Frequency": 8.0, "Average_Basket_Value": 160.0},
    "Young Trendsetter Impulse Shopper": {"Age": 24, "Annual_Income_k": 32.0, "Spending_Score": 85.0, "Purchase_Frequency": 24.0, "Average_Basket_Value": 85.0},
    "Frugal Budget Shopper": {"Age": 45, "Annual_Income_k": 28.0, "Spending_Score": 18.0, "Purchase_Frequency": 5.0, "Average_Basket_Value": 45.0},
    "Core Mainstream Family": {"Age": 41, "Annual_Income_k": 58.0, "Spending_Score": 50.0, "Purchase_Frequency": 16.0, "Average_Basket_Value": 110.0},
}

selected_preset = PRESETS.get(preset_choice, {})

st.sidebar.markdown("---")
age = st.sidebar.slider("Age", 18, 70, int(selected_preset.get("Age", 35)))
income = st.sidebar.slider("Annual Income ($k)", 15.0, 150.0, float(selected_preset.get("Annual_Income_k", 60.0)), 1.0)
spending = st.sidebar.slider("Spending Score (1-100)", 1.0, 100.0, float(selected_preset.get("Spending_Score", 50.0)), 1.0)
freq = st.sidebar.slider("Annual Purchase Frequency (Orders/Year)", 1.0, 50.0, float(selected_preset.get("Purchase_Frequency", 15.0)), 1.0)
basket = st.sidebar.slider("Average Basket Value ($)", 15.0, 350.0, float(selected_preset.get("Average_Basket_Value", 110.0)), 5.0)

# Main Application Title
st.title("🛍️ Retail Customer Segmentation & Persona Intelligence")
st.markdown(
    "Unsupervised machine learning system using K-Means, Agglomerative Hierarchical Clustering, and DBSCAN. "
    "Features automated Elbow and Silhouette optimization, 2D PCA latent projections, and tailored marketing playbooks."
)

segmenter = load_or_train_segmenter()

# Customer payload
customer_dict = {
    "Age": age,
    "Annual_Income_k": income,
    "Spending_Score": spending,
    "Purchase_Frequency": freq,
    "Average_Basket_Value": basket,
}

result = segmenter.score_single_customer(customer_dict)
cid = result["cluster_id"]
persona = result["persona_name"]
share = result["customer_share_pct"]
dist = result["prototypical_distance"]
desc = result["description"]
strategy = result["marketing_strategy"]

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Persona Profiler",
    "📐 Optimal K (Elbow & Silhouette)",
    "🗺️ Latent Space (2D PCA)",
    "📊 Segment Heatmap & Benchmarks",
    "📂 Customer Data Explorer",
])

with tab1:
    st.subheader("Customer Persona Assignment")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**Assigned Cluster**")
        st.markdown(f"<h2 style='color:#2980b9; margin-top:-10px;'>Segment C{cid}</h2>", unsafe_allow_html=True)
    with col2:
        st.markdown("**Persona Classification**")
        st.markdown(f"<h3 style='color:#27ae60; margin-top:-10px;'>{persona}</h3>", unsafe_allow_html=True)
    with col3:
        st.metric(label="Customer Base Share", value=f"{share:.1f}%")
    with col4:
        fit_label = "High Archetype" if dist < 1.0 else ("Moderate Fit" if dist < 2.0 else "Boundary Profile")
        st.metric(label="Centroid Distance", value=f"{dist:.3f}", delta=fit_label, delta_color="off")

    st.markdown("---")
    c_desc, c_strat = st.columns(2)
    with c_desc:
        st.info(f"**Behavioral Persona Profile:**\n\n{desc}")
    with c_strat:
        st.success(f"**Targeted Marketing & Retention Playbook:**\n\n{strategy}")

    st.markdown("---")
    st.subheader("Customer Attribute Breakdown vs. Cluster Centroid")
    p_info = segmenter.profiles.get(cid, {})
    mean_attrs = p_info.get("mean_attributes", {})

    attr_df = pd.DataFrame({
        "Attribute": ["Age", "Annual Income ($k)", "Spending Score", "Purchase Frequency", "Avg Basket Value ($)"],
        "Customer Value": [f"{age} yrs", f"${income:.1f}k", f"{spending:.1f}", f"{freq:.0f} orders/yr", f"${basket:.2f}"],
        "Cluster Mean": [
            f"{mean_attrs.get('Age', 0):.1f} yrs",
            f"${mean_attrs.get('Annual_Income_k', 0):.1f}k",
            f"{mean_attrs.get('Spending_Score', 0):.1f}",
            f"{mean_attrs.get('Purchase_Frequency', 0):.0f} orders/yr",
            f"${mean_attrs.get('Average_Basket_Value', 0):.2f}",
        ],
    })
    st.dataframe(attr_df, width="stretch")

with tab2:
    st.subheader("Optimal Cluster Selection: Elbow Method & Silhouette Analysis")
    st.markdown(
        """
        - **Elbow Method (Inertia / WCSS)**: Measures within-cluster sum of squares. As $k$ increases, inertia drops monotonically. The optimal $k$ corresponds to the inflection 'elbow' point where marginal variance reduction slows down.
        - **Silhouette Score**: Measures intra-cluster cohesion $a(i)$ relative to nearest neighbor cluster separation $b(i)$:
          $$\\text{Silhouette} = \\frac{b(i) - a(i)}{\\max(a(i), b(i))}$$
          Scores range from $-1$ (misclassified) to $+1$ (dense, well-separated clusters). Peak score indicates the most natural cluster count.
        """
    )

    _, sweep_df, _ = load_cached_clustering_data()
    st.dataframe(sweep_df, width="stretch")

    vis_dir = PROJECT_ROOT / "visualizations"
    p_elbow = vis_dir / "elbow_silhouette_analysis.png"
    if p_elbow.exists():
        st.image(str(p_elbow), caption="Elbow & Silhouette Dual-Axis Curve across k in [2, 10]", width="stretch")

with tab3:
    st.subheader("Latent Cluster Coordinates (2D PCA)")
    st.markdown(
        "Dimensionality reduction projects high-dimensional customer behaviors onto the two principal components "
        "capturing the vast majority (>85%) of total behavioral variance."
    )

    p_pca = vis_dir / "cluster_scatterplot_pca.png"
    if p_pca.exists():
        st.image(str(p_pca), caption="Customer Clusters in 2D PCA Latent Space with Centroids", width="stretch")

with tab4:
    st.subheader("Clustering Algorithm Benchmarks & Persona Heatmap")
    _, _, eval_df = load_cached_clustering_data()

    st.markdown("#### Multi-Algorithm Benchmark")
    st.dataframe(eval_df, width="stretch")

    col_h1, col_h2 = st.columns(2)
    p_heat = vis_dir / "cluster_profiles_heatmap.png"
    p_algo = vis_dir / "algorithm_comparison.png"
    if p_heat.exists():
        with col_h1:
            st.image(str(p_heat), caption="Normalized Feature Heatmap by Customer Segment", width="stretch")
    if p_algo.exists():
        with col_h2:
            st.image(str(p_algo), caption="Algorithm Comparison (K-Means vs Hierarchical vs DBSCAN)", width="stretch")

with tab5:
    st.subheader("Retail Customer Dataset Explorer")
    clean_df, _, _ = load_cached_clustering_data()
    st.markdown(f"**Total Customer Records:** `{len(clean_df):,}` | **Features Tracked:** `5`")
    st.dataframe(clean_df.head(50), width="stretch")
