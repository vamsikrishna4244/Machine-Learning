"""
Interactive Streamlit Application: Residential Property Valuation Engine.

Allows interactive specification of real estate attributes, live model inference,
model evaluation comparisons, and residual diagnostic visualizations.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import NEIGHBORHOODS, load_housing_data
from src.predictor import HousePricePredictor


@st.cache_resource
def get_predictor() -> HousePricePredictor:
    """Load and cache the trained house price predictor model."""
    return HousePricePredictor()


@st.cache_data
def get_sample_dataset() -> pd.DataFrame:
    """Load sample housing records for inspection."""
    return load_housing_data()


def main() -> None:
    st.set_page_config(
        page_title="House Price Prediction Engine | ML",
        page_icon="🏡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header Banner
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 1.5rem 2rem; border-radius: 12px; color: white; margin-bottom: 1.5rem;">
            <h1 style="margin: 0; font-size: 2rem;">🏡 Residential Property Valuation Engine</h1>
            <p style="margin: 0.3rem 0 0 0; opacity: 0.95; font-size: 1rem;">
                Supervised Regression • Scikit-learn Pipeline • Hedonic Feature Pricing • Residual Diagnostics
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    predictor = get_predictor()
    raw_df = get_sample_dataset()

    # Sidebar Property Controls
    st.sidebar.markdown("### 🛠️ Property Specifications")
    neighborhood = st.sidebar.selectbox("Neighborhood / District", options=NEIGHBORHOODS, index=3)
    bedrooms = st.sidebar.slider("Bedrooms", min_value=1, max_value=6, value=3)
    bathrooms = st.sidebar.slider("Bathrooms", min_value=1.0, max_value=5.0, value=2.5, step=0.25)
    sqft_living = st.sidebar.slider("Living Area (sqft)", min_value=600, max_value=5500, value=2200, step=50)
    sqft_lot = st.sidebar.slider("Lot Size (sqft)", min_value=1000, max_value=25000, value=7500, step=250)
    floors = st.sidebar.selectbox("Floors / Stories", options=[1.0, 1.5, 2.0, 2.5, 3.0], index=2)
    grade = st.sidebar.slider("Construction Grade (4=Basic, 12=Luxury)", min_value=4, max_value=12, value=8)
    condition = st.sidebar.slider("Condition Rating (1=Poor, 5=Mint)", min_value=1, max_value=5, value=4)
    view_quality = st.sidebar.slider("View Quality Rating (0=None, 4=Panoramic)", min_value=0, max_value=4, value=1)
    waterfront = st.sidebar.checkbox("Waterfront Property", value=False)
    yr_built = st.sidebar.slider("Year Built", min_value=1930, max_value=2024, value=2008)
    has_renovated = st.sidebar.checkbox("Property Renovated", value=False)
    yr_renovated = st.sidebar.slider("Year Renovated", min_value=yr_built, max_value=2024, value=2018) if has_renovated else 0

    # User Input Feature Dict
    property_input = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": float(sqft_living),
        "sqft_lot": float(sqft_lot),
        "floors": float(floors),
        "waterfront": 1 if waterfront else 0,
        "view_quality": view_quality,
        "condition": condition,
        "grade": grade,
        "yr_built": yr_built,
        "yr_renovated": yr_renovated,
        "neighborhood": neighborhood,
    }

    valuation = predictor.predict_single(property_input)

    # Top KPI Cards
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.metric(
            label="Estimated Market Value",
            value=f"${valuation['estimated_price']:,.0f}",
            help="Predicted point estimate using the serialized Scikit-learn Pipeline.",
        )
    with col_k2:
        st.metric(
            label="Expected Valuation Range",
            value=f"${valuation['estimated_range_lower']:,.0f} – ${valuation['estimated_range_upper']:,.0f}",
            help="± 1x RMSE uncertainty interval.",
        )
    with col_k3:
        st.metric(
            label="Model R² Score",
            value=f"{valuation['model_r2_score']:.4f}",
            help="Coefficient of determination on held-out test data.",
        )
    with col_k4:
        st.metric(
            label="Test Set MAE",
            value=f"${valuation['model_mae']:,.0f}",
            help="Mean Absolute Error on held-out test data.",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Tab Navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏷️ Property Valuation Breakdown",
        "📊 Model Comparison & Benchmarks",
        "🎯 Residuals & Error Analysis",
        "📋 Housing Data Inspector",
    ])

    # Tab 1: Valuation Breakdown
    with tab1:
        st.subheader("Hedonic Valuation Breakdown & Regional Context")
        col_v1, col_v2 = st.columns([1, 1])

        with col_v1:
            st.markdown("#### Specified Property Summary")
            summary_table = pd.DataFrame([
                {"Attribute": "Neighborhood", "Value": str(neighborhood)},
                {"Attribute": "Living Area", "Value": f"{sqft_living:,} sqft"},
                {"Attribute": "Bedrooms / Bathrooms", "Value": f"{bedrooms} beds / {bathrooms} baths"},
                {"Attribute": "Lot Size", "Value": f"{sqft_lot:,} sqft"},
                {"Attribute": "Construction Grade", "Value": f"{grade} / 12"},
                {"Attribute": "Condition Rating", "Value": f"{condition} / 5"},
                {"Attribute": "Property Age", "Value": f"{2024 - yr_built} years (Built {yr_built})"},
                {"Attribute": "Waterfront / View", "Value": f"{'Yes' if waterfront else 'No'} / Level {view_quality}"},
            ])
            st.table(summary_table)

        with col_v2:
            st.markdown("#### Neighborhood Price Benchmarks")
            neigh_stats = raw_df.groupby("neighborhood")["price"].mean().reset_index()
            neigh_stats.columns = ["Neighborhood", "Average Price"]
            neigh_stats = neigh_stats.sort_values(by="Average Price", ascending=True)

            fig_bar = px.bar(
                neigh_stats,
                x="Average Price",
                y="Neighborhood",
                orientation="h",
                title="Average Property Price by Neighborhood",
                color="Average Price",
                color_continuous_scale="Blues",
            )
            # Add user property marker
            fig_bar.add_vline(
                x=valuation["estimated_price"],
                line_dash="dash",
                line_color="red",
                annotation_text=f"This Property: ${valuation['estimated_price']:,.0f}",
            )
            fig_bar.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig_bar, width="stretch")

    # Tab 2: Model Comparison
    with tab2:
        st.subheader("Candidate Model Benchmark on Held-Out Test Set (N = 500)")
        st.markdown(
            """
            Three algorithms were trained on identical training folds using Scikit-learn Pipelines:
            - **Linear Regression (Baseline)**: Parametric OLS regression.
            - **Decision Tree Regressor**: Non-linear tree partition regressor.
            - **Random Forest Regressor**: Ensemble of 100 decorrelated decision trees.
            """
        )

        comparison_data = pd.DataFrame([
            {"Model": "Linear Regression", "MAE ($)": 23214.20, "RMSE ($)": 29503.30, "R² Score": 0.9859, "MAPE (%)": 2.46},
            {"Model": "Random Forest", "MAE ($)": 37700.93, "RMSE ($)": 49015.92, "R² Score": 0.9611, "MAPE (%)": 3.98},
            {"Model": "Decision Tree", "MAE ($)": 50398.68, "RMSE ($)": 65622.39, "R² Score": 0.9303, "MAPE (%)": 5.37},
        ])
        st.dataframe(comparison_data, width="stretch", hide_index=True)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            fig_r2 = px.bar(
                comparison_data,
                x="Model",
                y="R² Score",
                color="Model",
                title="R² Score Comparison (Higher is Better)",
                text_auto=".3f",
            )
            fig_r2.update_layout(template="plotly_white", yaxis_range=[0.85, 1.0], showlegend=False)
            st.plotly_chart(fig_r2, width="stretch")

        with col_m2:
            fig_rmse = px.bar(
                comparison_data,
                x="Model",
                y="RMSE ($)",
                color="Model",
                title="Root Mean Squared Error (Lower is Better)",
                text_auto="$,.0f",
            )
            fig_rmse.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig_rmse, width="stretch")

    # Tab 3: Residuals & Error Analysis
    with tab3:
        st.subheader("Model Diagnostic & Residual Distributions")
        col_res1, col_res2 = st.columns(2)

        vis_dir = PROJECT_ROOT / "visualizations"
        with col_res1:
            p_act = vis_dir / "actual_vs_predicted.png"
            if p_act.exists():
                st.image(str(p_act), caption="Actual vs. Predicted House Prices", width="stretch")
            else:
                st.info("Run `python train.py` to generate visual diagnostics.")

        with col_res2:
            p_res = vis_dir / "residual_analysis.png"
            if p_res.exists():
                st.image(str(p_res), caption="Residual Error Distribution & Residual Plot", width="stretch")
            else:
                st.info("Run `python train.py` to generate visual diagnostics.")

        p_feat = vis_dir / "feature_importance.png"
        if p_feat.exists():
            st.image(str(p_feat), caption="Top Predictive Feature Importances", width="stretch")

    # Tab 4: Dataset Inspector
    with tab4:
        st.subheader("Housing Dataset Inspection")
        st.markdown(f"Total Records: **{len(raw_df):,}** residential homes.")
        st.dataframe(raw_df.head(100), width="stretch")
        csv_bytes = raw_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Housing Dataset (CSV)",
            data=csv_bytes,
            file_name="housing_data.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
