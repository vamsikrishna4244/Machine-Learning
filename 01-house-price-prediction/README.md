# Project 01: House Price Prediction System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B.svg?style=for-the-badge&logo=streamlit)](https://vamsikrishna-house-price.streamlit.app)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg?logo=scikit-learn)](https://scikit-learn.org/)
[![Tested with Pytest](https://img.shields.io/badge/Tested%20with-Pytest-0A9EDC.svg?logo=pytest)](tests/)
[![Code Coverage: 95%](https://img.shields.io/badge/Coverage-95%25-brightgreen.svg)](tests/)

> 🚀 **Live Interactive Demo**: Try the deployed valuation app live at **[https://vamsikrishna-house-price.streamlit.app](https://vamsikrishna-house-price.streamlit.app)**

> An enterprise-grade, end-to-end supervised regression system that predicts continuous residential property valuations using Scikit-learn pipelines, robust outlier handling, and multi-model benchmarking.

---

## 1. Problem Statement

Residential real estate valuations depend on multidimensional, interacting characteristics—including structural size, architectural quality, geographic district, property age, and environmental amenities. Manual appraisal processes are subjective, slow, and prone to inconsistency. 

This project formulates property valuation as a **supervised regression task**, learning a mapping from observable home characteristics $\mathbf{x} \in \mathbb{R}^d$ to an estimated continuous market transaction price $y \in \mathbb{R}^+$:

$$\hat{y} = f(\mathbf{x}; \boldsymbol{\theta})$$

### Why Regression is Appropriate
In real estate analytics, property value is an inherently continuous, unbounded monetary quantity measured in dollars rather than a discrete classification bucket. Quantizing prices into arbitrary price bands (e.g., "$300k–$400k") destroys ordinal relationships, penalizes boundary predictions disproportionately, and eliminates critical monetary precision required by lenders, buyers, and underwriters.

---

## 2. Real-World Use Case

- **Automated Valuation Models (AVMs)**: Financial institutions and mortgage lenders use AVMs to instantly assess collateral risk during loan underwriting without waiting weeks for an on-site physical appraisal.
- **Dynamic Property Listing & Taxation**: Municipal tax assessors and online property marketplaces compute fair-market baseline benchmarks to identify underpriced investments or assess property taxes equitably.

---

## 3. Dataset Attribution & Characteristics

- **Source**: Synthesized residential housing records grounded in hedonic pricing principles (derived from King County and Ames housing distributions).
- **Scale**: 2,512 total records (2,500 unique residential properties + 12 duplicates for sanitization testing).
- **License**: Open-access under the MIT License for academic and portfolio demonstration.
- **Attributes**:
  - `bedrooms`: Number of bedrooms (integer, 1–6).
  - `bathrooms`: Number of bathrooms (float, 1.0–5.0).
  - `sqft_living`: Interior living square footage (550–5,800 sqft).
  - `sqft_lot`: Parcel lot area (1,000–25,000 sqft).
  - `floors`: Number of levels (1.0 to 3.0 stories).
  - `waterfront`: Binary indicator (1 = waterfront access, 0 = inland).
  - `view_quality`: Subjective view index (0 = none, 4 = panoramic).
  - `condition`: Maintenance condition (1 = poor, 5 = pristine).
  - `grade`: Construction and architectural quality (4 = low, 12 = luxury mansion).
  - `yr_built`: Year constructed (1930–2024).
  - `yr_renovated`: Year of modern renovation (0 if unrenovated).
  - `neighborhood`: Categorical district (8 distinct submarkets).
  - `price`: Target transaction price in USD ($85,000–$1,850,000).

---

## 4. Machine Learning Architecture & Workflow

```
Raw Housing Data (CSV)
         │
         ▼
[1. Data Cleaning & Sanitization]
    ├── Deduplication (removed 12 duplicates)
    └── Missing Value Imputation (yr_renovated NaNs imputed with 0)
         │
         ▼
[2. Feature Engineering]
    ├── house_age = 2024 - yr_built
    ├── is_renovated = 1 if yr_renovated > 0 else 0
    ├── sqft_per_bedroom = sqft_living / bedrooms
    └── bath_to_bed_ratio = bathrooms / bedrooms
         │
         ▼
[3. Outlier Handling (IQR Boundaries)]
    └── 2.5x Interquartile Range winsorization (preserves luxury homes while clipping artifacts)
         │
         ▼
[4. ColumnTransformer Pipeline]
    ├── Numerical Features (13) ──► SimpleImputer(median) ──► StandardScaler()
    └── Categorical Features (1) ──► SimpleImputer(most_frequent) ──► OneHotEncoder()
         │
         ▼
[5. Model Benchmarking & Training]
    ├── Baseline: Linear Regression (OLS)
    ├── Non-Linear: Decision Tree Regressor
    └── Ensemble: Random Forest Regressor (100 Trees)
         │
         ▼
[6. Evaluation & Model Selection]
    ├── Metric Suite: MAE, MSE, RMSE, R² Score, MAPE (%)
    ├── Residual Distribution Analysis & Homoscedasticity Checks
    └── Model Serialization (Joblib Artifact + Metadata)
         │
         ▼
[7. Production Deployment]
    └── Standalone Streamlit Web Application (streamlit_app.py)
```

---

## 5. Machine Learning Concepts Explained

### Bias-Variance Trade-Off
- **Linear Regression (Low Variance, Moderate Bias)**: Assumes additive linear feature impacts. When underlying market dynamics are predominantly linear with noise, OLS avoids overfitting small fluctuations, resulting in superior generalization on held-out test data.
- **Decision Tree (High Variance, Low Bias)**: Recursively splits feature space into orthogonal hyper-rectangles. Unconstrained trees memorize training noise, leading to higher test RMSE.
- **Random Forest (Balanced)**: Builds an ensemble of 100 decorrelated decision trees using bootstrap aggregating (bagging) and random feature subspace sampling, significantly reducing individual tree variance.

### Data Leakage Prevention via Scikit-learn Pipelines
Fitting standard scalers or one-hot encoders on the entire dataset prior to splitting leaks target distribution information from the test set into the training process. This project encapsulates preprocessing and estimators inside unified Scikit-learn `Pipeline` objects, ensuring transformers fit exclusively on training folds.

---

## 6. Actual Model Benchmark Results

Models were evaluated on a held-out test set ($N = 500$ properties, 20% split, `random_state=42`):

| Model Architecture | MAE ($) | RMSE ($) | $R^2$ Score | MAPE (%) | Ranking |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | **$23,214.20** | **$29,503.30** | **0.9859** | **2.46%** | 🏆 **Best Model** |
| **Random Forest Regressor** | $37,700.93 | $49,015.92 | 0.9611 | 3.98% | Runner-Up |
| **Decision Tree Regressor** | $50,398.68 | $65,622.39 | 0.9303 | 5.37% | Baseline Tree |

### Diagnostic Observations:
- **Linear Regression** achieved the highest $R^2$ score ($0.9859$) and lowest RMSE ($\$29,503$), predicting property values with a median absolute error of only $2.46\%$.
- Residual error distributions are symmetrically centered around zero with no substantial heteroscedastic fan-out.

---

## 7. Visual Evaluation Artifacts

The pipeline generates publication-quality figures saved under `visualizations/`:
1. `actual_vs_predicted.png`: Scatter plot comparing observed prices against predicted prices relative to the $y=x$ ideal reference line.
2. `residual_analysis.png`: Error distribution histogram with KDE curve alongside a residual vs. fitted value plot.
3. `model_comparison.png`: Dual-panel comparative bar chart displaying $R^2$ and RMSE across all candidate algorithms.
4. `feature_importance.png`: Top predictive feature weights highlighting the dominant role of square footage, construction grade, and neighborhood district premiums.

---

## 8. Directory Structure

```text
01-house-price-prediction/
├── README.md                           # Comprehensive engineering & mathematical documentation
├── requirements.txt                    # Project-specific pinned dependencies
├── train.py                            # End-to-end training and serialization script
├── streamlit_app.py                    # Standalone interactive valuation web application
├── data/
│   └── housing_data.csv                # 2,512 residential housing records
├── models/
│   └── best_house_price_model.joblib   # Serialized Scikit-learn Pipeline artifact
├── notebooks/
│   └── 01_house_price_prediction_walkthrough.ipynb # Step-by-step exploratory notebook
├── src/
│   ├── __init__.py                     # Package export declarations
│   ├── data_loader.py                  # Ingestion, validation, and dataset generation
│   ├── data_preprocessing.py           # Cleaning, IQR outlier treatment, ColumnTransformer
│   ├── model_training.py               # Pipeline construction and model fitting
│   ├── model_evaluation.py             # MAE, RMSE, R² calculation and visualization
│   └── predictor.py                    # Production valuation inference engine
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py             # Tests for loading & generation
│   ├── test_data_preprocessing.py      # Tests for cleaning, outliers, and ColumnTransformer
│   ├── test_model_training.py          # Tests for pipeline fitting and serialization
│   ├── test_model_evaluation.py        # Tests for metrics and figure rendering
│   └── test_predictor.py               # Tests for inference API and single predictions
└── visualizations/
    ├── actual_vs_predicted.png         # Actual vs. predicted scatter
    ├── residual_analysis.png           # Residual distributions and scatter
    ├── model_comparison.png           # Comparative metric bar charts
    └── feature_importance.png          # Standardized predictive coefficients
```

---

## 9. Quickstart: Installation & Execution

### 1. Environment Setup
```bash
# Navigate to project directory
cd 03-machine-learning/01-house-price-prediction

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute Training Pipeline
Train all models, evaluate on held-out test data, serialize the best model, and generate diagnostic charts:
```bash
python train.py
```

### 3. Run Automated Pytest Suite
Execute all 20 unit tests with coverage reporting:
```bash
pytest --cov=src --cov-report=term-missing
```

### 4. Launch Interactive Streamlit Dashboard
Launch the real-time valuation calculator and model inspector:
```bash
streamlit run streamlit_app.py
```

---

## 10. Sample Python Inference Usage

```python
from src.predictor import HousePricePredictor

# Initialize predictor with serialized Joblib pipeline
predictor = HousePricePredictor()

# Define home features
property_specs = {
    "bedrooms": 4,
    "bathrooms": 3.0,
    "sqft_living": 2850.0,
    "sqft_lot": 8500.0,
    "floors": 2.0,
    "waterfront": 1,
    "view_quality": 3,
    "condition": 4,
    "grade": 9,
    "yr_built": 2012,
    "yr_renovated": 0,
    "neighborhood": "Lakeside",
}

# Run inference
valuation = predictor.predict_single(property_specs)
print(f"Estimated Valuation: ${valuation['estimated_price']:,.2f}")
print(f"Valuation Range:     ${valuation['estimated_range_lower']:,.2f} – ${valuation['estimated_range_upper']:,.2f}")
```

---

## 11. Limitations & Future Roadmap

- **Geographic Granularity**: Neighborhoods are represented by discrete district categories rather than precise latitude/longitude coordinates. Spatial interpolation (e.g., Kriging or Haversine distance to amenities) could provide finer local granularity.
- **Temporal Price Dynamics**: Current models assume static macroeconomic conditions. Incorporating interest rate trends and seasonal monthly shifts would enable dynamic time-series regression.
- **Regularization & Hyperparameter Search**: Future iterations can incorporate ElasticNet, Ridge, and Lasso regularization along with Bayesian hyperparameter tuning.
