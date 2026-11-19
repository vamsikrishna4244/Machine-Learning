# Machine Learning Engineering Portfolio

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=vamsikrishna4244/Machine-Learning&branch=main&mainModule=app.py)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg?logo=streamlit)](https://streamlit.io/)

A curated repository of production-grade Machine Learning projects demonstrating rigorous modeling, leak-free pipelines, cross-validation, metric-driven evaluation, and interactive web deployments.

---

## Repository Projects

| Project | Domain & Task | Models / Techniques | Key Metrics | Status |
| :--- | :--- | :--- | :--- | :---: |
| [**01. House Price Prediction System**](./01-house-price-prediction/) | Real Estate Valuation (Supervised Regression) | Linear Regression, Decision Tree, Random Forest | $R^2 = 0.796$, $\text{RMSE} = \$36,547$, $\text{MAE} = \$27,330$ | Complete |

---

## Project 01: House Price Prediction System Overview

The **House Price Prediction System** is an end-to-end regression application that models residential property valuations based on structural and geographic characteristics.

### Key Highlights
- **Leak-Free Preprocessing**: Scikit-learn `Pipeline` and `ColumnTransformer` with `StandardScaler` and `OneHotEncoder`.
- **Comprehensive Benchmarking**: Comparative analysis of Linear Regression, Decision Tree Regressor, and Random Forest Regressor.
- **Residual Diagnostics**: Distribution plots, Q-Q analysis, and heteroskedasticity checks.
- **Interactive Web App**: A production-ready Streamlit dashboard featuring live property estimation, model comparisons, and interactive Plotly diagnostics.
- **Robust Quality**: 20/20 unit tests passing with 95% code coverage.

---

## Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/vamsikrishna4244/Machine-Learning.git
cd Machine-Learning
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Run House Price Prediction Project
```bash
cd 01-house-price-prediction
pip install -r requirements.txt

# Train models and generate performance visual diagnostics
python train.py

# Run unit tests with code coverage
pytest --cov=src --cov-report=term-missing

# Launch interactive Streamlit application
streamlit run streamlit_app.py
```

---

## License

This repository is distributed under the [MIT License](LICENSE).
