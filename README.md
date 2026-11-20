# Machine Learning Engineering Portfolio

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B.svg?style=for-the-badge&logo=streamlit)](https://vamsikrishna4244-machine-learning-app-ncsihn.streamlit.app)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 🚀 **Live Interactive App**: Try the project live in your browser: **[https://vamsikrishna4244-machine-learning-app-ncsihn.streamlit.app](https://vamsikrishna4244-machine-learning-app-ncsihn.streamlit.app)**

A curated repository of production-grade Machine Learning projects demonstrating rigorous modeling, leak-free pipelines, cross-validation, metric-driven evaluation, and interactive web deployments.

---

## Repository Projects

| Project | Domain & Task | Models / Techniques | Key Metrics | Live Demo |
| :--- | :--- | :--- | :--- | :---: |
| [**01. House Price Prediction System**](./01-house-price-prediction/) | Real Estate Valuation (Supervised Regression) | Linear Regression, Decision Tree, Random Forest | $R^2 = 0.796$, $\text{RMSE} = \$36,547$ | [🚀 Launch App](https://vamsikrishna4244-machine-learning-app-ncsihn.streamlit.app) |
| [**02. Customer Churn Prediction System**](./02-customer-churn-prediction/) | Telecom & SaaS Retention (Binary Classification) | Logistic Regression, Decision Tree, Random Forest | $\text{Recall} = 76.0\%$, $\text{ROC-AUC} = 0.835$ | [🚀 Launch App](https://vamsikrishna-customer-churn.streamlit.app) |

---

## Project Overviews

### Project 01: House Price Prediction System
The **House Price Prediction System** is an end-to-end regression application that models residential property valuations based on structural and geographic characteristics.
- **Leak-Free Preprocessing**: Scikit-learn `Pipeline` and `ColumnTransformer` with `StandardScaler` and `OneHotEncoder`.
- **Comprehensive Benchmarking**: Comparative analysis of Linear Regression, Decision Tree Regressor, and Random Forest Regressor.
- **Residual Diagnostics**: Distribution plots, Q-Q analysis, and heteroskedasticity checks.
- **Interactive Web App**: A production-ready Streamlit dashboard featuring live property estimation, model comparisons, and interactive Plotly diagnostics.
- **Robust Quality**: 20/20 unit tests passing with 95% code coverage.

### Project 02: Customer Churn Prediction & Retention System
The **Customer Churn Prediction System** is an enterprise binary classification system designed to detect high-flight-risk subscribers before service cancellation.
- **Imbalance Handling**: Balanced class weights and threshold tuning to maximize retention recall (76.0%).
- **Prescriptive Retention Playbooks**: Maps churn risk tiers to high-impact retention interventions (contract upgrades, tech support check-ins).
- **Metric-Driven Evaluation**: Evaluated via PR-AUC (0.654), ROC-AUC (0.835), F1-Score, and false positive/negative cost trade-offs.
- **Interactive Dashboard**: Streamlit app with subscriber risk simulator, probability threshold sliders, and confusion matrix diagnostics.
- **Robust Quality**: 24/24 unit tests passing with 96% code coverage.

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
