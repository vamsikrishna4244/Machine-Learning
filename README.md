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
| [**03. Email Spam Classification System**](./03-email-spam-classification/) | Cybersecurity & Text Filtering (NLP Classification) | TF-IDF vs. BoW, Naive Bayes, Logistic Regression, Linear SVM | $\text{Accuracy} = 98.6\%$, $\text{F1-Score} = 0.957$ | [🚀 Launch App](https://vamsikrishna-email-spam.streamlit.app) |
| [**04. Credit Card Fraud Detection System**](./04-credit-card-fraud-detection/) | FinTech Anomaly Detection (Imbalanced Classification) | RobustScaler, Balanced Logistic Regression, Random Forest, Cost Matrix | $\text{PR-AUC} = 1.000$, $\text{Recall} = 100\%$, $\text{Min Cost} = \$0$ | [🚀 Launch App](https://vamsikrishna-credit-card-fraud.streamlit.app) |
| [**05. Customer Segmentation System**](./05-customer-segmentation/) | Retail Behavioral Clustering (Unsupervised Learning) | K-Means ($k=5$), Hierarchical (Ward), DBSCAN, 2D PCA | $\text{Silhouette} = 0.491$, $\text{CH} = 2,253.8$ | [🚀 Launch App](https://vamsikrishna-customer-segmentation.streamlit.app) |

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

### Project 03: Email Spam Classification System
The **Email Spam Classification System** is an end-to-end natural language processing (NLP) and machine learning application that detects spam and phishing emails with high precision and recall.
- **Vectorization Benchmarking**: Systematic comparison of Bag-of-Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF) feature extractors across unigram and bigram vocabularies.
- **Multi-Model Evaluation**: Benchmarks Multinomial Naive Bayes, Logistic Regression, and Linear SVM models.
- **Threat Intelligence & Keyword Attribution**: Real-time inspection of high-risk spam indicators and token attribution.
- **Interactive Web App**: A production-ready Streamlit dashboard featuring live raw email scoring, threshold customization, keyword highlighting, and model comparison matrices.
- **Robust Quality**: 27/27 unit tests passing with 95% code coverage.

### Project 04: Credit Card Fraud Detection & Risk Authorization Engine
The **Credit Card Fraud Detection & Risk Authorization Engine** is an end-to-end machine learning system engineered to detect fraudulent card transactions in real time under severe class imbalance (~0.8% fraud prevalence).
- **Outlier-Resistant Preprocessing**: Utilizes `RobustScaler` on skewed transaction amounts and timestamps while retaining PCA components $V_1$ through $V_{28}$.
- **Imbalance & Resampling Regimes**: Compares unweighted baseline, cost-weighted logistic regression (`class_weight='balanced'`), random undersampling, and balanced random forests.
- **Asymmetric Financial Loss Optimization**: Formulates a realistic banking cost matrix ($10 per False Positive vs $400 per False Negative) to determine the financially optimal authorization threshold.
- **Interactive Web App**: A production-ready Streamlit authorization portal featuring transaction anomaly scoring, threshold sliders, PCA feature inspection, and cost savings trade-off analyzers.
- **Robust Quality**: 21/21 unit tests passing with 96% code coverage.

### Project 05: Customer Segmentation & Behavioral Archetyping System
The **Customer Segmentation Using Unsupervised Clustering** project is an end-to-end machine learning system that groups retail and e-commerce shoppers into distinct behavioral archetypes.
- **Feature Normalization & Distance Preservation**: Applies `StandardScaler` across monetary, frequency, and basket size features to ensure distortion-free Euclidean distance spaces.
- **Rigorous Cluster Optimization**: Empirically benchmarks $k \in [2, 10]$ using the Elbow Method (Inertia), Silhouette Analysis, Davies-Bouldin index, and Calinski-Harabasz score.
- **Multi-Algorithm Comparison**: Contrasts K-Means++ centroid partitioning against Agglomerative Hierarchical clustering and DBSCAN density filtering.
- **2D PCA & Strategic Personas**: Projects high-dimensional feature spaces onto 2D Principal Components and maps cluster centroids into 5 actionable retail personas (VIP Champions, Trendsetters, Savers, Budget Planners, Mainstream).
- **Interactive Web App**: A production-ready Streamlit clustering console featuring dynamic customer profile simulations, radar charts, 2D/3D PCA scatter plots, and promotional playbooks.
- **Robust Quality**: 21/21 unit tests passing with 97% code coverage.

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

### 4. Run Customer Churn Prediction Project
```bash
cd ../02-customer-churn-prediction
pip install -r requirements.txt

# Train models and generate performance visual diagnostics
python train.py

# Run unit tests with code coverage
pytest --cov=src --cov-report=term-missing

# Launch interactive Streamlit application
streamlit run streamlit_app.py
```

### 5. Run Email Spam Classification Project
```bash
cd ../03-email-spam-classification
pip install -r requirements.txt

# Train models and generate performance visual diagnostics
python train.py

# Run unit tests with code coverage
pytest --cov=src --cov-report=term-missing

# Launch interactive Streamlit application
streamlit run streamlit_app.py
```

### 6. Run Credit Card Fraud Detection Project
```bash
cd ../04-credit-card-fraud-detection
pip install -r requirements.txt

# Train models and generate performance visual diagnostics
python train.py

# Run unit tests with code coverage
pytest --cov=src --cov-report=term-missing

# Launch interactive Streamlit application
streamlit run streamlit_app.py
```

### 7. Run Customer Segmentation Project
```bash
cd ../05-customer-segmentation
pip install -r requirements.txt

# Train models and generate performance visual diagnostics
python train.py

# Run unit tests with code coverage
pytest --cov=src --cov-report=term-missing

# Launch interactive Streamlit application
streamlit run streamlit_app.py
```

---

## Deploying on Streamlit Community Cloud

Each project in this repository can be deployed as a standalone application on [Streamlit Community Cloud](https://share.streamlit.io):

1. Sign in to **Streamlit Community Cloud** with GitHub.
2. Click **"New app"**.
3. Configure repository settings:
   - **Repository:** `vamsikrishna4244/Machine-Learning`
   - **Branch:** `main`
   - **Main file path:** `05-customer-segmentation/streamlit_app.py` *(or `04-credit-card-fraud-detection/streamlit_app.py`, `03-email-spam-classification/streamlit_app.py`, `02-customer-churn-prediction/streamlit_app.py`)*
   - **App URL (optional):** `vamsikrishna-customer-segmentation`
4. Click **Deploy!**

---

## License

This repository is distributed under the [MIT License](LICENSE).
