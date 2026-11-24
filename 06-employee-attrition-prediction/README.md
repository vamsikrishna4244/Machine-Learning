# Project 06: End-to-End Enterprise Employee Attrition Prediction System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B.svg?style=for-the-badge&logo=streamlit)](https://vamsikrishna-employee-attrition.streamlit.app)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![Pytest Coverage](https://img.shields.io/badge/coverage-97%25-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 🚀 **Live Interactive Demo**: Try the deployed people analytics & attrition simulator live at **[https://vamsikrishna-employee-attrition.streamlit.app](https://vamsikrishna-employee-attrition.streamlit.app)**

An enterprise-ready People Analytics and Supervised Machine Learning system designed to identify voluntary employee turnover risks, isolate organizational flight drivers, evaluate demographic parity (EEOC 80% Rule), and generate proactive retention playbooks.

---

## 1. Problem Statement

Voluntary employee turnover imposes substantial direct and indirect costs on enterprise organizations. Industry research indicates that replacing a specialized knowledge worker or manager requires **1.5x to 2x** their annual compensation when factoring in recruiter fees, onboarding overhead, lost proprietary domain expertise, and project delays.

Traditional human resources teams rely on lagging indicators—such as exit interviews or annual sentiment surveys—which surface grievances only after top performers have decided to resign. This project formulates proactive employee retention as a supervised classification challenge, predicting individual attrition hazards while maintaining strict ethical AI boundaries and avoiding algorithmic discrimination.

---

## 2. Project Objectives

- **Develop a Leak-Free Pipeline**: Implement Scikit-learn `ColumnTransformer` and `Pipeline` architectures ensuring zero training-test data contamination.
- **Model Comparison & Hyperparameter Optimization**: Train and cross-validate Logistic Regression, Decision Tree, and Random Forest models, tuning key hyperparameters via 5-fold `GridSearchCV`.
- **Interpret Flight Drivers**: Extract Gini feature importances to determine whether turnover is driven by compensation, burnout (overtime), or lack of promotion.
- **Audit Demographic Fairness**: Perform algorithmic fairness checks across demographic groups (Gender) evaluating the Four-Fifths (80%) Disparate Impact rule.
- **Production Decision Support**: Build an interactive Streamlit application enabling HR business partners to simulate retention interventions in real time.

---

## 3. Real-World Use Case & Business Value

```
                               ┌───────────────────────────┐
                               │   Workforce Attributes    │
                               │  Tenure, Income, Overtime │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │   Scikit-Learn Pipeline   │
                               │ Imputation + OHE + Scaler │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  Tuned Random Forest Clf  │
                               │   Attrition Probability   │
                               └─────────────┬─────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
         ┌───────────────────────────┐               ┌───────────────────────────┐
         │     High Flight Risk      │               │     Low Flight Risk       │
         │  (Probability >= 65%)     │               │  (Probability < 20%)      │
         └─────────────┬─────────────┘               └─────────────┬─────────────┘
                       │                                           │
                       ▼                                           ▼
         ┌───────────────────────────┐               ┌───────────────────────────┐
         │ Priority Retention Sync   │               │ Leadership Sponsorship    │
         │ Workload rebalancing &    │               │ High-impact mentorship &  │
         │ compensation review       │               │ career path advancement   │
         └───────────────────────────┘               └───────────────────────────┘
```

---

## 4. Dataset Source and Licensing

- **Dataset Origin**: Modeled on the open IBM HR Analytics Employee Attrition & Performance dataset, a recognized benchmark in people analytics.
- **Synthetic HR Generator**: Built into `src/data_loader.py` to synthesize 2,000 realistic records with 23 features, including compensation tiers, role tenure, travel frequency, and ordinal job satisfaction scores.
- **Class Distribution**: Calibrated to ~16.0% voluntary attrition prevalence to match modern tech and enterprise turnover rates.
- **License**: Released under the [MIT License](../../LICENSE) for open academic and portfolio evaluation.

---

## 5. Technologies Used

- **Python 3.11+**
- **NumPy & Pandas**: Data manipulation, hazard modeling, and feature engineering.
- **Scikit-learn**: Pipelines, ColumnTransformer, StratifiedKFold, GridSearchCV, Random Forest, Logistic Regression.
- **Joblib**: Atomic model serialization and artifact persistence.
- **Matplotlib & Seaborn**: ROC/PR curves, confusion matrices, and feature importance rankings.
- **Streamlit**: Enterprise interactive flight risk scoring dashboard.
- **Pytest & Pytest-Cov**: Unit test suite with strict coverage enforcement ($\ge 90\%$).

---

## 6. Machine Learning Workflow

```
1. Ingestion ────────> 2. Validation ───────> 3. Pipeline ────────> 4. Stratified CV
   (Synthetic HR        (Deduplication &        (Imputer, OHE,       (5-Fold CV on
    2,000 records)       Missing Audits)         StandardScaler)      3 Architectures)
                                                                            │
                                                                            ▼
8. Interactive App <── 7. Fairness Check <─── 6. Evaluation <───── 5. Grid Search
   (Streamlit GUI,      (EEOC 80% Rule          (Accuracy, Recall,   (Hyperparameter
    Simulation)          Disparate Impact)       ROC-AUC, PR-AUC)     Optimization)
```

---

## 7. Project Architecture & Folder Structure

```
06-employee-attrition-prediction/
├── README.md
├── requirements.txt
├── train.py
├── streamlit_app.py
├── data/
│   └── employee_attrition.csv
├── models/
│   └── best_attrition_model.joblib
├── notebooks/
│   └── 06_employee_attrition_walkthrough.ipynb
├── reports/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── predictor.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_data_preprocessing.py
│   ├── test_model_training.py
│   ├── test_model_evaluation.py
│   └── test_predictor.py
└── visualizations/
    ├── model_comparison_benchmark.png
    ├── confusion_matrices.png
    ├── roc_pr_curves.png
    └── feature_importance_top15.png
```

---

## 8. Installation & Setup

```bash
# Navigate to the project directory
cd 03-machine-learning/06-employee-attrition-prediction

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 9. How to Run the Project

### Train and Evaluate Models
```bash
python train.py
```
This script cleans the dataset, runs 5-fold cross-validation across three candidate algorithms, performs GridSearchCV tuning on Random Forest, outputs benchmark metrics, performs an EEOC demographic fairness audit, serializes the best pipeline to `models/best_attrition_model.joblib`, and renders publication figures in `visualizations/`.

### Run Unit Tests & Coverage
```bash
pytest --cov=src --cov-report=term-missing
```

### Launch Interactive Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```

---

## 10. Model Details & Cross-Validation Benchmarks

### 5-Fold Stratified Cross-Validation (Training Set)

| Candidate Model | CV Mean ROC-AUC | CV Std ROC-AUC | Key Parameters |
| :--- | :---: | :---: | :--- |
| **Logistic Regression** | **0.8518** | ±0.0210 | `class_weight='balanced'`, L2 penalty, max_iter=1000 |
| **Random Forest (Baseline)** | **0.8468** | ±0.0162 | `n_estimators=100`, `max_depth=8`, `min_samples_leaf=4` |
| **Decision Tree** | 0.7798 | ±0.0267 | `max_depth=6`, `min_samples_split=20`, `class_weight='balanced'` |

### Hyperparameter Optimization (GridSearchCV)
- **Tuned Model**: `RandomForestClassifier` (balanced)
- **Optimal Hyperparameters**: `{'classifier__max_depth': 10, 'classifier__min_samples_leaf': 5, 'classifier__n_estimators': 120}`
- **Best Cross-Validation ROC-AUC**: **0.8585**

---

## 11. Held-Out Evaluation Results (20% Unseen Test Set)

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tuned Random Forest** | **87.50%** | **62.50%** | 54.69% | **58.33%** | 0.8559 | **0.6630** |
| **Baseline Random Forest** | 86.75% | 58.93% | 51.56% | 55.00% | 0.8532 | 0.6558 |
| **Logistic Regression** | 78.50% | 40.68% | **75.00%** | 52.75% | **0.8599** | 0.6409 |
| **Decision Tree** | 80.50% | 43.04% | 53.12% | 47.55% | 0.7758 | 0.4497 |

### Business Trade-Off Analysis
- **Tuned Random Forest**: Optimizes overall precision (62.5%) and accuracy (87.5%), making it ideal when retention interventions (e.g., retention bonuses, equity grants) carry substantial enterprise budgets.
- **Logistic Regression**: Maximizes recall (75.0%), capturing 3 out of every 4 departing employees at the cost of higher false alarms (40.7% precision), suitable for zero-cost interventions like casual manager check-ins.

---

## 12. Top Predictive Flight Drivers

Ranked by Gini feature importance from the tuned Random Forest:

1. **Job Satisfaction** (13.91%): Strongest negative predictor; disengaged employees leave rapidly.
2. **OverTime (Yes)** (10.25%): Chronic overtime without schedule relief significantly inflates flight probability.
3. **Monthly Income** (9.70%): Below-market compensation creates severe attrition vulnerability.
4. **OverTime (No)** (9.18%): Predictable 40-hour workweeks act as a strong retention anchor.
5. **Distance From Home** (5.22%): Lengthy daily commutes increase burnout and willingness to switch employers.
6. **Years Since Last Promotion** (4.93%): Promotion stagnation past 4+ years triggers external job searches.
7. **Years In Current Role** (4.81%): Role monotony correlates with career transition intent.
8. **Age** (4.58%): Early-career professionals exhibit higher mobility than late-career veterans.

---

## 13. Ethical AI & Demographic Fairness Audit

In compliance with the **Equal Employment Opportunity Commission (EEOC) Four-Fifths (80%) Rule**, we audited predictions across demographic groups on held-out data:

| Protected Group | Sample Count | Predicted Attrition Rate | Disparate Impact Ratio | Passes 80% Rule? |
| :--- | :---: | :---: | :---: | :---: |
| **Female** | 160 | 14.38% | 1.000 | **PASS** |
| **Male** | 240 | 13.75% | 0.956 | **PASS** |

The disparate impact ratio of 0.956 is well above the legal 0.80 threshold, verifying that the model does not disproportionately target either gender.

> [!IMPORTANT]
> **Responsible AI Mandate**: This model is strictly an advisory decision-support tool for employee wellness and retention. It must **never** be used for termination decisions, performance ranking, or adverse compensation adjustments.

---

## 14. Visualizations

Generated automatically during pipeline execution:
- [model_comparison_benchmark.png](visualizations/model_comparison_benchmark.png): Multi-metric bar chart comparing all candidate architectures.
- [confusion_matrices.png](visualizations/confusion_matrices.png): Side-by-side confusion matrix comparisons.
- [roc_pr_curves.png](visualizations/roc_pr_curves.png): ROC curves and Precision-Recall curves.
- [feature_importance_top15.png](visualizations/feature_importance_top15.png): Top 15 ranked turnover drivers.

---

## 15. Sample Production Inference

```python
from src.predictor import AttritionPredictor

predictor = AttritionPredictor()

employee = {
    "Age": 28,
    "Department": "Sales",
    "JobRole": "Sales Representative",
    "MonthlyIncome": 2500,
    "OverTime": "Yes",
    "JobSatisfaction": 1,
    "WorkLifeBalance": 1,
    "YearsSinceLastPromotion": 5,
    "DistanceFromHome": 25,
}

report = predictor.score_employee(employee)
print(f"Risk Tier: {report['risk_tier']}")
# Output: High Flight Risk (Probability: 82.4%)
print(f"Recommended Action: {report['recommended_action']}")
# Output: Priority 1-on-1 Retention Sync: Schedule confidential executive check-in...
```

---

## 16. Limitations

- **Cross-Sectional Data**: The dataset captures single-point snapshots rather than time-series trajectories (e.g., changes in satisfaction over successive quarters).
- **External Market Factors**: External macroeconomic shifts, competitor hiring sprees, and regional salary inflation are unobserved in internal HR records.
- **Survey Response Bias**: Self-reported satisfaction metrics can be skewed by recent performance evaluations or temporary project stress.

---

## 17. Future Improvements

- **Survival Analysis (Cox Proportional Hazards)**: Estimate time-to-event curves predicting *when* an employee is likely to leave rather than a binary classification.
- **SHAP / LIME Explanations**: Provide localized, per-employee Shapley value explanations within the Streamlit interface.
- **Organizational Network Analysis (ONA)**: Incorporate communication network density and mentorship graph connectivity as predictive features.
