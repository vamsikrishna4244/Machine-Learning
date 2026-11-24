"""
End-to-End Enterprise Employee Attrition Training, Tuning, and Ethical Audit Pipeline.

Executes data validation, ColumnTransformer pipeline construction, 5-fold cross-validation,
hyperparameter optimization via GridSearchCV, demographic fairness audits,
artifact serialization, and real-time employee flight risk inference.
"""

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import generate_attrition_dataset, load_attrition_data
from src.data_preprocessing import (
    clean_attrition_data,
    prepare_attrition_train_test,
)
from src.model_training import (
    train_baseline_models,
    tune_best_model_grid_search,
    save_attrition_pipeline,
)
from src.model_evaluation import (
    evaluate_attrition_models,
    extract_top_feature_importances,
    plot_attrition_evaluations,
    conduct_demographic_fairness_audit,
)
from src.predictor import AttritionPredictor


def run_pipeline() -> None:
    """Execute complete employee attrition training and tuning workflow."""
    print("=" * 85)
    print("      PROJECT 06: EMPLOYEE ATTRITION PREDICTION - TRAINING PIPELINE")
    print("=" * 85)

    data_dir = PROJECT_ROOT / "data"
    raw_csv = data_dir / "employee_attrition.csv"
    proc_csv = data_dir / "processed_attrition.csv"

    # Step 1: Ingestion
    print("\n[Step 1/7] Ingesting Employee Records...")
    if not raw_csv.exists():
        print("  Generating synthetic employee dataset (2,000 employees, ~16% attrition)...")
        raw_df = generate_attrition_dataset(n_samples=2000, seed=42, output_path=raw_csv)
        print(f"  Saved raw dataset to: {raw_csv}")
    else:
        print(f"  Loading existing dataset from: {raw_csv}")
        raw_df = load_attrition_data(raw_csv)

    print(f"  Raw dimensions: {raw_df.shape[0]} employees, {raw_df.shape[1]} attributes")

    # Step 2: Data Cleaning & Preprocessing
    print("\n[Step 2/7] Cleaning Data & Validating Feature Distributions...")
    clean_df, report = clean_attrition_data(raw_df)
    clean_df.to_csv(proc_csv, index=False)
    print(f"  Duplicates removed: {report['duplicates_removed']}")
    print(f"  Valid employees after cleaning: {report['final_records']}")
    print(f"  Retained: {report['retained_count']} | Left Company: {report['attrition_count']}")
    print(f"  Empirical Attrition Rate: {report['attrition_rate_pct']:.2f}%")
    print(f"  Saved processed dataset to: {proc_csv}")

    # Step 3: Stratified Train-Test Splitting
    print("\n[Step 3/7] Splitting Dataset (80/20 Stratified)...")
    X_train, X_test, y_train, y_test = prepare_attrition_train_test(clean_df, test_size=0.20, random_state=42)
    print(f"  Training partition: {len(X_train)} employees (Attrition: {int(y_train.sum())})")
    print(f"  Testing partition:  {len(X_test)} employees (Attrition: {int(y_test.sum())})")

    # Step 4: 5-Fold Stratified Cross-Validation on Baseline Models
    print("\n[Step 4/7] Evaluating 5-Fold Stratified CV on Candidate Classifiers...")
    candidate_models, cv_df = train_baseline_models(X_train, y_train, cv_folds=5, random_state=42)
    print("\n" + "-" * 55)
    print("      5-FOLD STRATIFIED CV BENCHMARK (ROC-AUC)")
    print("-" * 55)
    print(f"{'Model':<25} | {'Mean CV ROC-AUC':<16} | {'Std Dev':<10}")
    print("-" * 55)
    for _, r in cv_df.iterrows():
        print(f"{r['Model']:<25} | {r['CV_Mean_ROC_AUC']:<16.4f} | {r['CV_Std_ROC_AUC']:<10.4f}")
    print("-" * 55)

    # Step 5: Hyperparameter Tuning via GridSearchCV
    print("\n[Step 5/7] Tuning Random Forest via GridSearchCV...")
    tuned_rf, best_params, best_cv_score = tune_best_model_grid_search(X_train, y_train, cv_folds=5, random_state=42)
    print(f"  >>> Optimal Hyperparameters: {best_params}")
    print(f"  >>> Best CV ROC-AUC: {best_cv_score:.4f}")
    candidate_models["Tuned Random Forest"] = tuned_rf

    # Step 6: Held-Out Test Evaluation & Feature Importance
    print("\n[Step 6/7] Evaluating Final Models on Held-Out Test Set...")
    summary_df, predictions, probabilities = evaluate_attrition_models(candidate_models, X_test, y_test)

    print("\n" + "-" * 85)
    print("                     HELD-OUT TEST SET BENCHMARK")
    print("-" * 85)
    print(
        f"{'Model':<24} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'ROC-AUC':<8} | {'PR-AUC':<8}"
    )
    print("-" * 85)
    for _, row in summary_df.iterrows():
        print(
            f"{row['Model']:<24} | {row['Accuracy']:<8.4f} | {row['Precision']:<9.4f} | "
            f"{row['Recall']:<8.4f} | {row['F1-Score']:<8.4f} | {row['ROC-AUC']:<8.4f} | {row['PR-AUC']:<8.4f}"
        )
    print("-" * 85)

    best_model_name = summary_df.iloc[0]["Model"]
    best_pipe = candidate_models[best_model_name]
    best_metrics = summary_df.iloc[0].to_dict()

    # Feature Importance
    top_features = extract_top_feature_importances(best_pipe, top_n=10)
    print(f"\n  Top 5 Predictive Attrition Drivers ({best_model_name}):")
    for _, row in top_features.head(5).iterrows():
        print(f"    - {row['Feature']}: {row['Importance']:.4f}")

    # Demographic Fairness Audit
    print("\n  Conducting Demographic Disparate Impact Audit (Gender)...")
    audit_df = conduct_demographic_fairness_audit(best_pipe, X_test, sensitive_col="Gender")
    if not audit_df.empty:
        print(audit_df.to_string(index=False))

    # Step 7: Artifact Serialization & Verification
    print("\n[Step 7/7] Persisting Best Pipeline & Verifying Predictor Engine...")
    saved_path = save_attrition_pipeline(
        pipeline=best_pipe,
        model_name=best_model_name,
        metrics={
            "accuracy": best_metrics["Accuracy"],
            "precision": best_metrics["Precision"],
            "recall": best_metrics["Recall"],
            "f1_score": best_metrics["F1-Score"],
            "roc_auc": best_metrics["ROC-AUC"],
            "pr_auc": best_metrics["PR-AUC"],
        },
        best_params=best_params,
    )
    print(f"  Model artifact serialized to: {saved_path}")

    # Visualizations
    plots = plot_attrition_evaluations(
        y_test=y_test,
        predictions=predictions,
        probabilities=probabilities,
        summary_df=summary_df,
        best_pipeline=best_pipe,
    )
    print("  Generated visual diagnostics:")
    for plot_name, path in plots.items():
        print(f"    - {plot_name}: {path.name}")

    # Verify inference engine
    print("\n  Testing AttritionPredictor on Sample Employee Profiles...")
    predictor = AttritionPredictor(model_path=saved_path)

    sample_at_risk = {
        "Age": 28,
        "Gender": "Male",
        "MaritalStatus": "Single",
        "Department": "Sales",
        "JobRole": "Sales Representative",
        "JobLevel": 1,
        "MonthlyIncome": 2800,
        "TotalWorkingYears": 3,
        "YearsAtCompany": 2,
        "YearsInCurrentRole": 1,
        "YearsSinceLastPromotion": 2,
        "YearsWithCurrManager": 1,
        "DistanceFromHome": 25,
        "NumCompaniesWorked": 2,
        "JobSatisfaction": 1,
        "EnvironmentSatisfaction": 1,
        "WorkLifeBalance": 1,
        "JobInvolvement": 2,
        "OverTime": "Yes",
        "BusinessTravel": "Travel_Frequently",
        "StockOptionLevel": 0,
    }
    res = predictor.score_employee(sample_at_risk)
    print(f"\n  [Sample Employee: Sales Rep with Heavy Overtime]")
    print(f"  Attrition Probability: {res['attrition_probability_pct']}")
    print(f"  Risk Tier:              {res['risk_tier']}")
    print(f"  Primary Flight Drivers: {res['risk_factors']}")
    print(f"  Prescribed HR Action:   {res['recommended_action']}")

    print("\n" + "=" * 85)
    print("               PIPELINE RUN COMPLETED SUCCESSFULLY!")
    print("=" * 85)


if __name__ == "__main__":
    run_pipeline()
