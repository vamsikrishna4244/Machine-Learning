"""
End-to-End Model Training, Evaluation, and Serialization Pipeline for Customer Churn Prediction.

Executes data acquisition/generation, cleaning, feature engineering,
stratified train-test splitting, candidate model training with balanced weighting,
metric benchmarking (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC),
visualization generation, artifact serialization, and sample inference verification.
"""

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import generate_churn_dataset, load_churn_data
from src.data_preprocessing import (
    clean_churn_data,
    prepare_churn_train_test_data,
)
from src.model_training import (
    train_churn_classifiers,
    save_best_churn_model,
)
from src.model_evaluation import (
    evaluate_churn_models,
    compare_churn_models,
    plot_churn_evaluations,
)
from src.predictor import ChurnPredictor


def run_pipeline() -> None:
    """Execute the complete customer churn training workflow."""
    print("=" * 75)
    print("      PROJECT 02: CUSTOMER CHURN PREDICTION - TRAINING PIPELINE")
    print("=" * 75)

    data_dir = PROJECT_ROOT / "data"
    raw_data_path = data_dir / "customer_churn.csv"
    processed_data_path = data_dir / "processed_customer_churn.csv"

    # Step 1: Data Acquisition / Ingestion
    print("\n[Step 1/6] Ingesting Customer Churn Dataset...")
    if not raw_data_path.exists():
        print("  Generating synthetic subscriber dataset (3,500 records)...")
        raw_df = generate_churn_dataset(n_samples=3500, seed=42)
        raw_data_path.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(raw_data_path, index=False)
        print(f"  Saved raw dataset to: {raw_data_path}")
    else:
        print(f"  Loading existing dataset from: {raw_data_path}")
        raw_df = load_churn_data(raw_data_path)

    print(f"  Raw dataset dimensions: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

    # Step 2: Data Cleaning & Feature Engineering
    print("\n[Step 2/6] Cleaning Data & Engineering Domain Features...")
    clean_df, report = clean_churn_data(raw_df)
    clean_df.to_csv(processed_data_path, index=False)
    print(f"  Duplicates removed: {report['duplicates_removed']}")
    print(f"  Total records after cleaning: {report['final_records']}")
    print(f"  Churn prevalence (positive class): {report['churn_rate_pct']:.2f}%")
    print(f"  Saved processed dataset to: {processed_data_path}")

    # Step 3: Stratified Train-Test Split
    print("\n[Step 3/6] Splitting Dataset (80/20 Stratified)...")
    X_train, X_test, y_train, y_test = prepare_churn_train_test_data(
        clean_df, test_size=0.2, random_state=42
    )
    print(f"  Training samples: {X_train.shape[0]} (Churned: {int(y_train.sum())})")
    print(f"  Test samples:     {X_test.shape[0]} (Churned: {int(y_test.sum())})")
    print(f"  Train churn rate: {y_train.mean():.2%}, Test churn rate: {y_test.mean():.2%}")

    # Step 4: Model Training with Class Balancing
    print("\n[Step 4/6] Training Candidate Classifiers (Logistic Reg, Decision Tree, Random Forest)...")
    models = train_churn_classifiers(X_train, y_train, random_state=42)
    for name in models.keys():
        print(f"  [+] Trained: {name}")

    # Step 5: Model Evaluation & Benchmarking
    print("\n[Step 5/6] Benchmarking Classifiers on Held-Out Test Set...")
    summary_df, predictions, probabilities = evaluate_churn_models(models, X_test, y_test)

    print("\n" + "-" * 75)
    print("                     MODEL PERFORMANCE BENCHMARK")
    print("-" * 75)
    print(
        f"{'Model':<22} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'ROC-AUC':<8} | {'PR-AUC':<8}"
    )
    print("-" * 75)
    for _, row in summary_df.iterrows():
        print(
            f"{row['Model']:<22} | {row['Accuracy']:<8.4f} | {row['Precision']:<9.4f} | "
            f"{row['Recall']:<8.4f} | {row['F1-Score']:<8.4f} | {row['ROC-AUC']:<8.4f} | {row['PR-AUC']:<8.4f}"
        )
    print("-" * 75)

    best_name = compare_churn_models(summary_df)
    best_pipe = models[best_name]
    best_metrics = summary_df[summary_df["Model"] == best_name].iloc[0].to_dict()
    print(f"\n  >>> Best Overall Classifier: {best_name} (ROC-AUC: {best_metrics['ROC-AUC']:.4f})")

    # Step 6: Serialization, Visualizations, and Inference Verification
    print("\n[Step 6/6] Persisting Artifacts & Verifying Inference Engine...")
    model_path = save_best_churn_model(
        pipeline=best_pipe,
        model_name=best_name,
        metrics={
            "accuracy": best_metrics["Accuracy"],
            "precision": best_metrics["Precision"],
            "recall": best_metrics["Recall"],
            "f1_score": best_metrics["F1-Score"],
            "roc_auc": best_metrics["ROC-AUC"],
            "pr_auc": best_metrics["PR-AUC"],
        },
    )
    print(f"  Model artifact saved to: {model_path}")

    # Visualizations
    plots = plot_churn_evaluations(
        y_test=y_test,
        predictions=predictions,
        probabilities=probabilities,
        summary_df=summary_df,
        best_pipeline=best_pipe,
    )
    print("  Diagnostic figures generated:")
    for plot_name, path in plots.items():
        print(f"    - {plot_name}: {path.name}")

    # Verify inference engine
    print("\n  Verifying ChurnPredictor Inference Engine...")
    predictor = ChurnPredictor(model_path=model_path)
    sample_subscriber = {
        "gender": "Female",
        "senior_citizen": "No",
        "partner": "No",
        "dependents": "No",
        "tenure_months": 2,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 89.50,
        "total_charges": 179.00,
    }
    result = predictor.score_customer(sample_subscriber)
    print(f"  Sample Customer Churn Probability: {result['churn_probability']:.1%}")
    print(f"  Risk Tier:                         {result['risk_tier']}")
    print(f"  Churn Predicted:                   {result['predicted_churn']}")
    print(f"  Intervention Recommendation:       {result['recommended_action']}")

    print("\n" + "=" * 75)
    print("               PIPELINE RUN COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline()
