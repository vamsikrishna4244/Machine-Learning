"""
End-to-End Training, Cost-Sensitive Evaluation, and Serialization for Credit Card Fraud Detection.

Demonstrates the fallacy of accuracy on imbalanced data, trains baseline unweighted,
cost-sensitive balanced, and resampled pipelines across Logistic Regression, Decision Tree,
and Random Forest, benchmarks PR-AUC and ROC-AUC, and visualizes financial trade-offs.
"""

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import generate_credit_card_data, load_fraud_data
from src.data_preprocessing import (
    preprocess_fraud_data,
    prepare_stratified_train_test,
)
from src.model_training import train_fraud_classifiers, save_best_fraud_model
from src.model_evaluation import (
    evaluate_fraud_models,
    plot_fraud_evaluations,
)
from src.predictor import FraudPredictor


def run_pipeline() -> None:
    """Execute complete fraud detection training workflow."""
    print("=" * 85)
    print("      PROJECT 04: CREDIT CARD FRAUD DETECTION - TRAINING PIPELINE")
    print("=" * 85)

    data_dir = PROJECT_ROOT / "data"
    raw_csv = data_dir / "credit_card_fraud.csv"
    proc_csv = data_dir / "processed_transactions.csv"

    # Step 1: Ingestion
    print("\n[Step 1/6] Ingesting Credit Card Transactions...")
    if not raw_csv.exists():
        print("  Generating synthetic transaction dataset (6,000 transactions, ~0.8% fraud)...")
        raw_df = generate_credit_card_data(n_samples=6000, fraud_rate=0.008, seed=42, output_path=raw_csv)
        print(f"  Saved raw dataset to: {raw_csv}")
    else:
        print(f"  Loading existing dataset from: {raw_csv}")
        raw_df = load_fraud_data(raw_csv)

    print(f"  Raw dimensions: {raw_df.shape[0]} transactions, {raw_df.shape[1]} features")

    # Step 2: Data Cleaning & Preprocessing
    print("\n[Step 2/6] Cleaning Data & Validating Class Distribution...")
    clean_df, report = preprocess_fraud_data(raw_df)
    clean_df.to_csv(proc_csv, index=False)
    print(f"  Duplicates removed: {report['duplicates_removed']}")
    print(f"  Total valid transactions: {report['final_records']}")
    print(f"  Legitimate: {report['legitimate_count']} | Fraudulent: {report['fraud_count']}")
    print(f"  Empirical Fraud Rate: {report['fraud_ratio_pct']:.3f}%")
    print(f"  Saved processed dataset to: {proc_csv}")

    # Step 3: Stratified Train-Test Splitting
    print("\n[Step 3/6] Splitting Dataset (80/20 Stratified)...")
    X_train, X_test, y_train, y_test = prepare_stratified_train_test(clean_df, test_size=0.20, random_state=42)
    print(f"  Training partition: {len(X_train)} samples (Fraud: {int(y_train.sum())})")
    print(f"  Testing partition:  {len(X_test)} samples (Fraud: {int(y_test.sum())})")

    # Step 4: Model Training across Balancing Regimes
    print("\n[Step 4/6] Training Candidate Classifiers (Baseline vs. Cost-Weighted vs. Resampled)...")
    models = train_fraud_classifiers(X_train, y_train, random_state=42)
    for name in models.keys():
        print(f"  [+] Trained: {name}")

    # Step 5: Multi-Metric Evaluation & Cost Analysis
    print("\n[Step 5/6] Benchmarking Classifiers on Held-Out Test Data...")
    summary_df, predictions, probabilities = evaluate_fraud_models(models, X_test, y_test)

    print("\n" + "-" * 95)
    print("                     MODEL PERFORMANCE BENCHMARK (IMBALANCED METRICS)")
    print("-" * 95)
    print(
        f"{'Model':<34} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'PR-AUC':<8} | {'Total Cost ($)':<12}"
    )
    print("-" * 95)
    for _, row in summary_df.iterrows():
        print(
            f"{row['Model']:<34} | {row['Accuracy']:<8.4f} | {row['Precision']:<9.4f} | "
            f"{row['Recall']:<8.4f} | {row['F1-Score']:<8.4f} | {row['PR-AUC']:<8.4f} | ${row['Total Cost ($)']:<11,.0f}"
        )
    print("-" * 95)

    best_model_name = summary_df.iloc[0]["Model"]
    best_pipe = models[best_model_name]
    best_row = summary_df.iloc[0].to_dict()
    print(f"\n  >>> Top-Performing Architecture: {best_model_name}")
    print(f"      PR-AUC: {best_row['PR-AUC']:.4f} | F1-Score: {best_row['F1-Score']:.4f} | Est. Loss: ${best_row['Total Cost ($)']:,.0f}")

    # Step 6: Artifact Serialization & Verification
    print("\n[Step 6/6] Persisting Top Pipeline & Verifying Predictor Engine...")
    saved_path = save_best_fraud_model(
        pipeline=best_pipe,
        model_name=best_model_name,
        metrics={
            "accuracy": best_row["Accuracy"],
            "precision": best_row["Precision"],
            "recall": best_row["Recall"],
            "f1_score": best_row["F1-Score"],
            "pr_auc": best_row["PR-AUC"],
            "roc_auc": best_row["ROC-AUC"],
        },
    )
    print(f"  Serialized model pipeline to: {saved_path}")

    # Visualizations
    plots = plot_fraud_evaluations(
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
    print("\n  Testing FraudPredictor Inference Engine on Sample Transactions...")
    predictor = FraudPredictor(model_path=saved_path)

    # Legitimate purchase
    legit_txn = {
        "Time": 45000.0,
        "Amount": 34.50,
        "V4": 0.2,
        "V10": 0.1,
        "V12": 0.3,
        "V14": 0.4,
        "V17": -0.1,
    }
    res_legit = predictor.score_transaction(legit_txn)
    print(f"\n  [Sample 1: Everyday Grocery Purchase ($34.50)]")
    print(f"  Fraud Probability: {res_legit['fraud_probability_pct']}")
    print(f"  Decision:          {res_legit['decision']} ({res_legit['risk_tier']})")
    print(f"  Action:            {res_legit['recommended_action']}")

    # Suspicious fraud transaction
    fraud_txn = {
        "Time": 82000.0,
        "Amount": 1450.00,
        "V4": 4.1,
        "V10": -4.2,
        "V12": -4.5,
        "V14": -6.2,
        "V17": -3.8,
    }
    res_fraud = predictor.score_transaction(fraud_txn)
    print(f"\n  [Sample 2: Out-of-Pattern High-Value Transfer ($1,450.00)]")
    print(f"  Fraud Probability: {res_fraud['fraud_probability_pct']}")
    print(f"  Decision:          {res_fraud['decision']} ({res_fraud['risk_tier']})")
    print(f"  Signals:           {res_fraud['flagged_signals']}")
    print(f"  Action:            {res_fraud['recommended_action']}")

    print("\n" + "=" * 85)
    print("               PIPELINE RUN COMPLETED SUCCESSFULLY!")
    print("=" * 85)


if __name__ == "__main__":
    run_pipeline()
