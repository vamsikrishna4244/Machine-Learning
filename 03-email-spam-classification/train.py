"""
End-to-End Training, Vectorization Comparison, and Serialization Pipeline for Email Spam Classification.

Executes data generation/loading, text cleaning, stratified splitting,
benchmarking Bag-of-Words and TF-IDF across Naive Bayes, Logistic Regression,
and Support Vector Machines, artifact serialization, and sample inference verification.
"""

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import generate_spam_dataset, load_email_data
from src.text_preprocessing import preprocess_corpus, split_email_dataset
from src.model_training import train_spam_classifiers, save_best_spam_pipeline
from src.model_evaluation import (
    evaluate_spam_models,
    compare_vectorization_methods,
    plot_spam_evaluations,
)
from src.predictor import SpamPredictor


def run_pipeline() -> None:
    """Execute complete email spam classification workflow."""
    print("=" * 80)
    print("      PROJECT 03: EMAIL SPAM CLASSIFICATION - TRAINING PIPELINE")
    print("=" * 80)

    data_dir = PROJECT_ROOT / "data"
    raw_csv = data_dir / "spam_emails.csv"
    processed_csv = data_dir / "processed_emails.csv"

    # Step 1: Ingestion
    print("\n[Step 1/6] Ingesting Email Dataset...")
    if not raw_csv.exists():
        print("  Generating synthetic email dataset (2,500 messages)...")
        raw_df = generate_spam_dataset(n_samples=2500, spam_ratio=0.18, seed=42, output_path=raw_csv)
        print(f"  Saved raw emails to: {raw_csv}")
    else:
        print(f"  Loading existing emails from: {raw_csv}")
        raw_df = load_email_data(raw_csv)

    print(f"  Raw dataset dimensions: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

    # Step 2: Text Preprocessing & Cleaning
    print("\n[Step 2/6] Sanitizing Corpus & Filtering Stopwords...")
    clean_df, report = preprocess_corpus(raw_df)
    clean_df.to_csv(processed_csv, index=False)
    print(f"  Duplicates removed: {report['duplicates_removed']}")
    print(f"  Total valid emails after cleaning: {report['final_emails']}")
    print(f"  Ham messages: {report['ham_count']} | Spam messages: {report['spam_count']}")
    print(f"  Spam ratio: {report['spam_ratio_pct']:.2f}%")
    print(f"  Saved processed corpus to: {processed_csv}")

    # Step 3: Stratified Train-Test Split
    print("\n[Step 3/6] Splitting Dataset (80/20 Stratified)...")
    X_train, X_test, y_train, y_test = split_email_dataset(clean_df, test_size=0.20, random_state=42)
    print(f"  Training samples: {len(X_train)} (Spam: {int(y_train.sum())})")
    print(f"  Testing samples:  {len(X_test)} (Spam: {int(y_test.sum())})")

    # Step 4: Training all 6 Model Configurations
    print("\n[Step 4/6] Training Classifiers (BoW & TF-IDF x NB, LR, SVM)...")
    models = train_spam_classifiers(X_train, y_train, random_state=42)
    for name in models.keys():
        print(f"  [+] Trained: {name}")

    # Step 5: Multi-Metric Evaluation & Comparison
    print("\n[Step 5/6] Benchmarking Classifiers on Held-Out Test Data...")
    summary_df, predictions, probabilities = evaluate_spam_models(models, X_test, y_test)

    print("\n" + "-" * 85)
    print("                     MODEL PERFORMANCE BENCHMARK")
    print("-" * 85)
    print(
        f"{'Configuration':<32} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'ROC-AUC':<8}"
    )
    print("-" * 85)
    for _, row in summary_df.iterrows():
        print(
            f"{row['Configuration']:<32} | {row['Accuracy']:<8.4f} | {row['Precision']:<9.4f} | "
            f"{row['Recall']:<8.4f} | {row['F1-Score']:<8.4f} | {row['ROC-AUC']:<8.4f}"
        )
    print("-" * 85)

    # Vectorizer comparison aggregate
    vec_summary = compare_vectorization_methods(summary_df)
    print("\n--- Vectorizer Comparison (Average Metrics) ---")
    print(vec_summary.to_string(index=False))

    best_config = summary_df.iloc[0]["Configuration"]
    best_pipe = models[best_config]
    best_metrics = summary_df.iloc[0].to_dict()
    print(f"\n  >>> Top-Performing Pipeline: {best_config} (F1-Score: {best_metrics['F1-Score']:.4f})")

    # Step 6: Artifact Serialization & Verification
    print("\n[Step 6/6] Persisting Top Pipeline & Verifying Predictor Engine...")
    saved_path = save_best_spam_pipeline(
        pipeline=best_pipe,
        model_name=best_config,
        metrics={
            "accuracy": best_metrics["Accuracy"],
            "precision": best_metrics["Precision"],
            "recall": best_metrics["Recall"],
            "f1_score": best_metrics["F1-Score"],
            "roc_auc": best_metrics["ROC-AUC"],
        },
    )
    print(f"  Serialized model pipeline to: {saved_path}")

    # Visualizations
    plots = plot_spam_evaluations(
        y_test=y_test,
        predictions=predictions,
        summary_df=summary_df,
        best_pipeline=best_pipe,
    )
    print("  Generated visual diagnostics:")
    for plot_name, path in plots.items():
        print(f"    - {plot_name}: {path.name}")

    # Verify inference engine
    print("\n  Testing SpamPredictor on Sample Messages...")
    predictor = SpamPredictor(model_path=saved_path)

    sample_ham = (
        "Hi team, moving our 10 AM sync to 2 PM today to accommodate the engineering team's testing schedule."
    )
    res_ham = predictor.score_email(sample_ham)
    print(f"\n  [Sample 1: Legitimate Email]")
    print(f"  Text: {sample_ham[:70]}...")
    print(f"  Classification:  {res_ham['classification']} ({res_ham['spam_probability']:.1%})")
    print(f"  Confidence Tier: {res_ham['confidence_tier']}")

    sample_spam = (
        "Congratulations! You won the official Apple iPhone 15 lottery giveaway. "
        "Claim your prize immediately at http://promo-claim.xyz"
    )
    res_spam = predictor.score_email(sample_spam)
    print(f"\n  [Sample 2: Suspicious Email]")
    print(f"  Text: {sample_spam[:70]}...")
    print(f"  Classification:  {res_spam['classification']} ({res_spam['spam_probability']:.1%})")
    print(f"  Confidence Tier: {res_spam['confidence_tier']}")
    print(f"  Trigger Tokens:  {res_spam['trigger_keywords']}")
    print(f"  Action Playbook: {res_spam['recommended_action']}")

    print("\n" + "=" * 80)
    print("               PIPELINE RUN COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
