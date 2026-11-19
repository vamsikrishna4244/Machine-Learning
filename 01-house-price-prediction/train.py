"""
End-to-End Training, Evaluation, and Serialization Pipeline.

Executes data loading, cleaning, feature engineering, model training,
comparative evaluation, and artifact serialization.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_housing_data
from src.data_preprocessing import (
    clean_housing_data,
    detect_and_handle_outliers,
    prepare_train_test_data,
)
from src.model_training import train_regression_models, save_best_model
from src.model_evaluation import (
    evaluate_models,
    compare_models,
    compute_regression_metrics,
    plot_model_evaluations,
)
from src.predictor import HousePricePredictor


def main() -> None:
    print("=" * 70)
    print("      HOUSE PRICE PREDICTION SYSTEM — TRAINING & EVALUATION PIPELINE")
    print("=" * 70)

    # 1. Ingestion
    print("\n[1/6] Ingesting Housing Dataset...")
    raw_df = load_housing_data()
    print(f"  ✓ Loaded {len(raw_df):,} raw property records with {raw_df.shape[1]} attributes.")

    # 2. Data Cleaning & Feature Engineering
    print("\n[2/6] Cleaning Data & Engineering Features...")
    cleaned_df, clean_report = clean_housing_data(raw_df)
    print(f"  ✓ Deduplication: Removed {clean_report['duplicates_removed']} duplicate records.")
    print(f"  ✓ Missing values: Handled {clean_report['missing_values_handled']} nulls.")
    print(f"  ✓ Engineered features: house_age, is_renovated, sqft_per_bedroom, bath_to_bed_ratio.")

    # 3. Outlier Identification & Treatment
    print("\n[3/6] Identifying & Treating Numerical Outliers...")
    treated_df, outlier_report = detect_and_handle_outliers(cleaned_df, iqr_multiplier=2.5, action="cap")
    for col, count in outlier_report.items():
        print(f"  ✓ Column '{col}': Capped {count} extreme values via 2.5x IQR boundaries.")

    # 4. Train-Test Split
    print("\n[4/6] Splitting Dataset (80% Train, 20% Held-Out Test)...")
    X_train, X_test, y_train, y_test = prepare_train_test_data(treated_df, test_size=0.2, random_state=42)
    print(f"  ✓ Training records: {len(X_train):,}")
    print(f"  ✓ Testing records:  {len(X_test):,}")

    # 5. Model Training
    print("\n[5/6] Training Candidate Regression Models (Linear, Tree, Random Forest)...")
    trained_models = train_regression_models(X_train, y_train, random_state=42)
    for model_name in trained_models:
        print(f"  ✓ Model trained: {model_name}")

    # 6. Evaluation & Comparison
    print("\n[6/6] Evaluating on Held-Out Test Set...")
    summary_df, predictions = evaluate_models(trained_models, X_test, y_test)
    print("\n" + "=" * 70)
    print("                     MODEL COMPARISON RESULTS")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)

    best_name = compare_models(summary_df)
    best_pipeline = trained_models[best_name]
    best_metrics = compute_regression_metrics(y_test, predictions[best_name])
    print(f"\n🏆 Best Model Selected: {best_name} (R² = {best_metrics['r2_score']:.4f}, RMSE = ${best_metrics['rmse']:,.2f})")

    # Persist Best Model
    saved_path = save_best_model(best_pipeline, best_name, best_metrics)
    print(f"  ✓ Model artifact serialized to: {saved_path.resolve()}")

    # Render Visualizations
    plots = plot_model_evaluations(y_test, predictions, summary_df, best_pipeline)
    print("  ✓ Evaluation figures saved:")
    for plot_type, path in plots.items():
        print(f"    - {plot_type}: {path.name}")

    # Verify Production Inference Interface
    print("\n[Inference Verification] Testing HousePricePredictor on Sample Property:")
    predictor = HousePricePredictor()
    sample_house = {
        "bedrooms": 3,
        "bathrooms": 2.5,
        "sqft_living": 2400.0,
        "sqft_lot": 7500.0,
        "floors": 2.0,
        "waterfront": 0,
        "view_quality": 2,
        "condition": 4,
        "grade": 8,
        "yr_built": 2005,
        "yr_renovated": 0,
        "neighborhood": "Westside",
    }
    val = predictor.predict_single(sample_house)
    print(f"  • Sample Property: 3 bed, 2.5 bath, 2,400 sqft in Westside (Built 2005)")
    print(f"  • Estimated Valuation: ${val['estimated_price']:,.2f}")
    print(f"  • Valuation Range:     ${val['estimated_range_lower']:,.2f} – ${val['estimated_range_upper']:,.2f}")
    print("\n>>> PIPELINE EXECUTION COMPLETED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    main()
