"""
Credit Card Fraud Detection Package.

Production-grade machine learning system for severe class imbalance,
precision-recall optimization, cost-sensitive learning, and transaction scoring.
"""

from .data_loader import generate_credit_card_data, load_fraud_data
from .data_preprocessing import (
    preprocess_fraud_data,
    build_transaction_preprocessor,
    prepare_stratified_train_test,
    resample_training_data,
)
from .model_training import (
    train_fraud_classifiers,
    save_best_fraud_model,
    load_saved_fraud_model,
)
from .model_evaluation import (
    compute_imbalanced_metrics,
    evaluate_fraud_models,
    plot_fraud_evaluations,
    calculate_business_cost_impact,
)
from .predictor import FraudPredictor

__all__ = [
    "generate_credit_card_data",
    "load_fraud_data",
    "preprocess_fraud_data",
    "build_transaction_preprocessor",
    "prepare_stratified_train_test",
    "resample_training_data",
    "train_fraud_classifiers",
    "save_best_fraud_model",
    "load_saved_fraud_model",
    "compute_imbalanced_metrics",
    "evaluate_fraud_models",
    "plot_fraud_evaluations",
    "calculate_business_cost_impact",
    "FraudPredictor",
]
