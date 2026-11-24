"""
End-to-End Employee Attrition Prediction Package.

Enterprise machine learning system with Scikit-learn Pipelines, ColumnTransformers,
Stratified 5-Fold Cross-Validation, GridSearchCV Hyperparameter Tuning, and HR Risk Scoring.
"""

from .data_loader import generate_attrition_dataset, load_attrition_data
from .data_preprocessing import (
    clean_attrition_data,
    build_attrition_preprocessor,
    prepare_attrition_train_test,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)
from .model_training import (
    train_baseline_models,
    tune_best_model_grid_search,
    save_attrition_pipeline,
    load_attrition_pipeline,
)
from .model_evaluation import (
    compute_attrition_metrics,
    evaluate_attrition_models,
    extract_top_feature_importances,
    plot_attrition_evaluations,
    conduct_demographic_fairness_audit,
)
from .predictor import AttritionPredictor

__all__ = [
    "generate_attrition_dataset",
    "load_attrition_data",
    "clean_attrition_data",
    "build_attrition_preprocessor",
    "prepare_attrition_train_test",
    "NUMERICAL_FEATURES",
    "CATEGORICAL_FEATURES",
    "train_baseline_models",
    "tune_best_model_grid_search",
    "save_attrition_pipeline",
    "load_attrition_pipeline",
    "compute_attrition_metrics",
    "evaluate_attrition_models",
    "extract_top_feature_importances",
    "plot_attrition_evaluations",
    "conduct_demographic_fairness_audit",
    "AttritionPredictor",
]
