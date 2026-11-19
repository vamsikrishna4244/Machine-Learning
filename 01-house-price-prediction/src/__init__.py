"""
House Price Prediction System: Modular Machine Learning Package.
"""

from .data_loader import load_housing_data, generate_housing_dataset
from .data_preprocessing import (
    clean_housing_data,
    detect_and_handle_outliers,
    build_preprocessing_pipeline,
)
from .model_training import train_regression_models, save_best_model
from .model_evaluation import evaluate_models, compare_models
from .predictor import HousePricePredictor

__all__ = [
    "load_housing_data",
    "generate_housing_dataset",
    "clean_housing_data",
    "detect_and_handle_outliers",
    "build_preprocessing_pipeline",
    "train_regression_models",
    "save_best_model",
    "evaluate_models",
    "compare_models",
    "HousePricePredictor",
]
