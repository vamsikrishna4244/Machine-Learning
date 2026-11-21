"""
Email Spam Classification Package.

Traditional Machine Learning Pipeline with Bag-of-Words, TF-IDF,
Naive Bayes, Logistic Regression, and Support Vector Machines.
"""

from .data_loader import generate_spam_dataset, load_email_data
from .text_preprocessing import clean_email_text, preprocess_corpus
from .feature_extraction import build_bow_vectorizer, build_tfidf_vectorizer
from .model_training import train_spam_classifiers, save_best_spam_pipeline, load_saved_spam_pipeline
from .model_evaluation import evaluate_spam_models, plot_spam_evaluations
from .predictor import SpamPredictor

__all__ = [
    "generate_spam_dataset",
    "load_email_data",
    "clean_email_text",
    "preprocess_corpus",
    "build_bow_vectorizer",
    "build_tfidf_vectorizer",
    "train_spam_classifiers",
    "save_best_spam_pipeline",
    "load_saved_spam_pipeline",
    "evaluate_spam_models",
    "plot_spam_evaluations",
    "SpamPredictor",
]
