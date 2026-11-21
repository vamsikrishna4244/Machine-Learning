"""
Unit Tests for Vectorizers and Keyword Introspection.
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from src.data_loader import generate_spam_dataset
from src.text_preprocessing import preprocess_corpus
from src.feature_extraction import (
    build_bow_vectorizer,
    build_tfidf_vectorizer,
    extract_top_indicative_keywords,
)


def test_build_bow_vectorizer():
    """Verify CountVectorizer construction and transformation."""
    vec = build_bow_vectorizer(max_features=500, min_df=1)
    assert isinstance(vec, CountVectorizer)
    corpus = ["meeting schedule today", "urgent claim cash prize"]
    X = vec.fit_transform(corpus)
    assert X.shape[0] == 2


def test_build_tfidf_vectorizer():
    """Verify TfidfVectorizer construction and transformation."""
    vec = build_tfidf_vectorizer(max_features=500, min_df=1)
    assert isinstance(vec, TfidfVectorizer)
    corpus = ["sprint review notes", "exclusive lottery winner announcement"]
    X = vec.fit_transform(corpus)
    assert X.shape[0] == 2


def test_extract_top_indicative_keywords():
    """Verify keyword discrimination between spam and ham."""
    raw = generate_spam_dataset(n_samples=150, seed=42)
    cleaned, _ = preprocess_corpus(raw)

    vec = build_tfidf_vectorizer(max_features=1000)
    vec.fit(cleaned["cleaned_message"])

    keywords = extract_top_indicative_keywords(
        vectorizer=vec,
        X_text=cleaned["cleaned_message"],
        y_labels=cleaned["label"],
        top_n=5,
    )

    assert "ham_keywords" in keywords
    assert "spam_keywords" in keywords
    assert len(keywords["ham_keywords"]) == 5
    assert len(keywords["spam_keywords"]) == 5
