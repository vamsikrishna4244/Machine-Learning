"""
Unit Tests for Text Preprocessing, Stopword Removal, and Corpus Cleaning.
"""

import pandas as pd
from src.data_loader import generate_spam_dataset
from src.text_preprocessing import (
    clean_email_text,
    preprocess_corpus,
    split_email_dataset,
)


def test_clean_email_text_basic():
    """Verify lowercasing, punctuation stripping, and token normalization."""
    raw = "URGENT! Claim $50,000 cash prize at https://promo-scam.xyz/claim now!"
    cleaned = clean_email_text(raw)
    assert "urgent" in cleaned
    assert "moneytoken" in cleaned or "claim" in cleaned
    assert "!" not in cleaned
    assert "https://" not in cleaned


def test_clean_email_text_edge_cases():
    """Verify handling of empty strings and non-string inputs."""
    assert clean_email_text("") == ""
    assert clean_email_text(None) == ""  # type: ignore


def test_preprocess_corpus_deduplication_and_filtering():
    """Verify deduplication, feature extraction, and report statistics."""
    df = generate_spam_dataset(n_samples=150, seed=42)
    cleaned_df, report = preprocess_corpus(df)

    assert "cleaned_message" in cleaned_df.columns
    assert "char_count" in cleaned_df.columns
    assert "word_count" in cleaned_df.columns
    assert report["duplicates_removed"] > 0
    assert report["final_emails"] <= report["initial_emails"]
    assert report["spam_count"] > 0
    assert report["ham_count"] > 0


def test_split_email_dataset_stratification():
    """Verify train-test split sizes and preserved class proportions."""
    df = generate_spam_dataset(n_samples=200, spam_ratio=0.20, seed=42)
    cleaned_df, _ = preprocess_corpus(df)
    X_train, X_test, y_train, y_test = split_email_dataset(cleaned_df, test_size=0.25, random_state=42)

    assert len(X_train) + len(X_test) == len(cleaned_df)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    # Check stratification preservation
    train_spam_rate = y_train.mean()
    test_spam_rate = y_test.mean()
    assert abs(train_spam_rate - test_spam_rate) < 0.05


def test_clean_email_text_flags():
    """Verify clean_email_text with special token and stopword flags toggled."""
    raw = "Check http://example.com and call 12345"
    res1 = clean_email_text(raw, normalize_special_tokens=False)
    assert "http" not in res1
    assert "urltoken" not in res1

    res2 = clean_email_text("the quick brown fox", remove_stopwords=False)
    assert "the" in res2
