"""
Text Preprocessing, Cleaning, Tokenization, and Corpus Sanitization Module.

Applies standard NLP cleaning: lowercasing, URL normalization, email stripping,
punctuation removal, whitespace normalization, and stopword filtering without
requiring external runtime downloads.
"""

from __future__ import annotations

import re
import string
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Built-in English stopwords to avoid external runtime download dependencies
STANDARD_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves"
}

URL_REGEX = re.compile(r"https?://\S+|www\.\S+")
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
NUMBER_REGEX = re.compile(r"\b\d+\b")
CURRENCY_REGEX = re.compile(r"[$€£¥]\s*\d+")


def clean_email_text(
    text: str,
    remove_stopwords: bool = True,
    normalize_special_tokens: bool = True,
) -> str:
    """Preprocess a single raw email string into clean, normalized tokens.

    Parameters
    ----------
    text : str
        Raw email text string.
    remove_stopwords : bool, default=True
        Whether to filter out common English stopwords.
    normalize_special_tokens : bool, default=True
        Whether to map URLs and currency tokens into normalized placeholders.

    Returns
    -------
    str
        Sanitized text string.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase conversion
    cleaned = text.lower()

    # 2. Normalize URLs and emails
    if normalize_special_tokens:
        cleaned = URL_REGEX.sub(" urltoken ", cleaned)
        cleaned = EMAIL_REGEX.sub(" emailtoken ", cleaned)
        cleaned = CURRENCY_REGEX.sub(" moneytoken ", cleaned)
        cleaned = NUMBER_REGEX.sub(" numtoken ", cleaned)
    else:
        cleaned = URL_REGEX.sub(" ", cleaned)
        cleaned = EMAIL_REGEX.sub(" ", cleaned)
        cleaned = NUMBER_REGEX.sub(" ", cleaned)

    # 3. Punctuation removal
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))

    # 4. Tokenization & Stopword filtering
    tokens = cleaned.split()
    if remove_stopwords:
        tokens = [w for w in tokens if w not in STANDARD_STOPWORDS and len(w) > 1]
    else:
        tokens = [w for w in tokens if len(w) > 1]

    # 5. Rejoin into clean string
    return " ".join(tokens)


def preprocess_corpus(
    df: pd.DataFrame,
    text_column: str = "message",
    target_column: str = "label",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clean full email dataset, remove duplicates, and build sanitized message column.

    Parameters
    ----------
    df : pd.DataFrame
        Raw email records dataframe.
    text_column : str, default='message'
        Column name holding raw email text.
    target_column : str, default='label'
        Binary label column (0 = Ham, 1 = Spam).

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, int]]
        Cleaned dataframe with 'cleaned_message' and diagnostic summary dictionary.
    """
    initial_count = len(df)
    cleaned_df = df.copy()

    # Deduplication
    dupes = int(cleaned_df.duplicated(subset=[text_column]).sum())
    cleaned_df = cleaned_df.drop_duplicates(subset=[text_column]).reset_index(drop=True)

    # Apply text cleaning
    cleaned_df["cleaned_message"] = cleaned_df[text_column].apply(clean_email_text)

    # Filter out empty texts
    cleaned_df = cleaned_df[cleaned_df["cleaned_message"].str.len() > 0].reset_index(drop=True)

    # Add text length statistics
    cleaned_df["char_count"] = cleaned_df[text_column].astype(str).str.len()
    cleaned_df["word_count"] = cleaned_df["cleaned_message"].str.split().apply(len)

    report = {
        "initial_emails": initial_count,
        "final_emails": len(cleaned_df),
        "duplicates_removed": dupes,
        "spam_count": int(cleaned_df[target_column].sum()),
        "ham_count": int((cleaned_df[target_column] == 0).sum()),
        "spam_ratio_pct": round(float(cleaned_df[target_column].mean() * 100.0), 2),
    }

    return cleaned_df, report


def split_email_dataset(
    df: pd.DataFrame,
    text_column: str = "cleaned_message",
    target_column: str = "label",
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split corpus into train and test sets with stratification on the spam label.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned email dataframe.
    text_column : str, default='cleaned_message'
        Input text feature.
    target_column : str, default='label'
        Binary target.
    test_size : float, default=0.20
        Held-out evaluation ratio.
    random_state : int, default=42
        Reproducible random seed.

    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    X = df[text_column]
    y = df[target_column]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
