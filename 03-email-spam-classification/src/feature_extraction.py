"""
Feature Extraction and Vectorization Module for Email Spam Detection.

Provides Bag-of-Words (CountVectorizer) and Term Frequency-Inverse Document Frequency
(TfidfVectorizer) feature extractors with n-gram support and vocabulary introspection.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_bow_vectorizer(
    max_features: Optional[int] = 3000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
) -> CountVectorizer:
    """Build a Bag-of-Words (CountVectorizer) text transformer.

    Parameters
    ----------
    max_features : int, optional, default=3000
        Maximum vocabulary size.
    ngram_range : Tuple[int, int], default=(1, 2)
        Unigram and bigram bounds.
    min_df : int, default=2
        Minimum document frequency cutoff.

    Returns
    -------
    CountVectorizer
        Configured Bag-of-Words vectorizer.
    """
    return CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
    )


def build_tfidf_vectorizer(
    max_features: Optional[int] = 3000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """Build a TF-IDF (TfidfVectorizer) text transformer.

    Parameters
    ----------
    max_features : int, optional, default=3000
        Maximum vocabulary size.
    ngram_range : Tuple[int, int], default=(1, 2)
        Unigram and bigram bounds.
    min_df : int, default=2
        Minimum document frequency cutoff.
    sublinear_tf : bool, default=True
        Apply sublinear scaling (1 + log(tf)) to dampen high-frequency tokens.

    Returns
    -------
    TfidfVectorizer
        Configured TF-IDF vectorizer.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )


def extract_top_indicative_keywords(
    vectorizer: CountVectorizer | TfidfVectorizer,
    X_text: pd.Series | List[str],
    y_labels: pd.Series | np.ndarray,
    top_n: int = 15,
) -> Dict[str, List[Tuple[str, float]]]:
    """Identify the most discriminating keywords for Spam vs Ham classes.

    Parameters
    ----------
    vectorizer : CountVectorizer or TfidfVectorizer
        Fitted vectorizer.
    X_text : Series or list of str
        Text messages.
    y_labels : Series or array
        Binary labels (0 = Ham, 1 = Spam).

    Returns
    -------
    Dict[str, List[Tuple[str, float]]]
        Dictionary with 'ham_keywords' and 'spam_keywords' along with average weights.
    """
    X_mat = vectorizer.transform(X_text)
    feature_names = np.array(vectorizer.get_feature_names_out())
    y_arr = np.asarray(y_labels, dtype=int)

    ham_mask = (y_arr == 0)
    spam_mask = (y_arr == 1)

    ham_mean = np.asarray(X_mat[ham_mask].mean(axis=0)).flatten()
    spam_mean = np.asarray(X_mat[spam_mask].mean(axis=0)).flatten()

    ham_top_idx = ham_mean.argsort()[::-1][:top_n]
    spam_top_idx = spam_mean.argsort()[::-1][:top_n]

    return {
        "ham_keywords": [(feature_names[i], round(float(ham_mean[i]), 4)) for i in ham_top_idx],
        "spam_keywords": [(feature_names[i], round(float(spam_mean[i]), 4)) for i in spam_top_idx],
    }
