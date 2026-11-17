"""Feature engineering module extracting TF-IDF and stylometric features.

Builds a scikit-learn compatible pipeline combining:
1. Word TF-IDF (1, 2)-grams on normalized text.
2. Character TF-IDF (3, 5)-grams on cleaned text.
3. Scaled stylometric features extracted from raw review text.
"""

from __future__ import annotations

import collections
from pathlib import Path
import re
from typing import Any

import joblib
import nltk
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from ml.config import (
    FEATURES_PATH,
    TFIDF_CHAR_MAX_FEATURES,
    TFIDF_CHAR_NGRAM_RANGE,
    TFIDF_WORD_MAX_FEATURES,
    TFIDF_WORD_NGRAM_RANGE,
)
from ml.preprocess import clean_text, normalize_for_tfidf

# Regular expressions and linguistic constants for stylometrics
_WORD_RE: re.Pattern[str] = re.compile(r"\b[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?\b")

_FIRST_PERSON_PRONOUNS: frozenset[str] = frozenset({
    "i",
    "me",
    "my",
    "myself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
})

_SECOND_PERSON_PRONOUNS: frozenset[str] = frozenset({
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
})

_SUPERLATIVE_RE: re.Pattern[str] = re.compile(
    r"\b(?:best|amazing|perfect|worst|incredible)\b|100%", re.IGNORECASE
)

# Ordered feature names produced by StylometricFeatures
STYLOMETRIC_FEATURE_NAMES: list[str] = [
    "word_count",
    "char_count",
    "avg_word_length",
    "exclamation_count",
    "exclamation_ratio",
    "question_mark_ratio",
    "uppercase_word_ratio",
    "first_person_ratio",
    "second_person_ratio",
    "superlative_count",
    "lexical_diversity",
    "repeated_word_ratio",
    "digit_ratio",
    "sentence_count",
    "avg_sentence_length",
]


def _extract_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK sent_tokenize with regex fallback."""
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception:
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    return [s for s in sentences if s.strip()]


class StylometricFeatures(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer extracting 15 stylometric features per review.

    Features computed:
    1. word_count: Total word count.
    2. char_count: Total character length of raw text.
    3. avg_word_length: Average character length per word.
    4. exclamation_count: Total count of '!' marks.
    5. exclamation_ratio: Ratio of '!' marks to total characters.
    6. question_mark_ratio: Ratio of '?' marks to total characters.
    7. uppercase_word_ratio: Ratio of all-uppercase words to total words.
    8. first_person_ratio: Ratio of first-person pronouns (I, me, my, we, etc.) to words.
    9. second_person_ratio: Ratio of second-person pronouns (you, your, etc.) to words.
    10. superlative_count: Count of superlatives and intensifiers (best, amazing, perfect, worst, incredible, 100%).
    11. lexical_diversity: Type-Token Ratio (unique words / total words).
    12. repeated_word_ratio: Ratio of tokens belonging to repeated words to total words.
    13. digit_ratio: Ratio of numeric digit characters to total characters.
    14. sentence_count: Number of sentences in the review.
    15. avg_sentence_length: Average number of words per sentence.
    """

    def __init__(self) -> None:
        self.feature_names_ = list(STYLOMETRIC_FEATURE_NAMES)

    def fit(self, X: Any, y: Any = None) -> StylometricFeatures:
        """Fit transformer (no-op, sets feature names).

        Args:
            X: Input review texts.
            y: Optional labels (unused).

        Returns:
            self
        """
        self.n_features_in_ = 1
        return self

    def _compute_single(self, text: Any) -> list[float]:
        """Compute the 15 stylometric features for a single review text."""
        if not isinstance(text, str) or not text.strip():
            char_count = float(len(text)) if isinstance(text, str) else 0.0
            return [
                0.0,  # word_count
                char_count,  # char_count
                0.0,  # avg_word_length
                float(text.count("!")) if isinstance(text, str) else 0.0,  # exclamation_count
                0.0,  # exclamation_ratio
                0.0,  # question_mark_ratio
                0.0,  # uppercase_word_ratio
                0.0,  # first_person_ratio
                0.0,  # second_person_ratio
                0.0,  # superlative_count
                0.0,  # lexical_diversity
                0.0,  # repeated_word_ratio
                0.0,  # digit_ratio
                0.0,  # sentence_count
                0.0,  # avg_sentence_length
            ]

        char_count = float(len(text))
        words = _WORD_RE.findall(text)
        word_count = float(len(words))

        # Average word length
        avg_word_len = (
            sum(len(w) for w in words) / word_count if word_count > 0 else 0.0
        )

        # Exclamation and question mark metrics
        exclamation_count = float(text.count("!"))
        exclamation_ratio = (
            exclamation_count / char_count if char_count > 0 else 0.0
        )
        question_mark_ratio = (
            float(text.count("?")) / char_count if char_count > 0 else 0.0
        )

        # Uppercase word ratio (all-caps alphabetic tokens)
        upper_count = sum(
            1 for w in words if w.isupper() and any(c.isalpha() for c in w)
        )
        uppercase_word_ratio = (
            float(upper_count) / word_count if word_count > 0 else 0.0
        )

        # Lowercased word analysis
        lower_words = [w.lower() for w in words]

        # Pronoun ratios
        fp_count = sum(1 for w in lower_words if w in _FIRST_PERSON_PRONOUNS)
        first_person_ratio = (
            float(fp_count) / word_count if word_count > 0 else 0.0
        )

        sp_count = sum(1 for w in lower_words if w in _SECOND_PERSON_PRONOUNS)
        second_person_ratio = (
            float(sp_count) / word_count if word_count > 0 else 0.0
        )

        # Superlative and intensifier count
        superlative_count = float(len(_SUPERLATIVE_RE.findall(text)))

        # Lexical diversity (Type-Token Ratio)
        unique_words = len(set(lower_words))
        lexical_diversity = (
            float(unique_words) / word_count if word_count > 0 else 0.0
        )

        # Repeated-word ratio (tokens belonging to word types appearing > 1 time)
        word_freqs = collections.Counter(lower_words)
        repeated_tokens = sum(cnt for cnt in word_freqs.values() if cnt > 1)
        repeated_word_ratio = (
            float(repeated_tokens) / word_count if word_count > 0 else 0.0
        )

        # Digit ratio
        digit_count = sum(1 for c in text if c.isdigit())
        digit_ratio = (
            float(digit_count) / char_count if char_count > 0 else 0.0
        )

        # Sentence metrics
        sentences = _extract_sentences(text)
        sentence_count = float(len(sentences))
        avg_sentence_length = (
            word_count / sentence_count if sentence_count > 0 else 0.0
        )

        return [
            word_count,
            char_count,
            avg_word_len,
            exclamation_count,
            exclamation_ratio,
            question_mark_ratio,
            uppercase_word_ratio,
            first_person_ratio,
            second_person_ratio,
            superlative_count,
            lexical_diversity,
            repeated_word_ratio,
            digit_ratio,
            sentence_count,
            avg_sentence_length,
        ]

    def transform(self, X: Any) -> np.ndarray:
        """Compute stylometric features for an array/series of texts.

        Args:
            X: Iterable of review strings (DataFrame, Series, 1D array, or list).

        Returns:
            2D numpy array of shape (n_samples, 15) with float64 values.
        """
        texts = _to_text_list(X)
        if not texts:
            return np.empty((0, len(self.feature_names_)), dtype=np.float64)

        rows = [self._compute_single(text) for text in texts]
        return np.array(rows, dtype=np.float64)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return stable, documented feature names.

        Args:
            input_features: Optional input feature names (unused).

        Returns:
            Numpy array of string feature names.
        """
        return np.array(self.feature_names_, dtype=object)


class TextSelector(BaseEstimator, TransformerMixin):
    """Extract review text column from DataFrame, Series, array, or list."""

    def __init__(self, key: str = "text") -> None:
        self.key = key

    def fit(self, X: Any, y: Any = None) -> TextSelector:
        """Fit selector (no-op)."""
        return self

    def transform(self, X: Any) -> list[str]:
        """Convert input container to a list of raw strings."""
        return _to_text_list(X, key=self.key)

    def get_feature_names_out(self, input_features: Any = None) -> Any:
        """Pass-through feature names."""
        return input_features


def _to_text_list(X: Any, key: str = "text") -> list[str]:
    """Helper to safely extract a list of strings from various input types."""
    if isinstance(X, pd.DataFrame):
        if key in X.columns:
            return X[key].fillna("").astype(str).tolist()
        return X.iloc[:, 0].fillna("").astype(str).tolist()
    if isinstance(X, pd.Series):
        return X.fillna("").astype(str).tolist()
    if isinstance(X, np.ndarray):
        if X.ndim == 2:
            return [str(v) if v is not None else "" for v in X[:, 0]]
        return [str(v) if v is not None else "" for v in X]
    if isinstance(X, (list, tuple)):
        return [str(v) if v is not None else "" for v in X]
    return [str(X)]


def build_feature_pipeline() -> Pipeline:
    """Build an unfitted scikit-learn feature extraction pipeline.

    Combines:
    1. Word TF-IDF (1, 2)-grams on normalized text.
    2. Character TF-IDF (3, 5)-grams on cleaned text.
    3. Scaled stylometric features on raw text.

    Uses settings from ml/config.py.

    Returns:
        Unfitted Pipeline combining TextSelector and FeatureUnion.
    """
    word_tfidf = TfidfVectorizer(
        ngram_range=TFIDF_WORD_NGRAM_RANGE,
        max_features=TFIDF_WORD_MAX_FEATURES,
        token_pattern=r"(?u)\b\w+\b",
        preprocessor=normalize_for_tfidf,
    )

    char_tfidf = TfidfVectorizer(
        ngram_range=TFIDF_CHAR_NGRAM_RANGE,
        max_features=TFIDF_CHAR_MAX_FEATURES,
        analyzer="char",
        preprocessor=clean_text,
    )

    stylo_pipeline = Pipeline([
        ("extract", StylometricFeatures()),
        ("scale", StandardScaler()),
    ])

    features = FeatureUnion([
        ("word_tfidf", word_tfidf),
        ("char_tfidf", char_tfidf),
        ("stylometric", stylo_pipeline),
    ])

    pipeline = Pipeline([
        ("selector", TextSelector()),
        ("features", features),
    ])

    return pipeline


def save_feature_pipeline(
    pipeline: Pipeline, path: Path | str = FEATURES_PATH
) -> Path:
    """Save fitted feature pipeline to disk using joblib.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        path: Destination file path. Defaults to FEATURES_PATH (models/features.joblib).

    Returns:
        Path to the saved artifact.
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)
    return save_path


def load_feature_pipeline(path: Path | str = FEATURES_PATH) -> Pipeline:
    """Load a previously saved feature pipeline from disk using joblib.

    Args:
        path: Path to the .joblib file. Defaults to FEATURES_PATH (models/features.joblib).

    Returns:
        Loaded scikit-learn Pipeline.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(
            f"Feature pipeline artifact not found at: {load_path}"
        )
    return joblib.load(load_path)


def fit_and_save_feature_pipeline(
    X: Any, save_path: Path | str = FEATURES_PATH
) -> Pipeline:
    """Fit a new feature pipeline on dataset X and save the artifact.

    Args:
        X: Dataset (DataFrame with 'text' column, Series, or list of raw texts).
        save_path: Destination path. Defaults to FEATURES_PATH (models/features.joblib).

    Returns:
        Fitted Pipeline instance.
    """
    pipeline = build_feature_pipeline()
    pipeline.fit(X)
    save_feature_pipeline(pipeline, save_path)
    return pipeline
