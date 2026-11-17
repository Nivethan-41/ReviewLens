"""Text normalization and preprocessing pipeline for review text.

Provides pure, efficient text cleaning and TF-IDF normalization functions:
- clean_text: Lowercases, normalizes unicode, strips URLs and HTML, collapses whitespace,
  while keeping punctuation needed for stylometric analysis.
- normalize_for_tfidf: Tokenizes, removes stopwords except negations, and lemmatizes.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Pre-compiled regular expressions for speed
_URL_PATTERN: re.Pattern[str] = re.compile(
    r"https?://\S+|www\.\S+", re.IGNORECASE
)
_HTML_PATTERN: re.Pattern[str] = re.compile(r"<[^>]+>")
_WHITESPACE_PATTERN: re.Pattern[str] = re.compile(r"\s+")
_TOKEN_PATTERN: re.Pattern[str] = re.compile(r"\b[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?\b")

# Negation words to explicitly retain for sentiment and deception context
NEGATION_WORDS: frozenset[str] = frozenset({
    "no",
    "not",
    "nor",
    "never",
    "none",
    "neither",
    "nobody",
    "nowhere",
    "cannot",
    "without",
    "hardly",
    "scarcely",
    "barely",
    "don't",
    "dont",
    "doesn't",
    "doesnt",
    "didn't",
    "didnt",
    "won't",
    "wont",
    "wouldn't",
    "wouldnt",
    "can't",
    "cant",
    "couldn't",
    "couldnt",
    "shouldn't",
    "shouldnt",
    "mustn't",
    "mustnt",
    "isn't",
    "isnt",
    "aren't",
    "arent",
    "wasn't",
    "wasnt",
    "weren't",
    "werent",
    "hasn't",
    "hasnt",
    "haven't",
    "havent",
    "hadn't",
    "hadnt",
    "ain't",
    "aint",
    "n't",
})

# Initialize NLTK stopwords and lemmatizer with automatic fallback
try:
    _BASE_STOPWORDS: set[str] = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    _BASE_STOPWORDS = set(stopwords.words("english"))

# Stopwords set excluding negations
_TFIDF_STOPWORDS: frozenset[str] = frozenset(_BASE_STOPWORDS - NEGATION_WORDS)

try:
    _LEMMATIZER: WordNetLemmatizer = WordNetLemmatizer()
    _LEMMATIZER.lemmatize("tests")
except LookupError:
    nltk.download("wordnet", quiet=True)
    _LEMMATIZER = WordNetLemmatizer()


def clean_text(text: Any) -> str:
    """Clean raw review text while preserving sentence punctuation needed for stylometry.

    Performs:
    1. Type validation (gracefully returns empty string for None or non-str).
    2. Unicode normalization (NFKC).
    3. Lowercasing.
    4. Removal of web URLs (http, https, www).
    5. Removal of HTML/XML tags.
    6. Collapsing extra whitespace and stripping.

    Args:
        text: Raw review text (string or Any).

    Returns:
        Cleaned text string.
    """
    if text is None or not isinstance(text, str):
        return ""

    # Unicode normalization to standard compatibility decomposition + composition
    normalized = unicodedata.normalize("NFKC", text)

    # Lowercase
    lowered = normalized.lower()

    # Remove HTML tags first (replaces with space to prevent word joining across tags)
    no_html = _HTML_PATTERN.sub(" ", lowered)

    # Remove URLs
    no_urls = _URL_PATTERN.sub("", no_html)

    # Collapse repeated whitespace and strip edges
    cleaned = _WHITESPACE_PATTERN.sub(" ", no_urls).strip()

    return cleaned


def normalize_for_tfidf(text: Any) -> str:
    """Normalize text for TF-IDF vectorization: clean, tokenize, filter stopwords (keep negations), lemmatize.

    Performs:
    1. Preprocessing via clean_text (lowercasing, unicode, URLs, HTML tags, whitespace).
    2. Word tokenization preserving contractions.
    3. Stopword removal while explicitly keeping negation words (e.g., 'not', 'no', 'never', 'don't').
    4. Lemmatization using WordNet.
    5. Joining filtered tokens into a space-separated string.

    Args:
        text: Raw or cleaned review text.

    Returns:
        Normalized string of tokens suitable for word n-gram extraction.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return ""

    tokens = _TOKEN_PATTERN.findall(cleaned)
    if not tokens:
        return ""

    normalized_tokens: list[str] = []
    for token in tokens:
        # Filter out stopwords, but preserve negations
        if token in _TFIDF_STOPWORDS:
            continue
        lemmatized = _LEMMATIZER.lemmatize(token)
        normalized_tokens.append(lemmatized)

    return " ".join(normalized_tokens)
