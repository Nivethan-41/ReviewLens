"""Unit tests for ReviewLens preprocessing and normalization functions (ml/preprocess.py)."""

from __future__ import annotations

import pytest

from ml.preprocess import clean_text, normalize_for_tfidf


def test_clean_text_url_removal():
    """Verify clean_text removes http, https, and www URLs."""
    raw_texts = [
        "Check this out: https://example.com/product/123 for discounts!",
        "Visit http://deal-site.org?ref=42 now.",
        "Go to www.amazon.com/dp/B000123 for more info.",
        "Multiple links https://a.com and http://b.org/test and www.c.net here.",
    ]
    for raw in raw_texts:
        cleaned = clean_text(raw)
        assert "http" not in cleaned
        assert "www." not in cleaned
        assert ".com" not in cleaned
        assert ".org" not in cleaned
        assert ".net" not in cleaned


def test_clean_text_html_removal():
    """Verify clean_text removes diverse HTML and XML tags."""
    raw = "<p>This is a <b>superb</b> product! <a href='https://spam.com'>Click here</a></p><br/>"
    cleaned = clean_text(raw)
    assert "<" not in cleaned
    assert ">" not in cleaned
    assert "p>" not in cleaned
    assert "href" not in cleaned
    assert "this is a superb product! click here" in cleaned


def test_clean_text_preserves_sentence_punctuation():
    """Verify clean_text keeps exclamation marks, question marks, and periods needed for stylometry."""
    raw = "AMAZING PRODUCT!!! Is this the best purchase ever? Yes, absolutely."
    cleaned = clean_text(raw)
    assert "!" in cleaned
    assert cleaned.count("!") == 3
    assert "?" in cleaned
    assert "." in cleaned
    assert "," in cleaned
    # Lowercase conversion check
    assert "amazing product!!!" in cleaned


def test_clean_text_whitespace_and_unicode_normalization():
    """Verify clean_text normalizes unicode and collapses multiple whitespace characters."""
    raw = "   \u201cGreat\u201d\u00a0\u00a0item!   Multiple   spaces    and\tnewlines\nhere.  "
    cleaned = clean_text(raw)
    # Curly quotes normalized to NFKC (or kept as clean chars without multi-spaces)
    assert "  " not in cleaned
    assert "\t" not in cleaned
    assert "\n" not in cleaned
    assert not cleaned.startswith(" ")
    assert not cleaned.endswith(" ")


def test_clean_text_empty_and_edge_cases():
    """Verify clean_text handles empty strings, whitespace, None, and non-strings without crashing."""
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text(None) == ""
    assert clean_text(12345) == ""
    assert clean_text([]) == ""


def test_clean_text_is_pure_and_idempotent():
    """Verify clean_text is a pure function producing identical results on repeated calls."""
    raw = "Terrific sound quality! Check https://audio.com <p>Fast shipping</p>."
    res1 = clean_text(raw)
    res2 = clean_text(raw)
    assert res1 == res2
    assert clean_text(res1) == res1


def test_normalize_for_tfidf_keeps_negations():
    """Verify normalize_for_tfidf removes regular stopwords but retains negations."""
    raw = "This is not a good product, I did not like it, there is no value, and I will never buy again."
    normalized = normalize_for_tfidf(raw)
    tokens = normalized.split()

    # Negation words must be retained
    assert "not" in tokens
    assert "no" in tokens
    assert "never" in tokens

    # Common stopwords must be removed
    assert "is" not in tokens
    assert "this" not in tokens
    assert "a" not in tokens
    assert "and" not in tokens
    assert "will" not in tokens


def test_normalize_for_tfidf_contraction_negations():
    """Verify contractions with negations like don't, won't, didn't retain their negation."""
    raw = "I don't recommend this. It didn't work and won't turn on."
    normalized = normalize_for_tfidf(raw)
    tokens = normalized.split()

    # Contraction negations retained
    assert any(neg in tokens for neg in ["don't", "dont"])
    assert any(neg in tokens for neg in ["didn't", "didnt"])
    assert any(neg in tokens for neg in ["won't", "wont"])


def test_normalize_for_tfidf_lemmatization():
    """Verify normalize_for_tfidf lemmatizes nouns and inflected words."""
    raw = "The batteries and cables arrived within two days."
    normalized = normalize_for_tfidf(raw)
    tokens = normalized.split()

    # Plural nouns should be lemmatized to singular forms
    assert "battery" in tokens
    assert "cable" in tokens
    assert "day" in tokens


def test_normalize_for_tfidf_url_and_html_cleaning():
    """Verify normalize_for_tfidf removes URLs and HTML tags before tokenizing."""
    raw = "<div>Great phone with amazing camera! See https://phone.org</div>"
    normalized = normalize_for_tfidf(raw)
    assert "https" not in normalized
    assert "phone.org" not in normalized
    assert "div" not in normalized
    assert "camera" in normalized


def test_normalize_for_tfidf_empty_and_edge_cases():
    """Verify normalize_for_tfidf handles empty string, whitespace, None, and edge cases."""
    assert normalize_for_tfidf("") == ""
    assert normalize_for_tfidf("   ") == ""
    assert normalize_for_tfidf(None) == ""
    assert normalize_for_tfidf(12345) == ""
    assert normalize_for_tfidf("!!! ??? ...") == ""
