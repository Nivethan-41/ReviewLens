"""Unit tests for ReviewLens feature extraction pipeline and stylometric features (ml/features.py)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from ml.config import PROCESSED_DATA_DIR
from ml.features import (
    STYLOMETRIC_FEATURE_NAMES,
    StylometricFeatures,
    build_feature_pipeline,
    fit_and_save_feature_pipeline,
    load_feature_pipeline,
    save_feature_pipeline,
)


def test_stylometric_features_hand_made_reviews():
    """Verify all 15 stylometric feature calculations on 3 hand-made review texts."""
    transformer = StylometricFeatures()

    # Hand-made Review 1:
    # "AMAZING PRODUCT! BEST EVER!"
    # Words: ["AMAZING", "PRODUCT", "BEST", "EVER"] -> word_count = 4
    # Char count: 27
    # Avg word length: (7 + 7 + 4 + 4) / 4 = 5.5
    # Exclamation count: 2
    # Exclamation ratio: 2 / 27
    # Question mark ratio: 0.0
    # Uppercase-word ratio: 4 / 4 = 1.0 (all 4 words uppercase)
    # First-person ratio: 0.0
    # Second-person ratio: 0.0
    # Superlative count: 2 ("AMAZING", "BEST")
    # Lexical diversity: 4 / 4 = 1.0
    # Repeated-word ratio: 0.0
    # Digit ratio: 0.0
    # Sentence count: 2 ("AMAZING PRODUCT!", "BEST EVER!")
    # Avg sentence length: 4 / 2 = 2.0
    rev1 = "AMAZING PRODUCT! BEST EVER!"
    feats1 = transformer.transform([rev1])[0]
    names = transformer.get_feature_names_out().tolist()
    d1 = dict(zip(names, feats1))

    assert d1["word_count"] == 4.0
    assert d1["char_count"] == 27.0
    assert d1["avg_word_length"] == pytest.approx(5.5)
    assert d1["exclamation_count"] == 2.0
    assert d1["exclamation_ratio"] == pytest.approx(2.0 / 27.0)
    assert d1["question_mark_ratio"] == 0.0
    assert d1["uppercase_word_ratio"] == 1.0
    assert d1["first_person_ratio"] == 0.0
    assert d1["second_person_ratio"] == 0.0
    assert d1["superlative_count"] == 2.0
    assert d1["lexical_diversity"] == 1.0
    assert d1["repeated_word_ratio"] == 0.0
    assert d1["digit_ratio"] == 0.0
    assert d1["sentence_count"] == 2.0
    assert d1["avg_sentence_length"] == pytest.approx(2.0)

    # Hand-made Review 2:
    # "I bought 2 items and I loved them. Why did you not tell me?"
    # Words: ["I", "bought", "2", "items", "and", "I", "loved", "them", "Why", "did", "you", "not", "tell", "me"] -> 14 words
    # Char count: 59
    # Word lengths: [1, 6, 1, 5, 3, 1, 5, 4, 3, 3, 3, 3, 4, 2] -> sum = 44. Avg = 44 / 14 = 3.142857...
    # Exclamation count: 0, ratio: 0.0
    # Question mark ratio: 1 / 59 = 0.016949...
    # Uppercase-word ratio: "I" (appears 2 times) -> 2 / 14 = 0.142857...
    # First-person ratio: "I", "I", "me" -> 3 / 14 = 0.2142857...
    # Second-person ratio: "you" -> 1 / 14 = 0.0714285...
    # Superlative count: 0
    # Lexical diversity: 13 unique / 14 words = 0.9285714...
    # Repeated-word ratio: "I" repeats (2 tokens) -> 2 / 14 = 0.142857...
    # Digit ratio: 1 digit ('2') / 59 chars = 0.016949...
    # Sentence count: 2 ("I bought 2 items and I loved them.", "Why did you not tell me?")
    # Avg sentence length: 14 / 2 = 7.0
    rev2 = "I bought 2 items and I loved them. Why did you not tell me?"
    feats2 = transformer.transform([rev2])[0]
    d2 = dict(zip(names, feats2))

    assert d2["word_count"] == 14.0
    assert d2["char_count"] == 59.0
    assert d2["avg_word_length"] == pytest.approx(44.0 / 14.0)
    assert d2["exclamation_count"] == 0.0
    assert d2["exclamation_ratio"] == 0.0
    assert d2["question_mark_ratio"] == pytest.approx(1.0 / 59.0)
    assert d2["uppercase_word_ratio"] == pytest.approx(2.0 / 14.0)
    assert d2["first_person_ratio"] == pytest.approx(3.0 / 14.0)
    assert d2["second_person_ratio"] == pytest.approx(1.0 / 14.0)
    assert d2["superlative_count"] == 0.0
    assert d2["lexical_diversity"] == pytest.approx(13.0 / 14.0)
    assert d2["repeated_word_ratio"] == pytest.approx(2.0 / 14.0)
    assert d2["digit_ratio"] == pytest.approx(1.0 / 59.0)
    assert d2["sentence_count"] == 2.0
    assert d2["avg_sentence_length"] == pytest.approx(7.0)

    # Hand-made Review 3:
    # "great great great! 100% perfect quality."
    # Words: ["great", "great", "great", "100", "perfect", "quality"] -> 6 words
    # Char count: 40
    # Word lengths: [5, 5, 5, 3, 7, 7] -> sum = 32. Avg = 32 / 6 = 5.333333...
    # Exclamation count: 1, ratio: 1 / 40 = 0.025
    # Question mark ratio: 0.0
    # Uppercase-word ratio: 0.0
    # First-person ratio: 0.0
    # Second-person ratio: 0.0
    # Superlative count: 2 ("100%", "perfect")
    # Lexical diversity: 4 unique ("great", "100", "perfect", "quality") / 6 = 0.666666...
    # Repeated-word ratio: "great" repeated (3 tokens) -> 3 / 6 = 0.5
    # Digit ratio: 3 digits ('1', '0', '0') / 40 chars = 0.075
    # Sentence count: 2 ("great great great!", "100% perfect quality.")
    # Avg sentence length: 6 / 2 = 3.0
    rev3 = "great great great! 100% perfect quality."
    feats3 = transformer.transform([rev3])[0]
    d3 = dict(zip(names, feats3))

    assert d3["word_count"] == 6.0
    assert d3["char_count"] == 40.0
    assert d3["avg_word_length"] == pytest.approx(32.0 / 6.0)
    assert d3["exclamation_count"] == 1.0
    assert d3["exclamation_ratio"] == pytest.approx(0.025)
    assert d3["question_mark_ratio"] == 0.0
    assert d3["uppercase_word_ratio"] == 0.0
    assert d3["first_person_ratio"] == 0.0
    assert d3["second_person_ratio"] == 0.0
    assert d3["superlative_count"] == 2.0
    assert d3["lexical_diversity"] == pytest.approx(4.0 / 6.0)
    assert d3["repeated_word_ratio"] == pytest.approx(0.5)
    assert d3["digit_ratio"] == pytest.approx(0.075)
    assert d3["sentence_count"] == 2.0
    assert d3["avg_sentence_length"] == pytest.approx(3.0)


def test_stylometric_feature_names_out():
    """Verify get_feature_names_out returns the stable, documented 15 names."""
    transformer = StylometricFeatures()
    names = transformer.get_feature_names_out()
    assert isinstance(names, np.ndarray)
    assert len(names) == 15
    assert list(names) == STYLOMETRIC_FEATURE_NAMES


def test_stylometric_empty_and_whitespace():
    """Verify StylometricFeatures handles empty and whitespace-only strings without NaNs."""
    transformer = StylometricFeatures()
    res = transformer.transform(["", "   ", "!!!", None])
    assert res.shape == (4, 15)
    assert not np.isnan(res).any()
    assert not np.isinf(res).any()
    # Check that empty string outputs zeros
    assert np.all(res[0] == 0.0)


def test_pipeline_output_shape_consistency():
    """Verify feature pipeline fit and transform yield consistent feature dimensions."""
    pipeline = build_feature_pipeline()

    sample_texts = [
        "AMAZING PRODUCT! BEST EVER!",
        "Poor build quality, broke within two days.",
        "Average item, does what it says on the box.",
        "Will not buy again! Terrible customer support.",
    ]
    df_sample = pd.DataFrame({"text": sample_texts})

    X_fitted = pipeline.fit_transform(df_sample)
    X_trans = pipeline.transform(df_sample)

    assert X_fitted.shape == X_trans.shape
    assert X_fitted.shape[0] == len(sample_texts)

    # Transform new data
    new_data = pd.DataFrame({"text": ["Another test review."] * 2})
    X_new = pipeline.transform(new_data)
    assert X_new.shape[0] == 2
    assert X_new.shape[1] == X_fitted.shape[1]


def test_pipeline_no_nan_output():
    """Verify feature pipeline output contains no NaN or Inf values."""
    pipeline = build_feature_pipeline()

    diverse_texts = [
        "Normal review text.",
        "",  # Empty
        "   ",  # Spaces
        "100% AMAZING BEST INCREDIBLE PERFECT WORST!!!",
        "https://spam.com <script>alert('x')</script>",
        "Short.",
    ]
    df = pd.DataFrame({"text": diverse_texts})

    X_mat = pipeline.fit_transform(df)
    dense_mat = X_mat.toarray() if sparse.issparse(X_mat) else X_mat

    assert not np.isnan(dense_mat).any()
    assert not np.isinf(dense_mat).any()


def test_pipeline_input_types():
    """Verify pipeline accepts DataFrame with 'text' col, Series, and list of strings."""
    pipeline = build_feature_pipeline()
    raw = [
        "Great quality and fast delivery.",
        "Worst purchase ever made, do not buy!",
    ]

    # Test list of strings
    X_list = pipeline.fit_transform(raw)
    assert X_list.shape[0] == 2

    # Test Series
    X_series = pipeline.transform(pd.Series(raw))
    assert X_series.shape == X_list.shape

    # Test DataFrame
    X_df = pipeline.transform(pd.DataFrame({"text": raw}))
    assert X_df.shape == X_list.shape


def test_pipeline_fit_on_train_and_transform_on_val():
    """Verify pipeline fits cleanly on train.csv and transforms val.csv."""
    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"

    assert train_path.exists(), f"Missing train.csv at {train_path}"
    assert val_path.exists(), f"Missing val.csv at {val_path}"

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    pipeline = build_feature_pipeline()
    X_train = pipeline.fit_transform(df_train)
    X_val = pipeline.transform(df_val)

    assert X_train.shape[0] == len(df_train)
    assert X_val.shape[0] == len(df_val)
    assert X_train.shape[1] == X_val.shape[1]

    dense_val = X_val.toarray() if sparse.issparse(X_val) else X_val
    assert not np.isnan(dense_val).any()


def test_pipeline_save_and_load(tmp_path: Path):
    """Verify fit_and_save and load helpers work correctly with joblib."""
    df_train = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    artifact_path = tmp_path / "test_features.joblib"

    fitted_pipe = fit_and_save_feature_pipeline(df_train, save_path=artifact_path)
    assert artifact_path.exists()

    loaded_pipe = load_feature_pipeline(path=artifact_path)
    val_texts = ["A quick test review to check loaded model.", "Another sample."]

    res_fitted = fitted_pipe.transform(val_texts)
    res_loaded = loaded_pipe.transform(val_texts)

    np.testing.assert_allclose(
        res_fitted.toarray() if sparse.issparse(res_fitted) else res_fitted,
        res_loaded.toarray() if sparse.issparse(res_loaded) else res_loaded,
        rtol=1e-5,
    )
