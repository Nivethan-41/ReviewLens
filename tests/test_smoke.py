"""Smoke tests verifying module imports and configuration constants for ReviewLens."""

import importlib
import pytest
from ml.config import (
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    MODEL_PATH,
    RANDOM_SEED,
    TEST_SIZE,
    VAL_SIZE,
    MODEL_VERSION,
    TFIDF_WORD_NGRAM_RANGE,
    TFIDF_CHAR_NGRAM_RANGE,
    TFIDF_MAX_FEATURES,
)


def test_import_ml_modules():
    """Verify that all ml package modules can be imported without error."""
    ml_modules = [
        "ml.config",
        "ml.data",
        "ml.preprocess",
        "ml.features",
        "ml.train",
        "ml.evaluate",
        "ml.explain",
        "ml.reliability",
        "ml.predict",
    ]
    for module_name in ml_modules:
        mod = importlib.import_module(module_name)
        assert mod is not None, f"Failed to import {module_name}"


def test_config_constants():
    """Verify that all required config constants are set and of expected types."""
    assert PROJECT_ROOT.exists() and PROJECT_ROOT.is_dir()
    assert DATA_DIR.name == "data"
    assert RAW_DATA_DIR.name == "raw"
    assert PROCESSED_DATA_DIR.name == "processed"
    assert MODELS_DIR.name == "models"
    assert MODEL_PATH.name == "model.joblib"

    assert RANDOM_SEED == 42
    assert TEST_SIZE == 0.1
    assert VAL_SIZE == 0.1

    assert TFIDF_WORD_NGRAM_RANGE == (1, 2)
    assert TFIDF_CHAR_NGRAM_RANGE == (3, 5)
    assert TFIDF_MAX_FEATURES > 0
    assert isinstance(MODEL_VERSION, str) and len(MODEL_VERSION) > 0

