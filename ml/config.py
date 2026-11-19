"""Configuration module for ReviewLens.

Defines project-wide constants, filesystem paths, random seeds,
model hyperparameters, and helper utilities.
"""

from pathlib import Path
from typing import List, Tuple
import nltk

# Project Root and Directory Paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

TRAIN_DATA_PATH: Path = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA_PATH: Path = PROCESSED_DATA_DIR / "val.csv"
TEST_DATA_PATH: Path = PROCESSED_DATA_DIR / "test.csv"

MODELS_DIR: Path = PROJECT_ROOT / "models"
MODEL_PATH: Path = MODELS_DIR / "model.joblib"
FEATURES_PATH: Path = MODELS_DIR / "features.joblib"
METRICS_PATH: Path = MODELS_DIR / "metrics.json"
REPORTS_DIR: Path = MODELS_DIR / "reports"
CONFUSION_MATRIX_PATH: Path = REPORTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH: Path = REPORTS_DIR / "roc_curve.png"
CALIBRATION_CURVE_PATH: Path = REPORTS_DIR / "calibration_curve.png"
EVALUATION_REPORT_PATH: Path = REPORTS_DIR / "test_metrics.json"

NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

# Global Random Seed & Train/Val/Test Split Ratios
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.1
VAL_SIZE: float = 0.1

# Confidence and Uncertainty Band Thresholds
UNCERTAIN_LOWER: float = 0.45
UNCERTAIN_UPPER: float = 0.55

# TF-IDF Feature Extraction Settings
TFIDF_WORD_NGRAM_RANGE: Tuple[int, int] = (1, 2)
TFIDF_CHAR_NGRAM_RANGE: Tuple[int, int] = (3, 5)
TFIDF_WORD_MAX_FEATURES: int = 10000
TFIDF_CHAR_MAX_FEATURES: int = 10000
TFIDF_MAX_FEATURES: int = 20000

# Model Version
MODEL_VERSION: str = "0.1.0"

# Reliability Scorer Weights (must sum to 1.0)
RELIABILITY_WEIGHT_NON_SUSPICIOUS: float = 0.45
RELIABILITY_WEIGHT_NEAR_DUPLICATE: float = 0.20
RELIABILITY_WEIGHT_RATING_SKEW: float = 0.15
RELIABILITY_WEIGHT_REVIEW_BURST: float = 0.10
RELIABILITY_WEIGHT_LENGTH_SPECIFICITY: float = 0.10

# Reliability Thresholds and Threshold Boundaries
NEAR_DUPLICATE_SIMILARITY_THRESHOLD: float = 0.85
RELIABILITY_HIGH_THRESHOLD: int = 70
RELIABILITY_MODERATE_THRESHOLD: int = 40
MIN_REVIEWS_FOR_RELIABILITY: int = 5

# Rating Distribution Skew Thresholds
RATING_SKEW_5STAR_THRESHOLD: float = 0.65
RATING_J_SHAPED_THRESHOLD: float = 0.80

# Review Burst Thresholds
BURST_WINDOW_DAYS: int = 3
BURST_MAX_DAY_RATIO: float = 0.35

# Length and Specificity Thresholds
SPECIFICITY_TARGET_WORD_COUNT: int = 40
MAX_REVIEW_TEXT_LENGTH: int = 5000
MAX_REVIEWS_PER_CALL: int = 200

# API and CORS Configuration
API_HOST: str = "127.0.0.1"
API_PORT: int = 8000
CORS_ALLOWED_ORIGINS: List[str] = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
]
CORS_ALLOWED_ORIGIN_REGEX: str = r"^chrome-extension://.*"

# Required NLTK resources
NLTK_PACKAGES: List[str] = ["stopwords", "punkt", "punkt_tab", "wordnet"]


def download_nltk_resources(packages: List[str] | None = None) -> None:
    """Download required NLTK datasets and tokenizers.

    Args:
        packages: Optional list of package names to download. Defaults to NLTK_PACKAGES.
    """
    targets = packages or NLTK_PACKAGES
    for pkg in targets:
        try:
            nltk.download(pkg, quiet=True)
        except Exception as e:
            print(f"Warning: Failed to download NLTK package '{pkg}': {e}")


if __name__ == "__main__":
    download_nltk_resources()
    print("NLTK resources downloaded successfully.")

