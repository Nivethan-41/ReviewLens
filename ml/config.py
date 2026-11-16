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

MODELS_DIR: Path = PROJECT_ROOT / "models"
MODEL_PATH: Path = MODELS_DIR / "model.joblib"
METRICS_PATH: Path = MODELS_DIR / "metrics.json"

NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

# Global Random Seed & Train/Val/Test Split Ratios
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.1
VAL_SIZE: float = 0.1

# TF-IDF Feature Extraction Settings
TFIDF_WORD_NGRAM_RANGE: Tuple[int, int] = (1, 2)
TFIDF_CHAR_NGRAM_RANGE: Tuple[int, int] = (3, 5)
TFIDF_WORD_MAX_FEATURES: int = 10000
TFIDF_CHAR_MAX_FEATURES: int = 10000
TFIDF_MAX_FEATURES: int = 20000

# Model Version
MODEL_VERSION: str = "0.1.0"

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

