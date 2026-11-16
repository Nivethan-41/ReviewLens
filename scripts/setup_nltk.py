"""Helper script to download required NLTK resources for ReviewLens."""

import sys
from pathlib import Path

# Add project root to sys.path so ml package can be imported directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import download_nltk_resources

if __name__ == "__main__":
    print("Downloading NLTK resources (stopwords, punkt, wordnet)...")
    download_nltk_resources()
    print("NLTK resources setup complete.")
