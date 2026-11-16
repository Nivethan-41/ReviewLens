# ReviewLens

ReviewLens is an AI-powered browser extension that detects potentially fake product reviews using machine learning and stylometric analysis.

---

## Repository Structure

```
reviewlens/
  data/
    raw/                  # Raw dataset files (git-ignored)
    processed/            # Cleaned, train/val/test split datasets
  notebooks/
    01_eda.ipynb          # Exploratory Data Analysis notebook
  ml/
    config.py             # Global paths, constants, random seeds, TF-IDF settings
    data.py               # Ingestion, cleaning, and train/val/test splits
    preprocess.py         # Text preprocessing and normalization
    features.py           # TF-IDF and stylometric feature extraction
    train.py              # Model training and calibration
    evaluate.py           # Evaluation metrics and report generation
    explain.py            # Explainability and rationale generation
    reliability.py        # Reliability scoring (0-100)
    predict.py            # End-to-end inference pipeline
  models/                 # Saved .joblib model artifacts + metrics.json
  api/
    main.py               # FastAPI application entry point
    schemas.py            # Pydantic request and response schemas
  extension/
    manifest.json         # Chrome Manifest V3 configuration
    background.js         # Service worker
    content.js            # Content script for extracting reviews
    popup.html            # Extension popup user interface
    popup.js              # Extension UI interaction script
    popup.css             # Extension styling
  scripts/
    setup_nltk.py         # NLTK resource downloader helper
  tests/
    test_smoke.py         # Import and configuration smoke tests
  Makefile                # Project automation targets
  requirements.txt        # Pinned Python dependencies
  README.md               # Setup and development guide
```

---

## Setup Instructions

### Prerequisites
- Python 3.11 or higher
- Google Chrome browser (for the browser extension)

---

### Windows Setup (PowerShell)

1. **Clone the repository:**
   ```powershell
   git clone <repo-url> ReviewLens
   cd ReviewLens
   ```

2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Download NLTK resources:**
   ```powershell
   python -m ml.config
   # or
   python scripts/setup_nltk.py
   ```

5. **Run smoke tests:**
   ```powershell
   pytest
   ```

6. **Start the FastAPI backend server:**
   ```powershell
   uvicorn api.main:app --reload
   ```
   The API will be accessible at `http://localhost:8000`. Test the health check at `http://localhost:8000/health`.

---

### Linux / macOS Setup (Bash)

1. **Clone the repository:**
   ```bash
   git clone <repo-url> ReviewLens
   cd ReviewLens
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies & NLTK resources:**
   ```bash
   make install
   # or manually:
   pip install -r requirements.txt
   python -m ml.config
   ```

4. **Run tests:**
   ```bash
   make test
   # or:
   pytest
   ```

5. **Start the FastAPI backend server:**
   ```bash
   make serve
   # or:
   uvicorn api.main:app --reload
   ```

---

## Loading the Chrome Extension

1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Enable **Developer mode** toggle in the top-right corner.
3. Click **Load unpacked** and select the `extension/` directory of this repository.
4. The ReviewLens extension icon will appear in your browser toolbar.

---

## Makefile Targets

| Target | Description |
|---|---|
| `make install` | Installs pinned requirements and downloads NLTK datasets |
| `make test` | Executes the pytest test suite |
| `make train` | Runs the model training pipeline |
| `make evaluate` | Evaluates model performance on held-out test data |
| `make serve` | Starts the FastAPI backend server with live reload |

