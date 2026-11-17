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

## Feature Engineering & Deception Indicators

ReviewLens combines word and character TF-IDF representations with 15 stylometric features designed to capture linguistic anomalies characteristic of deceptive, incentivized, or computer-generated product reviews:

### 1. TF-IDF Representations
- **Word TF-IDF ((1, 2)-grams on normalized text):** Extracts unigram and bigram token sequences from normalized text while preserving critical negations (`not`, `no`, `never`, `don't`). Deceptive reviews frequently exhibit distinct semantic signatures and formulaic sentiment patterns.
- **Character TF-IDF ((3, 5)-grams on cleaned text):** Captures sub-word morphology, character n-grams, and typographical habits, providing robustness against misspellings and subtle styling patterns.

### 2. Stylometric Features (15 Metrics)
- **`word_count` (Word Count):** Deceptive reviews frequently skew toward extremes—either abnormally brief generic blurbs or artificially bloated, repetitive praise.
- **`char_count` (Character Count):** Measures raw text length, complementing word count to establish structural density.
- **`avg_word_length` (Average Word Length):** Authentic reviews often discuss technical or operational details with domain-specific vocabulary (longer words), whereas fake reviews tend to use simplistic, generic descriptions.
- **`exclamation_count` (Exclamation Count):** High counts of `!` signal manufactured enthusiasm, agitation, or hyperbole.
- **`exclamation_ratio` (Exclamation Ratio):** Proportion of exclamation marks relative to total text characters, normalizing punctuation intensity across short and long reviews.
- **`question_mark_ratio` (Question Mark Ratio):** Rhetorical questions (e.g., *"Why wait to buy?"*) are common in promotional copy and sales pitches.
- **`uppercase_word_ratio` (Uppercase-Word Ratio):** Ratio of all-caps words (e.g., *"AMAZING"*, *"MUST BUY"*), measuring aggressive shouting and sentiment inflation.
- **`first_person_ratio` (First-Person Pronoun Ratio: I, me, my, we):** Genuine reviews often ground their claims in tangible personal narratives, whereas deceptive reviews often lack personal grounding or unnaturally overcompensate with repeated self-reference.
- **`second_person_ratio` (Second-Person Pronoun Ratio: you, your):** Promotional spam and fake reviews frequently target the reader directly to convince them to purchase.
- **`superlative_count` (Superlative & Intensifier Count):** Tracks extreme evaluative terms (`best`, `amazing`, `perfect`, `worst`, `incredible`, `100%`) that indicate unbalanced praise or ungrounded defamation.
- **`lexical_diversity` (Lexical Diversity / Type-Token Ratio):** Ratio of unique words to total words. Low diversity indicates templated writing or limited vocabulary.
- **`repeated_word_ratio` (Repeated-Word Ratio):** Proportion of tokens belonging to repeated words, detecting keyword stuffing and formulaic sentence structures.
- **`digit_ratio` (Digit Ratio):** Authentic reviews regularly cite concrete numerical specifics (e.g., battery life in hours, dimensions, days of use, model numbers); fake reviews rarely contain verifiable digits.
- **`sentence_count` (Sentence Count):** Structural metric capturing the number of sentences in the review.
- **`avg_sentence_length` (Average Sentence Length):** Syntactic complexity metric (words per sentence); fake reviews often rely on punchy promotional slogans or rambling run-ons.

---

## Makefile Targets

| Target | Description |
|---|---|
| `make install` | Installs pinned requirements and downloads NLTK datasets |
| `make test` | Executes the pytest test suite |
| `make train` | Runs the model training pipeline |
| `make evaluate` | Evaluates model performance on held-out test data |
| `make serve` | Starts the FastAPI backend server with live reload |

