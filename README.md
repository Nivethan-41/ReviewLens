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
    test/
      sample_reviews.html # Local mock test page with 12 reviews
  scripts/
    setup_nltk.py         # NLTK resource downloader helper
  docs/
    E2E_CHECKLIST.md      # Manual end-to-end verification checklist
  tests/
    test_smoke.py         # Import and configuration smoke tests
    test_e2e_contract.py  # End-to-end contract validation tests
  Makefile                # Project automation targets
  requirements.txt        # Pinned Python dependencies
  README.md               # Setup and development guide
```

---

## Complete Setup Guide

Follow these steps to set up, train, serve, and load the extension end-to-end.

### 1. Prerequisites
- **Python 3.11+** installed
- **Google Chrome** browser (Manifest V3 compatible)
- **Git**

---

### 2. Installation (`install`)

#### Windows (PowerShell)
```powershell
# 1. Clone repository and navigate inside
git clone <repo-url> ReviewLens
cd ReviewLens

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install pinned Python dependencies
pip install -r requirements.txt

# 4. Download required NLTK resources
python scripts/setup_nltk.py
# or: python -m ml.config
```

#### Linux / macOS (Bash)
```bash
# 1. Clone repository and navigate inside
git clone <repo-url> ReviewLens
cd ReviewLens

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies & NLTK datasets using Makefile
make install
# or manually:
# pip install -r requirements.txt && python -m ml.config
```

---

### 3. Model Training (`train`)

Train the baseline classifiers, tune hyperparameters, calibrate confidence probabilities, and produce model artifacts in `models/`:

```bash
make train
# or manually via python:
python -m ml.train
```

To evaluate performance metrics and generate confusion matrix & ROC curves:
```bash
make evaluate
# or manually via python:
python -m ml.evaluate
```

To execute the automated test suite:
```bash
make test
# or:
pytest
```

---

### 4. Running the Backend Service (`serve`)

Start the FastAPI REST API server at `http://localhost:8000`:

```bash
make serve
# or manually via uvicorn:
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Verify service health:
```bash
curl http://localhost:8000/health
# Response: {"status": "ok", "model_version": "0.1.0"}
```

Interactive OpenAPI documentation is available in your browser at `http://localhost:8000/docs`.

---

### 5. Loading the Chrome Extension (`load`)

1. Open Google Chrome and navigate to:
   ```
   chrome://extensions/
   ```
2. Enable the **Developer mode** toggle in the top-right corner.
3. Click the **Load unpacked** button in the top-left toolbar.
4. Select the `extension/` directory from this repository.
5. ReviewLens will appear in your extensions list. Pin it to the Chrome toolbar for easy access.

---

## 3-Minute Demo Walkthrough Script

Use this structured script to demonstrate the full capabilities of ReviewLens to evaluators or stakeholders:

### Minute 0:00 – 0:45: Launching & Idle State
1. **Ensure Backend is Running:** Terminal displays `Uvicorn running on http://127.0.0.1:8000`.
2. **Open Test Sandbox:** In Chrome, open `extension/test/sample_reviews.html` (or drag and drop the file into Chrome).
   - *Explainer:* Point out the 12 mock customer reviews featuring varied lengths, exclamation marks, all-caps praise, and genuine technical feedback.
3. **Open Extension Popup:** Click the ReviewLens icon in the toolbar.
   - *Observe Idle State:* Clean 380px card showing the brand icon, title, short description, and the **"Analyze this page"** button.

### Minute 0:45 – 1:30: Live Extraction & Reliability Assessment
1. **Trigger Analysis:** Click **"Analyze this page"**.
2. **Observe Loading State:** Animated spinner appears with live counter: `Analyzing 12 reviews...`.
3. **Inspect Results View Summary Card:**
   - **Reliability Score:** Clear numerical score (0–100) with colored gauge bar (Green for high trust, Amber for moderate, Red for low).
   - **Categorical Grade:** Badge indicating `Moderate` or `High` reliability.
   - **Statistics:** Clear breakdown of Total Reviews (`12`), Suspicious Count, and Suspicious %.
   - **Key Insights:** Up to 3 AI-detected structural indicators (e.g., duplicated text phrases, rating skew).

### Minute 1:30 – 2:15: Deep Dive into Explainability & Filtering
1. **Sort & Review Cards:**
   - Point out that reviews are sorted **suspicious first by confidence**.
   - Show that each card displays a short snippet (first 120 characters), color-coded badge (`Suspicious` / `Genuine-looking` / `Uncertain`), and percentage confidence.
2. **Explainability ("Why?" Toggle):**
   - Click the **"Why?"** button on a flagged suspicious review.
   - Show the expanded list of rationales (e.g., *"Unusually high proportion of ALL-CAPS words"*, *"Unusually many exclamation marks"*).
   - Click **"Hide reasons"** to collapse.
3. **Filter Chips:**
   - Click **"Suspicious"** chip: immediately filters the list to only suspicious reviews.
   - Click **"Genuine-looking"** chip: filters to only genuine reviews.
   - Click **"All"** chip: restores all 12 reviews.

### Minute 2:15 – 3:00: Session Caching, Re-analysis & Error Resilience
1. **Session Caching:**
   - Close the popup by clicking anywhere outside.
   - Reopen the popup by clicking the ReviewLens icon.
   - *Observe:* The results view loads **instantly** from `chrome.storage.session` without re-extracting or re-calling the backend.
2. **Re-analyze:**
   - Click the **"Re-analyze"** button in the header.
   - Observe that the cache is invalidated and a fresh analysis cycle runs.
3. **Graceful Error Handling:**
   - In the backend terminal, press `Ctrl+C` to stop the server.
   - In the popup, click **"Re-analyze"**.
   - *Observe Error State:* Friendly alert message appears: `Start the ReviewLens server: make serve`.
   - Restart the server with `make serve` and click **"Retry"** in the popup.
   - *Observe Recovery:* ReviewLens reconnects seamlessly and renders results.
4. **Footer Check:**
   - Highlight the disclaimer: `Results are estimates, not proof. • Model v0.1.0`.

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
| `make test` | Executes the complete pytest test suite |
| `make train` | Runs the model training and calibration pipeline |
| `make evaluate` | Evaluates model performance on held-out test data |
| `make serve` | Starts the FastAPI backend server with live reload |
