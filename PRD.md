## 7. Architecture and Tech Stack

```
Product page --> Content script (extract reviews) --> Service worker --> POST /analyze
                                                                          |
                      Popup UI  <-- results + summary <-- FastAPI backend (localhost:8000)
                                                          |-- preprocessing
                                                          |-- feature extraction
                                                          |-- classifier + calibration
                                                          |-- explainer + reliability scorer
```

- **ML/Backend:** Python 3.11, pandas, scikit-learn, NLTK (or spaCy), joblib, FastAPI, Uvicorn, Pydantic, pytest.
- **Extension:** Chrome Manifest V3, vanilla JavaScript, HTML, CSS (no framework, to keep it simple).
- **Tools:** Git, Jupyter for EDA, VS Code.

## 8. Data and ML Approach

- **Datasets (verify availability and license before use):** Deceptive Opinion Spam Corpus (Ott et al., hotel reviews, 800 truthful and 800 deceptive), and a larger Amazon-style fake review dataset (for example the Kaggle "Fake Reviews Dataset" with original vs computer-generated labels). Pick one primary dataset; use the other only to test generalization.
- **Labels:** 0 = genuine-looking, 1 = potentially deceptive.
- **Features:** word and character TF-IDF n-grams; stylometric features (length, exclamation and caps ratio, first-person pronoun ratio, superlative count, lexical diversity, repeated-word ratio, digit ratio); optional rating-vs-sentiment mismatch if rating is available.
- **Models:** Logistic Regression (baseline and explainable), Linear SVM, Random Forest or Gradient Boosting for comparison. A transformer (DistilBERT) is a second-half stretch only.
- **Evaluation:** stratified 80/10/10 split; accuracy, precision, recall, F1, ROC-AUC, confusion matrix; target macro-F1 of 0.85 or higher on the held-out test set (report honestly if lower, and explain the dataset limits).
- **Reliability score (0-100):** combines the share of suspicious reviews, average confidence, near-duplicate review rate, rating distribution skew, and review-date bursts when dates exist.

## 9. API Contract

`POST /analyze`

```json
{
  "page_url": "https://...",
  "reviews": [{"id": "r1", "text": "...", "rating": 5, "date": "2026-10-01"}]
}
```

Response:

```json
{
  "results": [{
    "id": "r1",
    "label": "suspicious",
    "confidence": 0.87,
    "reasons": [{"type": "language", "text": "Uses many superlatives and generic praise"}]
  }],
  "summary": {
    "total": 50, "suspicious_count": 12, "suspicious_pct": 24.0,
    "reliability_score": 68, "reliability_label": "Moderate",
    "insights": ["3 near-duplicate reviews found"]
  }
}
```

`GET /health` returns `{"status": "ok", "model_version": "..."}`. `rating`, `date` and `id` are optional (the server generates ids).

## 10. Repository Structure

```
reviewlens/
  data/raw/  data/processed/
  notebooks/01_eda.ipynb
  ml/ (config.py, data.py, preprocess.py, features.py, train.py, evaluate.py, explain.py, reliability.py, predict.py)
  models/                 # saved .joblib artifacts + metrics.json
  api/ (main.py, schemas.py)
  extension/ (manifest.json, background.js, content.js, popup.html, popup.js, popup.css)
  tests/
  README.md  requirements.txt  Makefile
```