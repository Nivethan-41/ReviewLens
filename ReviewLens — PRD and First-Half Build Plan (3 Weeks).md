# ReviewLens — PRD and First-Half Build Plan (3 Weeks)

A Product Requirements Document distilled from your Review 1 deck, followed by 3 weekly plans (9 detailed prompts) for AI coding tools such as Claude Code, Cursor or Copilot.

# Part 1 — Product Requirements Document

## 1. Overview

ReviewLens is an AI-powered browser extension that analyzes online product reviews using Machine Learning and NLP. It labels each review as genuine-looking or potentially deceptive, shows a confidence score with short plain-language reasons, and gives a page-level reliability insight so users can make better purchasing decisions.

## 2. Problem Statement

Online shoppers rely on reviews, but fake and deceptive reviews are hard to spot by eye. Users need a tool that works where they shop, in real time, and explains why a review looks suspicious instead of just giving a verdict.

## 3. Goals and Non-Goals

**Goals**

- Train an ML classifier for fake/deceptive review detection using NLP and text-based features.
- Output a label, a calibrated confidence score and top reasons for every review.
- Compute a review reliability score for the whole set of reviews on a page.
- Deliver it as a browser extension that analyzes reviews on the page in real time.

**Non-Goals**

- No claim of 100% accuracy; results are "potentially deceptive", never a legal verdict.
- No user accounts, payments or cloud deployment at scale.
- No reviewer-network or graph analysis (future work).

## 4. Users and User Stories

- As an online shopper, I click the extension on a product page and see which reviews look suspicious.
- As a shopper, I see why a review was flagged (for example, excessive superlatives or very generic wording).
- As a shopper, I see one reliability score for the product's reviews so I can decide quickly.
- As a developer/evaluator (your faculty), I can run the model offline and see the metrics.

## 5. Functional Requirements

| ID | Requirement | Priority | Half |
| --- | --- | --- | --- |
| FR1 | Ingest and clean a labeled review dataset | Must | 1st |
| FR2 | Text preprocessing (cleaning, tokenizing, normalizing) | Must | 1st |
| FR3 | Feature extraction: TF-IDF n-grams plus stylometric/behavioral features | Must | 1st |
| FR4 | Train and evaluate ML classifiers; pick the best | Must | 1st |
| FR5 | Confidence score (calibrated probability) per review | Must | 1st |
| FR6 | Reasons: top 2-3 human-readable explanations per review | Must | 1st |
| FR7 | Reliability score and insights for a set of reviews | Must | 1st |
| FR8 | REST API: POST /analyze, GET /health | Must | 1st |
| FR9 | Extension: extract reviews from the page, call the API, show results | Must | 1st (basic), 2nd (polish) |
| FR10 | Per-site extractors (Amazon, Flipkart), in-page highlighting, UI polish | Should | 2nd |
| FR11 | Testing report, user guide, final demo, project report | Must | 2nd |

## 6. Non-Functional Requirements

- **Latency:** analyzing 50 reviews in under 2 seconds on a laptop.
- **Privacy:** only review text and optional rating/date are sent to the API; the API runs locally (localhost), nothing is stored.
- **Reproducibility:** one command to train, one to evaluate, fixed random seeds.
- **Maintainability:** modular code, type hints, docstrings, pytest tests.
- **Explainability:** every flag has at least one reason.

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

## 11. Success Metrics

- Macro-F1 of 0.85 or higher on the test split, with a documented confusion matrix.
- Every flagged review shows a confidence score and at least one reason.
- End-to-end demo: open a product page, click the extension, see results within 3 seconds.
- Test coverage of the core ML and API modules.

## 12. Risks

| Risk | Mitigation |
| --- | --- |
| Dataset is small or domain-specific (hotel vs product) | Test on a second dataset; state the limitation clearly |
| Shopping sites change their HTML | Generic extractor first; site-specific selectors isolated in one file |
| Model overconfident | Probability calibration and a conservative "uncertain" band (0.45-0.55) |
| Scope creep | Freeze the first-half scope below; the second half takes the Should items |

## 13. Scope Split

**First half (this plan, 3 weeks):** FR1-FR9 — data, features, trained and evaluated model, explanations, reliability scoring, working API, and a basic working extension wired end-to-end.

**Second half (later):** per-site extractors, statistics and datin-page highlighting and UI polish, optional transformer model, full testing, documentation, report and final demo, with the visualization of the scores of the products originality based on  the reviews, graphs, charts whatever else to finish the project completely.


## 14. Feature Addendum — Product Genuineness, Media Signals & Indian E-Commerce Focus

### 14.1 Product-Level Genuineness & Risk Assessment
- **Aggregated Product Genuinity:** Compute an overall product genuineness score (0–100) and confidence rating derived from the weighted aggregation of all individual reviews.
- **Review Volume & Cold-Start Penalty:** If a product listing has few reviews (e.g., < 5–10 reviews), it is flagged as potentially risky to invest in due to insufficient statistical confidence. A prominent risk warning ("High Risk / Insufficient Reviews") is generated.
- **Rich-Media Trust Multiplier (Photos & Videos):** Reviews that include customer-uploaded pictures or videos are prioritized and assigned higher authenticity weights, mitigating deceptive text signals as bot/fake review farms rarely provide verified customer media.
- **Seller-Level Confidence:** Factor in and display the overall confidence score of the product based on seller data (seller rating, tenure, fulfillment status) alongside review credibility.
- **Decision-Support Statistics:** Provide comprehensive, transparent metrics in the UI:
  - Overall Product Genuineness Index & Confidence Level.
  - Review sample size & volume risk indicator.
  - Percentage of media-backed reviews (photos/videos).
  - Suspicious vs. genuine review breakdown and duplicate counts.
  - Seller confidence rating and risk alerts.

### 14.2 Regional E-Commerce Focus (India)
- Primary target: Indian e-commerce ecosystem, specifically Amazon India (`amazon.in`), Flipkart, Meesho, Myntra, and Nykaa.
- Extractor rules tailored for Indian e-commerce DOM patterns: verified purchase tags ("Verified Purchase", "Flipkart Assured", "F-Assured"), customer media galleries, and seller profile cards.

### 14.3 Extended API Contract

`POST /analyze` (Updated Request):
```json
{
  "page_url": "https://www.amazon.in/dp/...",
  "seller_info": {
    "name": "RetailNet",
    "rating": 4.1
  },
  "reviews": [
    {
      "id": "r1",
      "text": "Excellent battery backup and screen quality. Attached pictures.",
      "rating": 5,
      "date": "2026-10-15",
      "has_images": true,
      "has_video": false,
      "verified_purchase": true
    }
  ]
}
```

Response (Updated Summary):
```json
{
  "results": [
    {
      "id": "r1",
      "label": "genuine",
      "confidence": 0.92,
      "has_media": true,
      "media_boost": true,
      "reasons": [
        {"type": "media", "text": "Verified customer photo attached (+trust)"},
        {"type": "language", "text": "Specific technical details mentioned"}
      ]
    }
  ],
  "summary": {
    "total": 45,
    "suspicious_count": 8,
    "suspicious_pct": 17.7,
    "media_review_count": 14,
    "media_review_pct": 31.1,
    "reliability_score": 82,
    "reliability_label": "High",
    "product_genuineness_score": 84.5,
    "product_genuineness_label": "Likely Genuine",
    "volume_risk": {
      "is_risky": false,
      "level": "Low",
      "warning": null
    },
    "seller_confidence": {
      "rating": 4.1,
      "label": "Trusted Seller",
      "score": 82
    },
    "insights": [
      "31% of reviews contain customer images/videos (strong positive signal)",
      "Low duplicate pattern detected"
    ]
  }
}
```

### 14.4 Functional Requirements Update

| ID | Requirement | Priority | Half |
| --- | --- | --- | --- |
| FR12 | Ingest and represent media presence (`has_images`, `has_video`) and seller info in data structures | Must | 1st |
| FR13 | Media-weighted review scoring (trust multiplier for reviews with media) | Must | 1st |
| FR14 | Review-volume penalty & cold-start risk detection (<10 reviews marked risky to invest) | Must | 1st |
| FR15 | Product-level genuineness aggregation & seller confidence score calculation | Must | 1st |
| FR16 | Indian e-commerce extractors (Amazon.in, Flipkart, Meesho) for reviews, media & seller cards | Must (Basic 1st, Refined 2nd) | 1st / 2nd |


## 14. Feature Addendum — Product Genuineness, Media Signals & Indian E-Commerce Focus

### 14.1 Product-Level Genuineness & Risk Assessment
- **Aggregated Product Genuinity:** Compute an overall product genuineness score (0–100) and confidence rating derived from the weighted aggregation of all individual reviews.
- **Review Volume & Cold-Start Penalty:** If a product listing has few reviews (e.g., < 5–10 reviews), it is flagged as potentially risky to invest in due to insufficient statistical confidence. A prominent risk warning ("High Risk / Insufficient Reviews") is generated.
- **Rich-Media Trust Multiplier (Photos & Videos):** Reviews that include customer-uploaded pictures or videos are prioritized and assigned higher authenticity weights, mitigating deceptive text signals as bot/fake review farms rarely provide verified customer media.
- **Seller-Level Confidence:** Factor in and display the overall confidence score of the product based on seller data (seller rating, tenure, fulfillment status) alongside review credibility.
- **Decision-Support Statistics:** Provide comprehensive, transparent metrics in the UI:
  - Overall Product Genuineness Index & Confidence Level.
  - Review sample size & volume risk indicator.
  - Percentage of media-backed reviews (photos/videos).
  - Suspicious vs. genuine review breakdown and duplicate counts.
  - Seller confidence rating and risk alerts.

### 14.2 Regional E-Commerce Focus (India)
- Primary target: Indian e-commerce ecosystem, specifically Amazon India (`amazon.in`), Flipkart, Meesho, Myntra, and Nykaa.
- Extractor rules tailored for Indian e-commerce DOM patterns: verified purchase tags ("Verified Purchase", "Flipkart Assured", "F-Assured"), customer media galleries, and seller profile cards.

### 14.3 Extended API Contract

`POST /analyze` (Updated Request):
```json
{
  "page_url": "https://www.amazon.in/dp/...",
  "seller_info": {
    "name": "RetailNet",
    "rating": 4.1
  },
  "reviews": [
    {
      "id": "r1",
      "text": "Excellent battery backup and screen quality. Attached pictures.",
      "rating": 5,
      "date": "2026-10-15",
      "has_images": true,
      "has_video": false,
      "verified_purchase": true
    }
  ]
}
```

Response (Updated Summary):
```json
{
  "results": [
    {
      "id": "r1",
      "label": "genuine",
      "confidence": 0.92,
      "has_media": true,
      "media_boost": true,
      "reasons": [
        {"type": "media", "text": "Verified customer photo attached (+trust)"},
        {"type": "language", "text": "Specific technical details mentioned"}
      ]
    }
  ],
  "summary": {
    "total": 45,
    "suspicious_count": 8,
    "suspicious_pct": 17.7,
    "media_review_count": 14,
    "media_review_pct": 31.1,
    "reliability_score": 82,
    "reliability_label": "High",
    "product_genuineness_score": 84.5,
    "product_genuineness_label": "Likely Genuine",
    "volume_risk": {
      "is_risky": false,
      "level": "Low",
      "warning": null
    },
    "seller_confidence": {
      "rating": 4.1,
      "label": "Trusted Seller",
      "score": 82
    },
    "insights": [
      "31% of reviews contain customer images/videos (strong positive signal)",
      "Low duplicate pattern detected"
    ]
  }
}
```

### 14.4 Functional Requirements Update

| ID | Requirement | Priority | Half |
| --- | --- | --- | --- |
| FR12 | Ingest and represent media presence (`has_images`, `has_video`) and seller info in data structures | Must | 1st |
| FR13 | Media-weighted review scoring (trust multiplier for reviews with media) | Must | 1st |
| FR14 | Review-volume penalty & cold-start risk detection (<10 reviews marked risky to invest) | Must | 1st |
| FR15 | Product-level genuineness aggregation & seller confidence score calculation | Must | 1st |
| FR16 | Indian e-commerce extractors (Amazon.in, Flipkart, Meesho) for reviews, media & seller cards | Must (Basic 1st, Refined 2nd) | 1st / 2nd |
