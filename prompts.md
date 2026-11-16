# Part 2 — Three-Week Build Plan (9 Prompts)

## How to use these prompts

- Open a fresh AI-coding session at the start of each prompt. Paste the **Context Header** first, then the prompt.
- Run the tests and commit to Git after every prompt (`git commit -m "prompt 1.2 done"`). Do not move on until the "Done when" checks pass.
- If the tool produces something wrong, paste the error back and ask for a fix rather than rewriting the prompt.
- Keep the PRD sections 7, 9 and 10 in a file named `PRD.md` in the repo so the tool can always read them.

**Context Header (paste at the start of every session):**

```
Project: ReviewLens, an AI browser extension that detects potentially fake product reviews.
Stack: Python 3.11, pandas, scikit-learn, NLTK, joblib, FastAPI, pytest; Chrome Manifest V3 extension in vanilla JS.
Read PRD.md in the repo root for architecture, API contract and folder structure, and follow them exactly.
Rules: modular code, type hints, docstrings, no hard-coded paths (use ml/config.py), fixed random seed 42, write pytest tests for new logic, do not add features beyond the prompt.
```

## Plan 1 — Week 1: Foundation, Data and Features

**Week goal:** a clean repo, a cleaned dataset, and a tested preprocessing and feature pipeline.

### Prompt 1.1 — Project scaffold and environment (Days 1-2)

```
Create the ReviewLens repository skeleton exactly as in the "Repository Structure" section of PRD.md.

Do the following:
1. Create all folders and empty module files with a module docstring each (ml/, api/, extension/, tests/, data/raw, data/processed, models/, notebooks/).
2. Create requirements.txt pinning: pandas, numpy, scikit-learn, nltk, joblib, fastapi, uvicorn, pydantic, pytest, matplotlib, seaborn, jupyter, python-dateutil.
3. Create ml/config.py with pathlib-based constants: project root, data paths, model path, RANDOM_SEED=42, TEST_SIZE=0.1, VAL_SIZE=0.1, TFIDF settings (word ngram (1,2), char ngram (3,5), max_features), and a MODEL_VERSION string.
4. Create a Makefile with targets: install, test, train, evaluate, serve (uvicorn api.main:app --reload).
5. Create .gitignore (venv, __pycache__, data/raw, models/*.joblib, node_modules, .ipynb_checkpoints), a README.md with setup steps for Windows and Linux, and a tests/test_smoke.py that imports every ml module.
6. Add a download-NLTK-resources helper (stopwords, punkt, wordnet) in ml/config.py or a scripts/setup_nltk.py.

Do not implement any ML logic yet.
```

**Done when:** `pip install -r requirements.txt` succeeds, `pytest` passes the smoke test, and the folder tree matches the PRD.

### Prompt 1.2 — Dataset ingestion, cleaning and EDA (Days 3-4)

```
Implement the data layer in ml/data.py and an EDA notebook.

Context: I will place the dataset CSV(s) in data/raw/. The primary dataset is a labeled fake-review dataset (columns vary: review text, a label such as truthful/deceptive or OR/CG, optionally rating and category).

Do the following:
1. In ml/data.py write load_raw(path) with a configurable column-mapping dict at the top of the file (text_col, label_col, rating_col, category_col, label_map) so I can adapt it to any dataset by editing only the mapping. Normalize labels to 0 = genuine-looking, 1 = potentially deceptive.
2. Write clean_dataset(df): drop null or empty text, strip whitespace, drop exact duplicate texts (keep first), drop reviews shorter than 5 words, and return a report dict (rows before/after, duplicates removed, class balance).
3. Write split_dataset(df) that does a stratified train/val/test split using TEST_SIZE and VAL_SIZE from config and saves train.csv, val.csv, test.csv to data/processed/. Make sure duplicates cannot leak across splits.
4. Add a CLI entry: python -m ml.data --input data/raw/<file>.csv which runs load, clean, split and prints the report.
5. Create notebooks/01_eda.ipynb with: class balance plot, review-length distribution per class, top 20 words per class, average exclamation marks and capital-letter ratio per class, and 5 sample reviews per class. Add a short markdown conclusion cell I can edit.
6. Write tests in tests/test_data.py using a tiny in-memory DataFrame: label mapping, duplicate removal, short-review removal, split sizes and stratification.
```

**Done when:** the CLI produces three CSVs and a printed report, the notebook runs top to bottom, and tests pass.

### Prompt 1.3 — Preprocessing and feature extraction (Days 5-7)

```
Implement ml/preprocess.py and ml/features.py with tests.

1. preprocess.py: clean_text(text) (lowercase, normalize unicode, remove URLs, HTML tags and extra whitespace, keep sentence punctuation needed for stylometry), and normalize_for_tfidf(text) (tokenize, remove stopwords except negations like not/no/never, lemmatize). Keep both functions pure and fast. Raw text must remain available for stylometric features.
2. features.py: build a scikit-learn compatible pipeline:
   a. StylometricFeatures(BaseEstimator, TransformerMixin) that computes per review: word count, char count, avg word length, exclamation count and ratio, question mark ratio, uppercase-word ratio, first-person pronoun ratio (I, me, my, we), second-person ratio (you, your), superlative/intensifier count (best, amazing, perfect, worst, incredible, 100%), lexical diversity (unique words / words), repeated-word ratio, digit ratio, sentence count, and average sentence length. Return a DataFrame/array with stable, documented column names via get_feature_names_out().
   b. A ColumnTransformer / FeatureUnion combining word TF-IDF (1,2)-grams on normalized text, char TF-IDF (3,5)-grams on cleaned text, and the scaled stylometric features. Use settings from ml/config.py.
   c. build_feature_pipeline() returning the unfitted transformer, plus fit/save and load helpers using joblib to models/features.joblib.
3. Tests in tests/test_preprocess.py and tests/test_features.py: URL and HTML removal, negation kept, empty string handled without crashing, stylometric values correct on 3 hand-made reviews, output shape consistent between fit and transform, and no NaN in the output.
4. Add a short section to README.md explaining each feature and why it may signal deception.
```

**Done when:** the feature pipeline fits on train.csv and transforms val.csv without errors, and all tests pass.

**Week 1 checkpoint:** show your guide the EDA notebook and the feature list.

## Plan 2 — Week 2: Model, Explainability and Reliability

**Week goal:** a trained, evaluated and calibrated model that can explain its output, plus a reliability scorer, all behind one `predict` function.

### Prompt 2.1 — Model training and evaluation (Days 8-10)

```
Implement ml/train.py and ml/evaluate.py.

1. train.py: load train/val CSVs, build the feature pipeline from ml/features.py, and train and compare these classifiers inside full sklearn Pipelines: LogisticRegression (class_weight balanced, saga or liblinear), LinearSVC, RandomForest (n_jobs=-1) and HistGradientBoosting if the feature matrix is dense enough, otherwise skip it with a log message. Use 5-fold stratified cross-validation on train for macro-F1, then evaluate each on val.
2. Do a small, time-boxed hyperparameter search (GridSearchCV or RandomizedSearchCV) only for the best two models; keep it under about 10 minutes on a laptop and make the n_iter configurable.
3. Select the best model by validation macro-F1 (ties go to the simpler, more explainable model) and save: models/model.joblib (the full pipeline), models/metrics.json (all models' CV and val scores, chosen model name, MODEL_VERSION, training date, dataset row counts).
4. evaluate.py: evaluate the saved model on the untouched test set only once per run, and save accuracy, precision, recall, F1 per class, macro-F1, ROC-AUC, a confusion matrix PNG and a ROC curve PNG into models/reports/. Print a clean summary table.
5. Add an ablation: macro-F1 using (a) word TF-IDF only, (b) + char TF-IDF, (c) + stylometric features. Save it to metrics.json (useful for the project report).
6. Add make train and make evaluate to work end to end. Seeds fixed to 42. Tests: a fast test that trains on a 40-row synthetic dataset and checks that the artifacts are created and metrics.json has the expected keys.
```

**Done when:** `make train` and `make evaluate` run cleanly, and you have a table of model comparisons plus the confusion matrix. Aim for macro-F1 of 0.85 or higher; if you are far below, ask the tool to inspect data leakage and label mapping before tuning.

### Prompt 2.2 — Confidence calibration and reasons (Days 11-12)

```
Implement ml/explain.py and add calibration to the training flow.

1. Calibration: wrap the chosen classifier with CalibratedClassifierCV (sigmoid or isotonic, selected by val Brier score) fitted so that predict_proba is trustworthy. Save a calibration curve PNG to models/reports/. Define confidence = probability of the predicted class (0.5 to 1.0). Define an "uncertain" band where the probability of deceptive is between 0.45 and 0.55; those reviews get the label "uncertain" in the output.
2. explain.py: write explain_review(text, model_bundle, top_k=3) that returns a list of reasons: {"type": "language" | "style" | "pattern", "text": "...", "weight": float}.
   - language reasons: from the linear model's coefficients times the review's TF-IDF values, take the top contributing word n-grams toward the predicted class and turn them into a sentence such as "Contains phrases common in deceptive reviews: 'highly recommend', 'best ever'".
   - style reasons: compare the review's stylometric values with the genuine-class mean and std from training (z-score above 1.5), then use plain templates, for example "Unusually many exclamation marks", "Very short and generic", "Almost no specific details (numbers or first-person experience)".
   - Always return at least one reason; if nothing stands out, return "No strong deceptive signals found" for genuine-looking reviews.
   - If the chosen model is not linear, fall back to the LogisticRegression coefficients from a surrogate model trained on the same features and say so in a docstring.
3. Store the training-set stylometric means and stds in the saved bundle so explain_review needs no training data at runtime.
4. Tests: calibrated probabilities are within [0,1], the uncertain band works at the boundaries, reasons are non-empty for 10 sample reviews, and the reason text never contains raw feature names like 'tfidf__'.
```

**Done when:** a sample call prints a label, a confidence and 2-3 readable reasons for 10 different reviews, and the calibration plot looks sensible.

### Prompt 2.3 — Reliability scorer and unified predict module (Days 13-14)

```
Implement ml/reliability.py and ml/predict.py.

1. reliability.py: compute_reliability(reviews, results) where reviews is a list of dicts (id, text, rating optional, date optional) and results are the per-review predictions. Return {"reliability_score": 0-100, "reliability_label": "High" | "Moderate" | "Low", "insights": [str]} using this weighted formula (weights in config.py so they are easy to tune):
   - 45% share of non-suspicious reviews (confidence-weighted)
   - 20% near-duplicate penalty: pairwise cosine similarity on TF-IDF above 0.85 counts as near-duplicate; penalize by the share of reviews involved
   - 15% rating-distribution skew: penalize a J-shaped or all-5-star distribution when ratings exist, skip when they do not
   - 10% review-burst penalty: many reviews on the same day or within a very short window, skip when dates are missing
   - 10% average length/specificity of reviews
   Labels: 70+ High, 40-69 Moderate, below 40 Low.
   Insights are short sentences, for example "3 near-duplicate reviews found" or "68% of reviews are 5 stars, higher than typical". Handle fewer than 5 reviews by returning a label of "Not enough data".
2. predict.py: a ModelService class that loads models/model.joblib once, and exposes analyze(reviews: list[dict]) -> dict matching the response in the API contract of PRD.md (results + summary). Include input validation (empty text, very long text truncated to 5,000 characters, max 200 reviews per call), generate ids when missing, and keep order. It must be thread-safe for read-only use.
3. Add python -m ml.predict --demo that analyzes 10 hard-coded sample reviews and prints the JSON.
4. Tests: reliability on an obviously clean set versus a set of identical 5-star reviews (the second must score lower), the fewer-than-5 edge case, missing rating/date fields, and analyze() output schema.
```

**Done when:** `python -m ml.predict --demo` returns a full response JSON that matches the API contract, and the duplicate-heavy test set scores clearly lower.

**Week 2 checkpoint:** show metrics.json, the confusion matrix, and the demo output.

## Plan 3 — Week 3: API and Working Extension

**Week goal:** a FastAPI service and a basic Chrome extension that works end to end on a product page.

### Prompt 3.1 — FastAPI backend (Days 15-16)

```
Implement the REST API in api/schemas.py and api/main.py using ml/predict.py.

1. schemas.py: Pydantic v2 models: ReviewIn (id optional, text required 1-5000 chars, rating optional 1-5, date optional ISO string), AnalyzeRequest (page_url optional, reviews list 1-200), ReasonOut, ReviewResult (id, label, confidence, reasons), Summary, AnalyzeResponse. Match PRD.md section 9 exactly.
2. main.py: FastAPI app with a lifespan handler that loads ModelService once at startup. Endpoints: GET /health (status, model_version) and POST /analyze. Add CORS middleware that allows chrome-extension:// origins and http://localhost (make the allowed origins configurable in config.py). Return clear 422 messages for bad input and a 503 if the model file is missing, with a helpful message telling the user to run make train.
3. Add a simple request-timing log and a processing_ms field in the response.
4. Add make serve and document how to run the server and test it with curl in README.md, including a sample curl command with 3 reviews.
5. Tests in tests/test_api.py using FastAPI TestClient: health OK, valid analyze response schema, empty list rejected, text too long handled, missing optional fields accepted, order preserved, latency for 50 reviews under 2 seconds (mark as slow if needed).
```

**Done when:** `make serve` starts the API, the curl example returns a valid response, and `pytest` passes.

### Prompt 3.2 — Extension scaffold and generic review extractor (Days 17-19)

```
Create the Chrome Manifest V3 extension in extension/.

1. manifest.json (manifest_version 3): name ReviewLens, permissions: activeTab, scripting, storage; host_permissions for http://localhost:8000/* and https://*/* (for reading the active page); a background service worker (background.js); an action with popup.html; content script is injected on demand via chrome.scripting, not on every page.
2. content.js: export a function extractReviews() that returns [{id, text, rating, date}] from the current page. Implement a generic strategy that works without site-specific selectors, in this order:
   a. JSON-LD / schema.org Review and AggregateRating blocks if present;
   b. elements whose itemprop='reviewBody', or whose class/id/data-hook attributes contain 'review' and 'text/body/content' (for example Amazon's data-hook='review-body');
   c. fallback: repeated sibling blocks that contain a star rating pattern plus 40-3000 characters of text.
   Deduplicate by text, trim whitespace, cap at 100 reviews, and try to parse a rating (aria-label like '4.0 out of 5 stars', or a star count) and a date when available.
3. Put every site-specific selector in a single SELECTORS object at the top of content.js so that it can be extended later. Add selector entries for Amazon (data-hook) and Flipkart as a starting point and mark them TODO for refinement in the second half.
4. background.js: on a message {type:'ANALYZE'} from the popup, inject content.js into the active tab with chrome.scripting.executeScript, get the extracted reviews, POST them to http://localhost:8000/analyze, and return the response. Handle errors: no reviews found, server offline (fetch failure), HTTP errors, with clear error codes sent back to the popup.
5. Put the API base URL in a single constant in a small config.js. Provide icons as simple generated PNGs (16, 48, 128) with a script or plain colored placeholders.
6. Write extension/README.md with steps to load it unpacked at chrome://extensions with Developer mode on. Add a local test page extension/test/sample_reviews.html with 12 fake reviews (using itemprop markup) so extraction can be tested without a real shop.
```

**Done when:** the extension loads without manifest errors, and on `sample_reviews.html` the extraction returns the 12 reviews (check in the service worker console).

### Prompt 3.3 — Popup UI, end-to-end wiring and integration test (Days 20-21)

```
Build the popup and complete the end-to-end flow.

1. popup.html/popup.css/popup.js, clean and simple, about 380px wide, with states: idle (Analyze this page button), loading (spinner and 'Analyzing N reviews...'), results, and error (friendly message with a Retry button; for server offline, say 'Start the ReviewLens server: make serve').
2. Results view: a summary card with the reliability score (0-100) as a colored gauge or progress bar (green/amber/red), the reliability label, suspicious count and percent, and up to 3 insights; below it a scrollable list of reviews. Each item shows a short snippet (first 120 characters), a colored badge (Genuine-looking / Suspicious / Uncertain), the confidence as a percent, and a 'Why?' toggle that expands the reasons. Add filter chips: All, Suspicious, Genuine-looking. Sort suspicious first by confidence.
3. Use textContent (never innerHTML with review text) to avoid injection from page content.
4. Cache the last result per tab URL in chrome.storage.session so reopening the popup does not re-run analysis; add a Re-analyze button.
5. Add a small footer: 'Results are estimates, not proof.' and the model version from /health.
6. End-to-end verification: write docs/E2E_CHECKLIST.md with manual test steps and expected results for (a) the local sample page, (b) one real product page with reviews, (c) a page with no reviews, (d) server stopped. Also add tests/test_e2e_contract.py that posts the sample reviews from extension/test/ to the API and asserts the response satisfies everything the popup reads (field names, types, confidence range).
7. Update README.md with a complete setup: install, train, serve, load the extension, demo script for a 3-minute walkthrough.
```

**Done when:** clicking the extension on the sample page shows a reliability score and a flagged list with working 'Why?' details, errors are handled gracefully, and `pytest` passes fully.

**Week 3 checkpoint (end of first half):** record a 2-minute screen demo and tag the repo `v0.5-first-half`.

## Timeline at a Glance

| Week | Plan | Prompts | Output |
| --- | --- | --- | --- |
| 1 | Foundation, data and features | 1.1, 1.2, 1.3 | Repo, cleaned data, EDA, feature pipeline |
| 2 | Model, explainability, reliability | 2.1, 2.2, 2.3 | Trained and calibrated model, reasons, reliability scorer |
| 3 | API and working extension | 3.1, 3.2, 3.3 | FastAPI service, extension, end-to-end demo |

## Second Half Preview (not part of this plan)

Per-site extractors and in-page highlighting, UI polish, optional DistilBERT comparison, cross-dataset testing, full test report, documentation, project report and final demo.


## Addendum — Prompts for Product Genuineness, Media Signals & Indian E-Commerce Support

This section adds the internal features required for:
1. Product-level genuineness scoring (averaging review confidence with media boosts).
2. Review-volume risk detection (cold-start penalty for listings with few reviews).
3. Media trust weighting (prioritizing reviews containing photos/videos).
4. Seller-level confidence integration.
5. Indian e-commerce platform extractors (Amazon India, Flipkart, Meesho, Myntra).

---

### Supplementary Prompt A.1 — Media Weighting, Volume Penalty & Product Genuineness (Extends Prompts 2.2 & 2.3)

```
Enhance ml/reliability.py, ml/explain.py and ml/predict.py with product-level genuineness, media trust weighting, volume penalties, and seller confidence.

Context: ReviewLens needs to evaluate not just individual text, but the overall product investment risk. Reviews with photos and videos carry higher credibility, while products with very few reviews represent a high-risk purchase.

Do the following:
1. In ml/reliability.py, update compute_reliability(reviews, results, seller_info=None):
   a. Compute Media Trust Multiplier: Give reviews with has_images=True or has_video=True an authenticity boost (e.g., reduce deceptive probability by 25% or weight their confidence higher).
   b. Compute Product Genuineness Score (0-100): The weighted average of all review genuineness scores (1 - P(deceptive)), giving higher weight to media-verified reviews.
   c. Review-Volume Penalty: If len(reviews) < 5, mark volume_risk={"is_risky": True, "level": "High", "warning": "Very few reviews. High risk of fabricated ratings or insufficient track record."}. If 5 <= len(reviews) < 10, mark volume_risk={"is_risky": True, "level": "Moderate", "warning": "Limited review sample size."}.
   d. Seller Confidence Factor: When seller_info is provided (e.g. seller rating out of 5), compute a seller_confidence score (rating / 5.0 * 100) and flag any major dissonance between poor seller ratings and high review praise.
   e. Return product_genuineness_score, product_genuineness_label ("Likely Genuine", "Moderate / Mixed", "Potentially Deceptive / Risky"), media_review_count, media_review_pct, volume_risk, and seller_confidence in the summary dictionary.
2. In ml/explain.py:
   a. If has_images or has_video is True on a genuine review, append a positive reason: {"type": "media", "text": "Customer photo/video attached (higher credibility signal)"}.
3. In ml/predict.py:
   a. Update ModelService.analyze() to accept optional seller_info and pass it to compute_reliability.
   b. Prioritize media-rich reviews in results ordering when sorting for user inspection.
4. Write tests in tests/test_reliability.py:
   - Verify that adding media reviews increases the product genuineness score.
   - Verify that sets with < 5 reviews trigger the high-risk volume warning.
   - Verify seller confidence calculation.
```

**Done when:** Running tests demonstrates that listings with customer media receive higher trust scores, products with < 5 reviews are clearly flagged as risky, and the summary outputs all decision statistics.

---

### Supplementary Prompt A.2 — Indian E-Commerce Extractor with Media & Seller Selectors (Extends Prompt 3.2)

```
Extend extension/content.js to extract customer photos/videos, seller information, and platform indicators specifically for Indian e-commerce sites.

Do the following:
1. In extension/content.js, expand the SELECTORS dictionary with explicit support for Indian e-commerce platforms:
   a. Amazon India (amazon.in):
      - Reviews: [data-hook="review"]
      - Media indicators: .review-image-tile, [data-hook="review-image-tile"], video tags
      - Verified purchase: [data-hook="avp-badge"] ("Verified Purchase")
      - Seller card: #merchant-info, #sellerProfileTriggerId
   b. Flipkart (flipkart.com):
      - Reviews: div._27M-vq, div.col._2wzgFH
      - Media indicators: div._2tNq8e, img._3kidjx, video thumbnails
      - Verified purchase: ._2mcSum ("Certified Buyer")
      - Seller card: #sellerName, ._1RLviY
   c. Meesho / Generic Indian sites:
      - Extract review containers, image attachments (.review-image, img tags within review cards), and supplier name.
2. Update extractReviews() to return:
   - reviews: list of {id, text, rating, date, has_images, has_video, verified_purchase}
   - seller_info: {name: string, rating: float} (if detected on page)
3. Ensure graceful fallbacks: if seller info or media tags are absent, set default boolean/null values without breaking the extraction flow.
4. Add sample test fixtures in extension/test/sample_indian_ecom.html simulating Flipkart and Amazon.in review cards with photos, videos, and seller boxes.
```

**Done when:** Opening `sample_indian_ecom.html` or live Indian e-commerce product pages extracts review text, flags reviews that have pictures/videos (`has_images: true`), and extracts seller details into the payload.

---

### Supplementary Prompt A.3 — Product Genuineness Dashboard & Decision UI (Extends Prompt 3.3)

```
Update extension/popup.html, popup.css, and popup.js to display the Product Genuineness Dashboard, volume risk alerts, media statistics, and seller confidence.

Do the following:
1. In popup.html:
   a. Add a Top Product Genuineness Card:
      - Large circular or horizontal gauge showing overall Product Genuineness Score (0-100%).
      - Overall decision verdict label (e.g. "Likely Genuine", "Moderate Risk", "High Risk").
   b. Add a Volume Alert Banner:
      - Display a prominent warning banner when volume_risk.is_risky is true: "⚠️ Risky to Invest: Only N reviews found. Sample size too small for full confidence."
   c. Add Decision-Support Stats Grid:
      - Media Verified: "X% reviews with photos/videos"
      - Suspicious Rate: "X% flagged suspicious"
      - Seller Trust: "Seller Rating: X/5.0"
   d. In the review list:
      - Add a camera icon / media badge for reviews with photos or videos.
      - Add a "Verified Purchase" tag.
      - Highlight why media-backed reviews are more reliable.
2. In popup.js:
   - Render the new stats and warnings dynamically from response.summary.
   - Add filter chips: "All", "With Photos/Videos", "Suspicious", "Genuine".
   - Sort reviews so that suspicious ones and media-backed ones can be easily toggled.
3. In popup.css:
   - Style the risk banner (amber/red alert box), the score gauge, and media badges with clean, modern styles fitting Manifest V3 popup standards.
```

**Done when:** The extension popup renders the overall product genuineness score, displays the volume risk warning if reviews are few, shows media % statistics, and highlights seller confidence clearly to the shopper.


## Addendum — Prompts for Product Genuineness, Media Signals & Indian E-Commerce Support

This section adds the internal features required for:
1. Product-level genuineness scoring (averaging review confidence with media boosts).
2. Review-volume risk detection (cold-start penalty for listings with few reviews).
3. Media trust weighting (prioritizing reviews containing photos/videos).
4. Seller-level confidence integration.
5. Indian e-commerce platform extractors (Amazon India, Flipkart, Meesho, Myntra).

---

### Supplementary Prompt A.1 — Media Weighting, Volume Penalty & Product Genuineness (Extends Prompts 2.2 & 2.3)

```
Enhance ml/reliability.py, ml/explain.py and ml/predict.py with product-level genuineness, media trust weighting, volume penalties, and seller confidence.

Context: ReviewLens needs to evaluate not just individual text, but the overall product investment risk. Reviews with photos and videos carry higher credibility, while products with very few reviews represent a high-risk purchase.

Do the following:
1. In ml/reliability.py, update compute_reliability(reviews, results, seller_info=None):
   a. Compute Media Trust Multiplier: Give reviews with has_images=True or has_video=True an authenticity boost (e.g., reduce deceptive probability by 25% or weight their confidence higher).
   b. Compute Product Genuineness Score (0-100): The weighted average of all review genuineness scores (1 - P(deceptive)), giving higher weight to media-verified reviews.
   c. Review-Volume Penalty: If len(reviews) < 5, mark volume_risk={"is_risky": True, "level": "High", "warning": "Very few reviews. High risk of fabricated ratings or insufficient track record."}. If 5 <= len(reviews) < 10, mark volume_risk={"is_risky": True, "level": "Moderate", "warning": "Limited review sample size."}.
   d. Seller Confidence Factor: When seller_info is provided (e.g. seller rating out of 5), compute a seller_confidence score (rating / 5.0 * 100) and flag any major dissonance between poor seller ratings and high review praise.
   e. Return product_genuineness_score, product_genuineness_label ("Likely Genuine", "Moderate / Mixed", "Potentially Deceptive / Risky"), media_review_count, media_review_pct, volume_risk, and seller_confidence in the summary dictionary.
2. In ml/explain.py:
   a. If has_images or has_video is True on a genuine review, append a positive reason: {"type": "media", "text": "Customer photo/video attached (higher credibility signal)"}.
3. In ml/predict.py:
   a. Update ModelService.analyze() to accept optional seller_info and pass it to compute_reliability.
   b. Prioritize media-rich reviews in results ordering when sorting for user inspection.
4. Write tests in tests/test_reliability.py:
   - Verify that adding media reviews increases the product genuineness score.
   - Verify that sets with < 5 reviews trigger the high-risk volume warning.
   - Verify seller confidence calculation.
```

**Done when:** Running tests demonstrates that listings with customer media receive higher trust scores, products with < 5 reviews are clearly flagged as risky, and the summary outputs all decision statistics.

---

### Supplementary Prompt A.2 — Indian E-Commerce Extractor with Media & Seller Selectors (Extends Prompt 3.2)

```
Extend extension/content.js to extract customer photos/videos, seller information, and platform indicators specifically for Indian e-commerce sites.

Do the following:
1. In extension/content.js, expand the SELECTORS dictionary with explicit support for Indian e-commerce platforms:
   a. Amazon India (amazon.in):
      - Reviews: [data-hook="review"]
      - Media indicators: .review-image-tile, [data-hook="review-image-tile"], video tags
      - Verified purchase: [data-hook="avp-badge"] ("Verified Purchase")
      - Seller card: #merchant-info, #sellerProfileTriggerId
   b. Flipkart (flipkart.com):
      - Reviews: div._27M-vq, div.col._2wzgFH
      - Media indicators: div._2tNq8e, img._3kidjx, video thumbnails
      - Verified purchase: ._2mcSum ("Certified Buyer")
      - Seller card: #sellerName, ._1RLviY
   c. Meesho / Generic Indian sites:
      - Extract review containers, image attachments (.review-image, img tags within review cards), and supplier name.
2. Update extractReviews() to return:
   - reviews: list of {id, text, rating, date, has_images, has_video, verified_purchase}
   - seller_info: {name: string, rating: float} (if detected on page)
3. Ensure graceful fallbacks: if seller info or media tags are absent, set default boolean/null values without breaking the extraction flow.
4. Add sample test fixtures in extension/test/sample_indian_ecom.html simulating Flipkart and Amazon.in review cards with photos, videos, and seller boxes.
```

**Done when:** Opening `sample_indian_ecom.html` or live Indian e-commerce product pages extracts review text, flags reviews that have pictures/videos (`has_images: true`), and extracts seller details into the payload.

---

### Supplementary Prompt A.3 — Product Genuineness Dashboard & Decision UI (Extends Prompt 3.3)

```
Update extension/popup.html, popup.css, and popup.js to display the Product Genuineness Dashboard, volume risk alerts, media statistics, and seller confidence.

Do the following:
1. In popup.html:
   a. Add a Top Product Genuineness Card:
      - Large circular or horizontal gauge showing overall Product Genuineness Score (0-100%).
      - Overall decision verdict label (e.g. "Likely Genuine", "Moderate Risk", "High Risk").
   b. Add a Volume Alert Banner:
      - Display a prominent warning banner when volume_risk.is_risky is true: "⚠️ Risky to Invest: Only N reviews found. Sample size too small for full confidence."
   c. Add Decision-Support Stats Grid:
      - Media Verified: "X% reviews with photos/videos"
      - Suspicious Rate: "X% flagged suspicious"
      - Seller Trust: "Seller Rating: X/5.0"
   d. In the review list:
      - Add a camera icon / media badge for reviews with photos or videos.
      - Add a "Verified Purchase" tag.
      - Highlight why media-backed reviews are more reliable.
2. In popup.js:
   - Render the new stats and warnings dynamically from response.summary.
   - Add filter chips: "All", "With Photos/Videos", "Suspicious", "Genuine".
   - Sort reviews so that suspicious ones and media-backed ones can be easily toggled.
3. In popup.css:
   - Style the risk banner (amber/red alert box), the score gauge, and media badges with clean, modern styles fitting Manifest V3 popup standards.
```

**Done when:** The extension popup renders the overall product genuineness score, displays the volume risk warning if reviews are few, shows media % statistics, and highlights seller confidence clearly to the shopper.
