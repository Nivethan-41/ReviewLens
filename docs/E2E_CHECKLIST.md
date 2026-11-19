# ReviewLens End-to-End Verification Checklist

This document details the manual verification procedures and expected results for validating the ReviewLens Manifest V3 Chrome extension and FastAPI ML backend.

---

## Pre-requisites

1. **Python Environment & Dependencies:**
   ```bash
   pip install -r requirements.txt
   python -m ml.config
   ```

2. **Model Artifacts:**
   Ensure `models/model.joblib` exists (or run `make train`).

3. **FastAPI Backend Server:**
   Start the service:
   ```bash
   make serve
   # or: uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
   ```
   Verify health check at `http://localhost:8000/health`:
   ```json
   {"status": "ok", "model_version": "0.1.0"}
   ```

4. **Extension Loaded in Chrome:**
   - Open Chrome and navigate to `chrome://extensions/`.
   - Enable **Developer mode** (toggle in top-right corner).
   - Click **Load unpacked** and select the `extension/` directory.
   - Confirm ReviewLens appears with icon and version `0.1.0`.

---

## Test Scenarios

### Scenario A: Local Sample Page (`sample_reviews.html`)

**Objective:** Verify full end-to-end review extraction, ML analysis, score calculation, explainability reasons, filtering, sorting, and session caching on 12 mock reviews.

1. **Steps:**
   - In Chrome, open `extension/test/sample_reviews.html` (e.g. via `file:///.../extension/test/sample_reviews.html` or a local web server).
   - Click the **ReviewLens** icon in the Chrome toolbar.
   - **Idle State:**
     - Observe popup width (~380px) and clean branding.
     - Observe the **"Analyze this page"** button.
   - Click **"Analyze this page"**.
   - **Loading State:**
     - Observe the animated spinner.
     - Verify text displays: `Analyzing 12 reviews...`.
   - **Results View:**
     - **Summary Card:**
       - Reliability Score: displayed as integer between 0 and 100.
       - Colored Progress Gauge: filled proportionally to the score (Green for $\ge 70$, Amber for $40-69$, Red for $< 40$).
       - Reliability Label Badge: displays categorical grade (e.g., `Moderate` or `High`).
       - Stats Grid: displays Total Reviews (`12`), Suspicious count, and Suspicious %.
       - Key Insights: lists up to 3 detected anomalies (e.g., near-duplicate reviews, rating distribution skew).
     - **Reviews List:**
       - Header displays `Reviews (12 of 12)`.
       - Reviews are sorted with **suspicious reviews first**, ordered by confidence descending.
       - Each review card displays:
         - Colored badge: `Suspicious` (red), `Genuine-looking` (green), or `Uncertain` (amber).
         - Confidence percent (e.g., `87% confidence`).
         - Short snippet: first 120 characters of review text (rendered safely with `textContent`).
         - **"Why?" Toggle:** click to expand explainability reasons; click **"Hide reasons"** to collapse.
     - **Filter Chips:**
       - Click **"Suspicious"**: only reviews flagged as suspicious appear.
       - Click **"Genuine-looking"**: only reviews flagged as genuine appear.
       - Click **"All"**: all 12 reviews appear.
     - **Session Caching:**
       - Click outside the popup to close it.
       - Click the ReviewLens icon again to reopen.
       - Verify the results view renders immediately without re-running analysis.
     - **Re-analyze:**
       - Click the **"Re-analyze"** button.
       - Verify cache is cleared, loading state appears with review count, and fresh results render.
     - **Footer:**
       - Verify footer displays: `Results are estimates, not proof. • Model v0.1.0`.

2. **Expected Result:**
   - All 12 reviews analyzed, UI displays clear reliability score and sorted review cards with expandable reasons, session cache persists across popup opens, and no injection occurs.

---

### Scenario B: Real E-Commerce Product Page with Reviews

**Objective:** Verify live review extraction and AI classification on an active e-commerce page (e.g., Amazon or Flipkart).

1. **Steps:**
   - Navigate to any product listing with customer reviews on Amazon or Flipkart (e.g., an electronics or headphone product page).
   - Scroll down so that review cards are rendered in the DOM.
   - Click the **ReviewLens** icon in the toolbar.
   - Click **"Analyze this page"**.
   - Watch the loading state display: `Analyzing N reviews...` where $N \ge 1$.
   - Inspect the results view.

2. **Expected Result:**
   - Generic review extractor extracts reviews via schema.org/Review or site selectors.
   - Backend evaluates reviews and returns reliability score, suspicious count, and reasons.
   - Review cards show first 120 characters safely with badges and explainability.

---

### Scenario C: Page with No Reviews

**Objective:** Verify graceful error handling when analyzing a page that contains zero product reviews.

1. **Steps:**
   - Open a non-product page in Chrome (e.g., `https://en.wikipedia.org/wiki/Machine_learning` or `https://www.google.com`).
   - Click the **ReviewLens** icon in the toolbar.
   - Click **"Analyze this page"**.
   - Wait for extractor execution.

2. **Expected Result:**
   - Popup transitions to the **Error State**.
   - Error Title: `No Reviews Found`.
   - Error Message: `No product reviews were detected on this page.`
   - A **"Retry"** button is displayed.
   - The UI does not crash and handles the empty state gracefully.

---

### Scenario D: Backend Server Stopped (Offline Handling)

**Objective:** Verify graceful error handling and recovery when the FastAPI backend is offline.

1. **Steps:**
   - Stop the backend server by pressing `Ctrl+C` in the terminal where `make serve` or `uvicorn` is running.
   - In Chrome, open `extension/test/sample_reviews.html`.
   - Click the **ReviewLens** icon.
   - Click **"Analyze this page"** (or click **"Re-analyze"** if previous results were cached).
   - Observe error UI.
   - Verify error details:
     - Error Title: `Server Offline`.
     - Error Message: `Start the ReviewLens server: make serve`.
     - Code block displays `make serve`.
     - **"Retry"** button is present.
   - Restart the backend server in your terminal:
     ```bash
     make serve
     ```
   - In the popup, click the **"Retry"** button.

2. **Expected Result:**
   - When offline, friendly error guidance instructs user to start the server via `make serve`.
   - Clicking **"Retry"** after restarting the server immediately recovers and renders the full results view.

---

## Verification Summary Table

| Test Case | Scenario | Expected State | Pass Criteria |
|---|---|---|---|
| TC-01 | Sample page analysis | Results State | Score (0-100), gauge bar, 12 reviews sorted suspicious first, working "Why?" toggles |
| TC-02 | Filter chips | Results State | "Suspicious" filters to suspicious items; "Genuine-looking" filters to genuine items |
| TC-03 | Session caching | Results State | Reopening popup instantly displays cached results for current tab URL |
| TC-04 | Re-analyze | Loading -> Results | Clears cache, re-extracts, and re-analyzes |
| TC-05 | Page with no reviews | Error State | Friendly error: "No product reviews were detected on this page." with Retry button |
| TC-06 | Server offline | Error State | Friendly error: "Start the ReviewLens server: make serve" with Retry button |
| TC-07 | Server recovery | Error -> Results | Clicking Retry recovers to Results state when server is restarted |
| TC-08 | Footer & Health | Footer | Displays "Results are estimates, not proof. • Model v0.1.0" |
