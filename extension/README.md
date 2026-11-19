# ReviewLens Chrome Extension (Manifest V3)

ReviewLens is an AI-powered browser extension that detects potentially deceptive, computer-generated, and fake product reviews on e-commerce platforms.

---

## 1. Directory Structure

```
extension/
├── manifest.json       # Manifest V3 extension configuration
├── background.js       # Background service worker coordinating extraction & API calls
├── content.js          # On-demand content script extracting reviews from DOM
├── config.js           # Central configuration (API_BASE_URL, MAX_REVIEWS)
├── popup.html          # Extension popup UI
├── popup.js            # Popup controller logic & results rendering
├── popup.css           # Styling for popup and score gauge
├── generate_icons.py   # Helper script to create PNG icons
├── icons/              # Extension icons (16x16, 48x48, 128x128)
├── test/
│   └── sample_reviews.html  # Standalone test page with 12 fake reviews (itemprop markup)
└── README.md           # Instructions for installation, usage, and debugging
```

---

## 2. Installation (Load Unpacked)

1. **Open Google Chrome** (or any Chromium-based browser like Brave or Edge).
2. Navigate to the extensions page by entering `chrome://extensions` in the address bar.
3. In the top-right corner, toggle **Developer mode** to **ON**.
4. In the top-left toolbar, click **Load unpacked**.
5. In the file picker, select the `ReviewLens/extension` folder (the directory containing `manifest.json`).
6. ReviewLens will appear in your installed extensions list with its version (`0.1.0`) and icon.

---

## 3. Starting the Backend API

ReviewLens requires the local FastAPI inference backend running on `http://localhost:8000`.

From the project root directory:

```bash
# Ensure virtual environment is activated and dependencies are installed
pip install -r requirements.txt

# Run the FastAPI server
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Verify that the backend is healthy by visiting `http://localhost:8000/health` in your browser. You should receive:
```json
{"status": "ok", "model_version": "0.1.0"}
```

---

## 4. Testing with `sample_reviews.html`

To test review extraction and classification without needing an active internet connection or e-commerce shop:

1. Open the test page in Chrome:
   - Either double-click `extension/test/sample_reviews.html` to open it via `file://` protocol, or run a simple local web server:
     ```bash
     python -m http.server 8080
     ```
     and open `http://localhost:8080/extension/test/sample_reviews.html`.
2. Inspect the Service Worker console:
   - Go to `chrome://extensions`.
   - On the ReviewLens extension card, click the link **"service worker"** next to **Inspect views**.
   - This opens Chrome DevTools for `background.js`.
3. Return to the `sample_reviews.html` tab.
4. Click the **ReviewLens extension icon** in the Chrome toolbar to open the popup.
5. Click **"Analyze Reviews"**.
6. **Expected outcome:**
   - In the Service Worker console, you will see:
     ```
     [ReviewLens ServiceWorker] Extracted reviews (12): [ ... ]
     [ReviewLens ServiceWorker] Analysis succeeded: { results: [...], summary: {...} }
     ```
   - In the popup UI, the Product Reliability gauge will update showing the score, suspicious review count, percentage, insights, and individual review predictions.

---

## 5. Review Extraction Architecture

The content script (`content.js`) is injected **on demand** (only when the user clicks "Analyze Reviews") via `chrome.scripting.executeScript`. It uses a generic 3-tier fallback strategy:

1. **Schema.org / JSON-LD:**
   - Checks `<script type="application/ld+json">` for `Review` blocks or `Product.review`.
   - Checks microdata elements with `itemtype*="Review"` or `itemprop="review"`.
2. **Semantic Attributes:**
   - Detects `itemprop="reviewBody"` or elements whose `data-hook`, `class`, or `id` attributes match `review` and (`body`, `text`, or `content`) (e.g. Amazon's `data-hook="review-body"`).
   - Extracts ratings (`[itemprop="ratingValue"]`, aria-labels, star counts) and dates (`[itemprop="datePublished"]`, `<time datetime="...">`).
3. **Sibling Fallback:**
   - Finds repeating sibling cards containing a star rating pattern (`★`, `X out of 5`) and 40–3000 characters of text.
4. **Deduplication & Limits:**
   - Deduplicates identical texts, normalizes whitespace, and caps extraction at 100 reviews.
5. **Site-Specific Selectors:**
   - Defined in a centralized `SELECTORS` dictionary at the top of `content.js` with initial selectors for Amazon (`data-hook`) and Flipkart.

---

## 6. Error Handling

ReviewLens sends structured error codes back to the popup UI:

| Error Code | Cause | Resolution |
| :--- | :--- | :--- |
| `SERVER_OFFLINE` | FastAPI backend is not running at `http://localhost:8000` | Start backend with `uvicorn api.main:app --reload` |
| `NO_REVIEWS_FOUND` | No reviews matched any of the 3 extraction tiers | Verify the page contains customer reviews |
| `RESTRICTED_PAGE` | Attempted analysis on internal page (`chrome://`, `about:`) | Open a valid webpage or `sample_reviews.html` |
| `HTTP_ERROR` | Backend returned 4xx or 5xx (e.g. 503 if model missing) | Ensure `make train` has been executed |
| `INJECTION_FAILED` | Extension lacked permissions to inject script | Ensure activeTab permission is granted |
