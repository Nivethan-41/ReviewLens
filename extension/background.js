/**
 * ReviewLens Background Service Worker (Manifest V3)
 * Orchestrates content script injection, review extraction, and FastAPI communication.
 */

try {
  importScripts("config.js");
} catch (e) {
  console.warn("[ReviewLens ServiceWorker] Could not import config.js:", e);
}

const API_BASE_URL =
  typeof CONFIG !== "undefined" && CONFIG.API_BASE_URL
    ? CONFIG.API_BASE_URL
    : "http://localhost:8000";

console.log("[ReviewLens ServiceWorker] Initialized with API_BASE_URL:", API_BASE_URL);

/**
 * Handle incoming runtime messages from the popup or other extension components.
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message && message.type === "ANALYZE") {
    handleAnalyzeRequest()
      .then((response) => sendResponse(response))
      .catch((err) => {
        console.error("[ReviewLens ServiceWorker] Unexpected error:", err);
        sendResponse({
          success: false,
          error: "INTERNAL_ERROR",
          message: err.message || "An unexpected error occurred during analysis.",
        });
      });
    return true; // Keep asynchronous message channel open for sendResponse
  }
});

/**
 * Coordinate review extraction from the active tab and analyze via FastAPI backend.
 * @returns {Promise<object>} Status and analysis result payload.
 */
async function handleAnalyzeRequest() {
  // 1. Retrieve the current active tab
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab || !tab.id) {
    return {
      success: false,
      error: "NO_ACTIVE_TAB",
      message: "No active browser tab found to inspect.",
    };
  }

  // 2. Validate URL permissions (prevent attempting script injection into browser internal pages)
  const url = tab.url || "";
  if (
    url.startsWith("chrome://") ||
    url.startsWith("chrome-extension://") ||
    url.startsWith("edge://") ||
    url.startsWith("about:") ||
    url.startsWith("view-source:")
  ) {
    return {
      success: false,
      error: "RESTRICTED_PAGE",
      message: "Cannot analyze browser internal pages. Please open a product or test page.",
    };
  }

  // 3. Inject content.js on demand
  try {
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ["content.js"],
    });
  } catch (injectErr) {
    console.error("[ReviewLens ServiceWorker] Injection failed:", injectErr);
    return {
      success: false,
      error: "INJECTION_FAILED",
      message: `Failed to inject review extractor into active page: ${injectErr.message}`,
    };
  }

  // 4. Invoke extractReviews() in the active tab context
  let reviews = [];
  try {
    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => {
        if (typeof window.extractReviews === "function") {
          return window.extractReviews();
        }
        return [];
      },
    });
    reviews = (results && results[0] && results[0].result) || [];
  } catch (execErr) {
    console.error("[ReviewLens ServiceWorker] Extraction execution failed:", execErr);
    return {
      success: false,
      error: "EXTRACTION_FAILED",
      message: `Failed to extract reviews: ${execErr.message}`,
    };
  }

  // Log extracted reviews to service worker console (satisfies verification requirement)
  console.log(`[ReviewLens ServiceWorker] Extracted reviews (${reviews.length}):`, reviews);

  // 5. Handle empty reviews error condition
  if (!reviews || reviews.length === 0) {
    return {
      success: false,
      error: "NO_REVIEWS_FOUND",
      message: "No product reviews were detected on this page.",
      reviewsCount: 0,
    };
  }

  // 6. Send payload to FastAPI backend POST /analyze
  const endpoint = `${API_BASE_URL}/analyze`;
  const payload = {
    page_url: url,
    reviews: reviews,
  };

  let response;
  try {
    response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
  } catch (networkErr) {
    console.error("[ReviewLens ServiceWorker] Backend connection failed:", networkErr);
    return {
      success: false,
      error: "SERVER_OFFLINE",
      message: `Backend server is offline at ${API_BASE_URL}. Ensure FastAPI is running via 'make serve' or 'uvicorn api.main:app --reload'.`,
      reviewsCount: reviews.length,
    };
  }

  // 7. Handle non-200 HTTP errors from backend
  if (!response.ok) {
    let errorDetail = "";
    try {
      const errJson = await response.json();
      errorDetail = errJson.detail || JSON.stringify(errJson);
    } catch (_) {
      errorDetail = await response.text();
    }
    console.error(`[ReviewLens ServiceWorker] HTTP ${response.status}:`, errorDetail);
    return {
      success: false,
      error: "HTTP_ERROR",
      status: response.status,
      message: `Backend API error (${response.status}): ${errorDetail}`,
      reviewsCount: reviews.length,
    };
  }

  // 8. Return successful analysis payload to popup
  const data = await response.json();
  console.log("[ReviewLens ServiceWorker] Analysis succeeded:", data);
  return {
    success: true,
    reviewsCount: reviews.length,
    data: data,
  };
}
