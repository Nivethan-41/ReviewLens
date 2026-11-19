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
  if (!message || !message.type) return;

  if (message.type === "HEALTH") {
    handleHealthRequest()
      .then((res) => sendResponse(res))
      .catch((err) => {
        sendResponse({ success: false, error: err.message || "Health check failed." });
      });
    return true;
  }

  if (message.type === "EXTRACT") {
    extractReviewsFromActiveTab()
      .then((res) => sendResponse(res))
      .catch((err) => {
        console.error("[ReviewLens ServiceWorker] Extraction error:", err);
        sendResponse({
          success: false,
          error: "INTERNAL_ERROR",
          message: err.message || "Failed to extract reviews from active tab.",
        });
      });
    return true;
  }

  if (message.type === "ANALYZE") {
    handleAnalyzeRequest(message)
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
 * Perform health check against FastAPI backend.
 * @returns {Promise<object>} Status and health payload.
 */
async function handleHealthRequest() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (!res.ok) {
      return { success: false, status: res.status, error: `Health check failed with status ${res.status}` };
    }
    const data = await res.json();
    return { success: true, data };
  } catch (err) {
    return { success: false, error: err.message || "Server offline" };
  }
}

/**
 * Extract reviews from the active browser tab.
 * @returns {Promise<object>} Extracted reviews array and page URL.
 */
async function extractReviewsFromActiveTab() {
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
  let extraction = null;
  try {
    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => {
        if (typeof window.extractReviews === "function") {
          return window.extractReviews();
        }
        return { reviews: [], seller_info: null };
      },
    });
    extraction = (results && results[0] && results[0].result) || null;
  } catch (execErr) {
    console.error("[ReviewLens ServiceWorker] Extraction execution failed:", execErr);
    return {
      success: false,
      error: "EXTRACTION_FAILED",
      message: `Failed to extract reviews: ${execErr.message}`,
    };
  }

  const reviews = Array.isArray(extraction)
    ? extraction
    : (extraction && Array.isArray(extraction.reviews) ? extraction.reviews : []);
  const sellerInfo = extraction && extraction.seller_info ? extraction.seller_info : null;

  console.log(`[ReviewLens ServiceWorker] Extracted reviews (${reviews.length}):`, reviews);
  if (sellerInfo) {
    console.log("[ReviewLens ServiceWorker] Extracted seller info:", sellerInfo);
  }

  return {
    success: true,
    reviews: reviews,
    seller_info: sellerInfo,
    url: url,
    reviewsCount: reviews.length,
  };
}

/**
 * Coordinate review extraction and analysis via FastAPI backend.
 * @param {object} message - Incoming message payload (may contain pre-extracted reviews).
 * @returns {Promise<object>} Status and analysis result payload.
 */
async function handleAnalyzeRequest(message) {
  let reviews = message && Array.isArray(message.reviews) ? message.reviews : null;
  let sellerInfo = message && message.seller_info !== undefined ? message.seller_info : null;
  let pageUrl = message && message.url ? message.url : "";

  // If reviews were not passed directly, extract them now
  if (!reviews) {
    const extractResult = await extractReviewsFromActiveTab();
    if (!extractResult.success) {
      return extractResult;
    }
    reviews = extractResult.reviews;
    sellerInfo = extractResult.seller_info || null;
    pageUrl = extractResult.url;
  }

  // Handle empty reviews error condition
  if (!reviews || reviews.length === 0) {
    return {
      success: false,
      error: "NO_REVIEWS_FOUND",
      message: "No product reviews were detected on this page.",
      reviewsCount: 0,
    };
  }

  // Send payload to FastAPI backend POST /analyze
  const endpoint = `${API_BASE_URL}/analyze`;
  const payload = {
    page_url: pageUrl,
    reviews: reviews,
    seller_info: sellerInfo,
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
      message: "Start the ReviewLens server: make serve",
      reviewsCount: reviews.length,
    };
  }

  // Handle non-200 HTTP errors from backend
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

  // Return successful analysis payload to popup
  const data = await response.json();
  console.log("[ReviewLens ServiceWorker] Analysis succeeded:", data);
  return {
    success: true,
    reviewsCount: reviews.length,
    data: data,
    reviews: reviews,
  };
}
