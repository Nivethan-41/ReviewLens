/**
 * ReviewLens Extension Configuration
 * Centralized settings for backend API connection and extraction limits.
 */

const CONFIG = {
  // Base URL of the ReviewLens FastAPI backend
  API_BASE_URL: "http://localhost:8000",

  // Analysis endpoint path
  ANALYZE_ENDPOINT: "http://localhost:8000/analyze",

  // Maximum number of reviews to extract and submit in one batch
  MAX_REVIEWS: 100,
};

// Export for Node.js / testing environments if available
if (typeof module !== "undefined" && module.exports) {
  module.exports = CONFIG;
}
