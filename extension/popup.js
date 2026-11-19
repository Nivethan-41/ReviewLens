/**
 * ReviewLens Popup Script
 * Coordinates user actions, triggers review analysis via background service worker,
 * and renders results and error states in the popup UI.
 */

document.addEventListener("DOMContentLoaded", () => {
  const analyzeBtn = document.getElementById("analyze-btn");
  const statusContainer = document.getElementById("status-container");
  const resultsContainer = document.getElementById("results");

  if (!analyzeBtn) return;

  analyzeBtn.addEventListener("click", () => {
    // Set UI to loading state
    analyzeBtn.disabled = true;
    resultsContainer.innerHTML = "";
    showLoading("Extracting page reviews and analyzing with ReviewLens AI...");

    // Send ANALYZE message to background service worker
    chrome.runtime.sendMessage({ type: "ANALYZE" }, (response) => {
      analyzeBtn.disabled = false;

      // Handle runtime connection failure
      if (chrome.runtime.lastError) {
        showError(
          "EXTENSION_ERROR",
          chrome.runtime.lastError.message || "Failed to communicate with ReviewLens extension."
        );
        return;
      }

      if (!response) {
        showError("NO_RESPONSE", "Received empty response from extension worker.");
        return;
      }

      if (!response.success) {
        showError(response.error || "UNKNOWN_ERROR", response.message);
        return;
      }

      // Success: render response summary and review analysis
      hideStatus();
      renderResults(response.data);
    });
  });

  function showLoading(message) {
    statusContainer.className = "status-container status-loading";
    statusContainer.innerHTML = `
      <div class="spinner"></div>
      <span>${escapeHtml(message)}</span>
    `;
  }

  function showError(code, message) {
    statusContainer.className = "status-container status-error";
    statusContainer.innerHTML = `
      <strong>Error: ${escapeHtml(code)}</strong>
      <div>${escapeHtml(message || "An unexpected error occurred.")}</div>
    `;
  }

  function hideStatus() {
    statusContainer.className = "status-container hidden";
    statusContainer.innerHTML = "";
  }

  function renderResults(data) {
    if (!data || !data.summary) {
      resultsContainer.innerHTML = "<p>No analysis summary returned.</p>";
      return;
    }

    const { summary, results } = data;
    const score = Math.round(summary.reliability_score ?? 0);
    const label = summary.reliability_label || "Unknown";

    let badgeClass = "badge-moderate";
    if (score >= 75) badgeClass = "badge-high";
    else if (score < 50) badgeClass = "badge-low";

    const insightsHtml =
      summary.insights && summary.insights.length > 0
        ? `
        <div class="insights-box">
          <h4>Key Insights</h4>
          <ul>
            ${summary.insights.map((ins) => `<li>${escapeHtml(ins)}</li>`).join("")}
          </ul>
        </div>
      `
        : "";

    const reviewsHtml =
      results && results.length > 0
        ? `
        <div class="reviews-list">
          <h3>Sample Review Predictions (${results.length})</h3>
          ${results
            .slice(0, 10)
            .map((r) => {
              const isSuspicious = r.label === "suspicious";
              const confPct = Math.round((r.confidence || 0) * 100);
              const reasonsText =
                r.reasons && r.reasons.length > 0
                  ? r.reasons.map((rs) => escapeHtml(rs.text)).join("; ")
                  : "";

              return `
              <div class="review-item ${isSuspicious ? "suspicious" : "genuine"}">
                <div class="review-item-header">
                  <span>ID: ${escapeHtml(r.id)}</span>
                  <span class="confidence-tag ${isSuspicious ? "suspicious" : "genuine"}">
                    ${isSuspicious ? "Suspicious" : "Genuine"} (${confPct}%)
                  </span>
                </div>
                ${reasonsText ? `<div class="reasons-list">${reasonsText}</div>` : ""}
              </div>
            `;
            })
            .join("")}
        </div>
      `
        : "";

    resultsContainer.innerHTML = `
      <div class="score-card">
        <div class="score-header">
          <span class="score-title">Product Reliability</span>
          <span class="score-badge ${badgeClass}">${escapeHtml(label)}</span>
        </div>
        <div class="score-value">${score}<span> / 100</span></div>
        <div class="stats-grid">
          <div class="stat-item">
            <div class="stat-num">${summary.total || 0}</div>
            <div class="stat-label">Reviews</div>
          </div>
          <div class="stat-item">
            <div class="stat-num">${summary.suspicious_count || 0}</div>
            <div class="stat-label">Suspicious</div>
          </div>
          <div class="stat-item">
            <div class="stat-num">${Math.round(summary.suspicious_pct || 0)}%</div>
            <div class="stat-label">Suspicious %</div>
          </div>
        </div>
      </div>
      ${insightsHtml}
      ${reviewsHtml}
    `;
  }

  function escapeHtml(text) {
    if (!text) return "";
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
