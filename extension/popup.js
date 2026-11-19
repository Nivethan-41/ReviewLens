/**
 * ReviewLens Popup Script
 * Coordinates 4 states: idle, loading, error, and results.
 * Implements session caching, XSS-safe textContent rendering,
 * confidence-based sorting, filter chips, and explainability toggles.
 */

document.addEventListener("DOMContentLoaded", () => {
  // State Views
  const views = {
    idle: document.getElementById("state-idle"),
    loading: document.getElementById("state-loading"),
    error: document.getElementById("state-error"),
    results: document.getElementById("state-results"),
  };

  // Buttons & Controls
  const analyzeBtn = document.getElementById("analyze-btn");
  const retryBtn = document.getElementById("retry-btn");
  const reanalyzeBtn = document.getElementById("reanalyze-btn");
  const loadingText = document.getElementById("loading-text");
  const errorTitle = document.getElementById("error-title");
  const errorMessage = document.getElementById("error-message");
  const errorHelp = document.getElementById("error-help");

  // Summary Elements
  const reliabilityScore = document.getElementById("reliability-score");
  const reliabilityLabel = document.getElementById("reliability-label");
  const gaugeFill = document.getElementById("gauge-fill");
  const statTotal = document.getElementById("stat-total");
  const statSuspicious = document.getElementById("stat-suspicious");
  const statSuspiciousPct = document.getElementById("stat-suspicious-pct");
  const insightsContainer = document.getElementById("insights-container");
  const insightsList = document.getElementById("insights-list");

  // Reviews Elements
  const reviewsHeaderCount = document.getElementById("reviews-header-count");
  const reviewsList = document.getElementById("reviews-list");
  const reviewsEmpty = document.getElementById("reviews-empty");
  const filterChips = document.querySelectorAll(".filter-chips .chip");
  const footerVersion = document.getElementById("footer-version");

  // State Variables
  let currentUrl = "";
  let currentAnalysisData = null;
  let currentReviews = [];
  let currentFilter = "all";

  /**
   * Transition the UI to one of the 4 defined states.
   * @param {"idle"|"loading"|"error"|"results"} stateName
   */
  function showState(stateName) {
    Object.keys(views).forEach((key) => {
      if (views[key]) {
        if (key === stateName) {
          views[key].classList.remove("hidden");
        } else {
          views[key].classList.add("hidden");
        }
      }
    });
  }

  /**
   * Set loading text message.
   * @param {string} msg
   */
  function setLoadingMessage(msg) {
    if (loadingText) {
      loadingText.textContent = msg;
    }
  }

  /**
   * Display error state with custom message and offline help if needed.
   * @param {string} title
   * @param {string} message
   * @param {boolean} isServerOffline
   */
  function showError(title, message, isServerOffline = false) {
    if (errorTitle) errorTitle.textContent = title;
    if (errorMessage) errorMessage.textContent = message;

    if (errorHelp) {
      if (isServerOffline || (message && message.includes("make serve"))) {
        errorHelp.classList.remove("hidden");
      } else {
        errorHelp.classList.add("hidden");
      }
    }

    showState("error");
  }

  /**
   * Retrieve active browser tab.
   * @returns {Promise<chrome.tabs.Tab|null>}
   */
  async function getActiveTab() {
    if (typeof chrome !== "undefined" && chrome.tabs && chrome.tabs.query) {
      const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
      return tabs && tabs[0] ? tabs[0] : null;
    }
    return null;
  }

  /**
   * Fetch health info from /health and update model version in footer.
   */
  async function loadHealthInfo() {
    const apiBase =
      typeof CONFIG !== "undefined" && CONFIG.API_BASE_URL
        ? CONFIG.API_BASE_URL
        : "http://localhost:8000";

    try {
      const res = await fetch(`${apiBase}/health`);
      if (res.ok) {
        const data = await res.json();
        if (data && data.model_version && footerVersion) {
          footerVersion.textContent = `Model v${data.model_version}`;
          return;
        }
      }
    } catch (_) {
      // Direct fetch may fail if offline or blocked; try background service worker
    }

    if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.sendMessage) {
      chrome.runtime.sendMessage({ type: "HEALTH" }, (response) => {
        if (response && response.success && response.data && response.data.model_version && footerVersion) {
          footerVersion.textContent = `Model v${response.data.model_version}`;
        }
      });
    }
  }

  /**
   * Generate storage cache key for current URL.
   * @param {string} url
   * @returns {string}
   */
  function getCacheKey(url) {
    return `rl_cache_${url}`;
  }

  function getReviewsKey(url) {
    return `rl_reviews_${url}`;
  }

  /**
   * Initialize popup, check session cache for active tab URL.
   */
  async function init() {
    loadHealthInfo();

    const tab = await getActiveTab();
    if (!tab || !tab.url) {
      showState("idle");
      return;
    }

    currentUrl = tab.url;

    // Check URL restrictions
    if (
      currentUrl.startsWith("chrome://") ||
      currentUrl.startsWith("chrome-extension://") ||
      currentUrl.startsWith("edge://") ||
      currentUrl.startsWith("about:") ||
      currentUrl.startsWith("view-source:")
    ) {
      showError(
        "Restricted Page",
        "Cannot analyze browser internal pages. Please open a product or test page.",
        false
      );
      if (retryBtn) retryBtn.disabled = true;
      return;
    }

    // Check session storage cache
    if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.session) {
      const cKey = getCacheKey(currentUrl);
      const rKey = getReviewsKey(currentUrl);

      chrome.storage.session.get([cKey, rKey], (items) => {
        if (items && items[cKey]) {
          currentAnalysisData = items[cKey];
          currentReviews = items[rKey] || [];
          renderResults(currentAnalysisData, currentReviews);
          showState("results");
        } else {
          showState("idle");
        }
      });
    } else {
      showState("idle");
    }
  }

  /**
   * Execute full review extraction and analysis flow.
   */
  async function startAnalysis() {
    if (retryBtn) retryBtn.disabled = false;

    showState("loading");
    setLoadingMessage("Extracting reviews from page...");

    if (typeof chrome === "undefined" || !chrome.runtime || !chrome.runtime.sendMessage) {
      showError("Extension Error", "Chrome runtime messaging is unavailable.");
      return;
    }

    // 1. Extract reviews from active tab via background script
    chrome.runtime.sendMessage({ type: "EXTRACT" }, (extractRes) => {
      if (chrome.runtime.lastError) {
        showError(
          "Communication Error",
          chrome.runtime.lastError.message || "Failed to communicate with extension background worker."
        );
        return;
      }

      if (!extractRes) {
        showError("No Response", "No response received from extension background service worker.");
        return;
      }

      if (!extractRes.success) {
        if (extractRes.error === "RESTRICTED_PAGE") {
          showError("Restricted Page", extractRes.message || "Cannot analyze browser internal pages.");
        } else if (extractRes.error === "NO_REVIEWS_FOUND") {
          showError("No Reviews Found", "No product reviews were detected on this page.");
        } else {
          showError("Extraction Failed", extractRes.message || "Could not extract reviews from page.");
        }
        return;
      }

      const reviews = extractRes.reviews || [];
      if (reviews.length === 0) {
        showError("No Reviews Found", "No product reviews were detected on this page.");
        return;
      }

      currentReviews = reviews;
      currentUrl = extractRes.url || currentUrl;

      // 2. Update loading state with review count
      setLoadingMessage(`Analyzing ${reviews.length} reviews...`);

      // 3. Request analysis from FastAPI backend
      chrome.runtime.sendMessage(
        { type: "ANALYZE", reviews: reviews, url: currentUrl },
        (analyzeRes) => {
          if (chrome.runtime.lastError) {
            showError(
              "Communication Error",
              chrome.runtime.lastError.message || "Failed to communicate with backend."
            );
            return;
          }

          if (!analyzeRes) {
            showError("No Response", "Received empty response from extension worker.");
            return;
          }

          if (!analyzeRes.success) {
            if (analyzeRes.error === "SERVER_OFFLINE") {
              showError(
                "Server Offline",
                "Start the ReviewLens server: make serve",
                true
              );
            } else if (analyzeRes.error === "NO_REVIEWS_FOUND") {
              showError("No Reviews Found", "No product reviews were detected on this page.");
            } else {
              showError("Analysis Failed", analyzeRes.message || "An error occurred during analysis.");
            }
            return;
          }

          // 4. Analysis succeeded: Cache result in chrome.storage.session
          currentAnalysisData = analyzeRes.data;

          if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.session && currentUrl) {
            const saveObj = {};
            saveObj[getCacheKey(currentUrl)] = currentAnalysisData;
            saveObj[getReviewsKey(currentUrl)] = currentReviews;
            chrome.storage.session.set(saveObj);
          }

          // 5. Render results view
          renderResults(currentAnalysisData, currentReviews);
          showState("results");
        }
      );
    });
  }

  /**
   * Render analysis summary and review cards safely using textContent.
   * @param {object} data - API response payload matching PRD contract.
   * @param {Array<object>} rawReviews - Raw extracted reviews with text.
   */
  function renderResults(data, rawReviews) {
    if (!data || !data.summary) {
      showError("Invalid Data", "Analysis payload missing summary data.");
      return;
    }

    const { summary, results } = data;
    const score = Math.round(summary.reliability_score ?? 0);
    const label = summary.reliability_label || "Unknown";

    // Build ID-to-Text map for safe text rendering
    const reviewMap = new Map();
    if (Array.isArray(rawReviews)) {
      rawReviews.forEach((rev, idx) => {
        if (rev && rev.id) reviewMap.set(rev.id, rev.text);
        reviewMap.set(`idx_${idx}`, rev ? rev.text : "");
      });
    }

    // 1. Reliability Score
    if (reliabilityScore) reliabilityScore.textContent = String(score);

    // 2. Reliability Label Badge
    if (reliabilityLabel) {
      reliabilityLabel.textContent = label;
      reliabilityLabel.className = "badge";
      if (score >= 70) {
        reliabilityLabel.classList.add("badge-high");
      } else if (score >= 40) {
        reliabilityLabel.classList.add("badge-moderate");
      } else {
        reliabilityLabel.classList.add("badge-low");
      }
    }

    // 3. Colored Progress Gauge Bar
    if (gaugeFill) {
      const clampedScore = Math.max(0, Math.min(100, score));
      gaugeFill.style.width = `${clampedScore}%`;
      gaugeFill.className = "gauge-fill";
      if (score >= 70) {
        gaugeFill.classList.add("gauge-green");
      } else if (score >= 40) {
        gaugeFill.classList.add("gauge-amber");
      } else {
        gaugeFill.classList.add("gauge-red");
      }
    }

    // 4. Statistics
    if (statTotal) statTotal.textContent = String(summary.total || 0);
    if (statSuspicious) statSuspicious.textContent = String(summary.suspicious_count || 0);
    if (statSuspiciousPct) {
      statSuspiciousPct.textContent = `${Math.round(summary.suspicious_pct || 0)}%`;
    }

    // 5. Key Insights (up to 3)
    if (insightsContainer && insightsList) {
      insightsList.replaceChildren(); // Clear existing insights
      const insights = Array.isArray(summary.insights) ? summary.insights.slice(0, 3) : [];
      if (insights.length > 0) {
        insights.forEach((ins) => {
          const li = document.createElement("li");
          li.textContent = ins; // STRICTLY textContent
          insightsList.appendChild(li);
        });
        insightsContainer.classList.remove("hidden");
      } else {
        insightsContainer.classList.add("hidden");
      }
    }

    // 6. Reviews List
    renderReviewsList(results || [], reviewMap);
  }

  /**
   * Sort and filter reviews, then render list items safely via textContent.
   * @param {Array<object>} results - API results list.
   * @param {Map<string, string>} reviewMap - Map of review IDs to raw text.
   */
  function renderReviewsList(results, reviewMap) {
    if (!reviewsList) return;
    reviewsList.replaceChildren(); // Clear existing items

    // Sort suspicious first by confidence descending
    const sorted = [...results].sort((a, b) => {
      const isSuspA = a.label === "suspicious" ? 1 : 0;
      const isSuspB = b.label === "suspicious" ? 1 : 0;
      if (isSuspA !== isSuspB) {
        return isSuspB - isSuspA; // Suspicious (1) before others (0)
      }
      return (b.confidence || 0) - (a.confidence || 0);
    });

    // Apply Filter Chips: 'all' | 'suspicious' | 'genuine'
    const filtered = sorted.filter((r) => {
      if (currentFilter === "all") return true;
      if (currentFilter === "suspicious") return r.label === "suspicious";
      if (currentFilter === "genuine") return r.label === "genuine";
      return true;
    });

    // Update Header Count
    if (reviewsHeaderCount) {
      reviewsHeaderCount.textContent = `Reviews (${filtered.length} of ${sorted.length})`;
    }

    // Empty Filter State
    if (reviewsEmpty) {
      if (filtered.length === 0) {
        reviewsEmpty.textContent = `No ${currentFilter === "suspicious" ? "suspicious" : currentFilter === "genuine" ? "genuine-looking" : ""} reviews found.`;
        reviewsEmpty.classList.remove("hidden");
      } else {
        reviewsEmpty.classList.add("hidden");
      }
    }

    // Build Review Cards
    filtered.forEach((r, idx) => {
      const item = document.createElement("div");
      item.className = "review-item";

      const labelType = r.label || "uncertain";
      item.classList.add(`review-${labelType}`);

      // Header: Badge + Confidence
      const header = document.createElement("div");
      header.className = "review-item-header";

      const badge = document.createElement("span");
      badge.className = "badge";

      if (labelType === "suspicious") {
        badge.classList.add("badge-suspicious");
        badge.textContent = "Suspicious";
      } else if (labelType === "genuine") {
        badge.classList.add("badge-genuine");
        badge.textContent = "Genuine-looking";
      } else {
        badge.classList.add("badge-uncertain");
        badge.textContent = "Uncertain";
      }

      const confSpan = document.createElement("span");
      confSpan.className = "confidence-val";
      const confPct = Math.round((r.confidence || 0) * 100);
      confSpan.textContent = `${confPct}% confidence`;

      header.appendChild(badge);
      header.appendChild(confSpan);
      item.appendChild(header);

      // Snippet: First 120 characters rendered strictly via textContent
      const snippetEl = document.createElement("p");
      snippetEl.className = "review-snippet";

      const rawText = (reviewMap.get(r.id) || reviewMap.get(`idx_${idx}`) || r.text || "").trim();
      let snippet = rawText;
      if (snippet.length > 120) {
        snippet = `${snippet.substring(0, 120)}…`;
      }
      snippetEl.textContent = snippet || "(Review text not available)";
      item.appendChild(snippetEl);

      // 'Why?' Toggle Button
      const toggleBtn = document.createElement("button");
      toggleBtn.type = "button";
      toggleBtn.className = "why-toggle";
      toggleBtn.textContent = "Why? ▾";
      item.appendChild(toggleBtn);

      // Collapsible Reasons Box
      const reasonsBox = document.createElement("div");
      reasonsBox.className = "reasons-box hidden";

      if (Array.isArray(r.reasons) && r.reasons.length > 0) {
        const ul = document.createElement("ul");
        ul.className = "reasons-list";
        r.reasons.forEach((reason) => {
          const li = document.createElement("li");
          li.textContent = reason.text; // STRICTLY textContent
          ul.appendChild(li);
        });
        reasonsBox.appendChild(ul);
      } else {
        const emptyMsg = document.createElement("p");
        emptyMsg.className = "reasons-empty";
        emptyMsg.textContent = "No specific flags triggered.";
        reasonsBox.appendChild(emptyMsg);
      }
      item.appendChild(reasonsBox);

      // Toggle Event Listener
      toggleBtn.addEventListener("click", () => {
        const isCollapsed = reasonsBox.classList.contains("hidden");
        if (isCollapsed) {
          reasonsBox.classList.remove("hidden");
          toggleBtn.textContent = "Hide reasons ▴";
        } else {
          reasonsBox.classList.add("hidden");
          toggleBtn.textContent = "Why? ▾";
        }
      });

      reviewsList.appendChild(item);
    });
  }

  // Event Listeners
  if (analyzeBtn) {
    analyzeBtn.addEventListener("click", () => startAnalysis());
  }

  if (retryBtn) {
    retryBtn.addEventListener("click", () => startAnalysis());
  }

  if (reanalyzeBtn) {
    reanalyzeBtn.addEventListener("click", () => {
      // Clear session cache for current URL and re-run
      if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.session && currentUrl) {
        chrome.storage.session.remove([getCacheKey(currentUrl), getReviewsKey(currentUrl)], () => {
          startAnalysis();
        });
      } else {
        startAnalysis();
      }
    });
  }

  // Filter Chips Click Listeners
  filterChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      filterChips.forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      currentFilter = chip.getAttribute("data-filter") || "all";

      if (currentAnalysisData && currentAnalysisData.results) {
        const reviewMap = new Map();
        if (Array.isArray(currentReviews)) {
          currentReviews.forEach((rev, idx) => {
            if (rev && rev.id) reviewMap.set(rev.id, rev.text);
            reviewMap.set(`idx_${idx}`, rev ? rev.text : "");
          });
        }
        renderReviewsList(currentAnalysisData.results, reviewMap);
      }
    });
  });

  // Initialize
  init();
});
