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

  // Volume Risk Banner
  const volumeRiskBanner = document.getElementById("volume-risk-banner");
  const volumeRiskText = document.getElementById("volume-risk-text");

  // Top Product Genuineness Elements
  const genuinenessVerdict = document.getElementById("genuineness-verdict");
  const genuinenessVerdictBadge = document.getElementById("genuineness-verdict-badge");
  const genuinenessCircle = document.getElementById("genuineness-circle");
  const productGenuinenessScore = document.getElementById("product-genuineness-score");
  const verdictHeadline = document.getElementById("verdict-headline");
  const verdictDescription = document.getElementById("verdict-description");

  // Decision-Support Stats Grid Elements
  const statMediaPct = document.getElementById("stat-media-pct");
  const statMediaLabel = document.getElementById("stat-media-label");
  const statSuspiciousPct = document.getElementById("stat-suspicious-pct");
  const statSuspiciousLabel = document.getElementById("stat-suspicious-label");
  const statSellerRating = document.getElementById("stat-seller-rating");
  const statSellerLabel = document.getElementById("stat-seller-label");

  // Legacy Summary Elements (for backward compatibility)
  const reliabilityScore = document.getElementById("reliability-score");
  const reliabilityLabel = document.getElementById("reliability-label");
  const gaugeFill = document.getElementById("gauge-fill");
  const statTotal = document.getElementById("stat-total");
  const statSuspicious = document.getElementById("stat-suspicious");
  const insightsContainer = document.getElementById("insights-container");
  const insightsList = document.getElementById("insights-list");

  // Reviews Elements & Sorting Controls
  const reviewsHeaderCount = document.getElementById("reviews-header-count");
  const reviewsList = document.getElementById("reviews-list");
  const reviewsEmpty = document.getElementById("reviews-empty");
  const filterChips = document.querySelectorAll(".filter-chips .chip");
  const sortToggleBtn = document.getElementById("sort-toggle-btn");
  const sortToggleText = document.getElementById("sort-toggle-text");
  const footerVersion = document.getElementById("footer-version");

  // State Variables
  let currentUrl = "";
  let currentAnalysisData = null;
  let currentReviews = [];
  let currentFilter = "all";
  let currentSort = "suspicious"; // "suspicious" | "media"

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
      const sellerInfo = extractRes.seller_info || null;
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
        { type: "ANALYZE", reviews: reviews, seller_info: sellerInfo, url: currentUrl },
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
   * Render analysis summary, genuineness dashboard, volume alert, and review cards.
   * @param {object} data - API response payload matching PRD contract.
   * @param {Array<object>} rawReviews - Raw extracted reviews with media & verification metadata.
   */
  function renderResults(data, rawReviews) {
    if (!data || !data.summary) {
      showError("Invalid Data", "Analysis payload missing summary data.");
      return;
    }

    const { summary, results } = data;

    // Build ID-to-Review metadata map
    const reviewMap = new Map();
    if (Array.isArray(rawReviews)) {
      rawReviews.forEach((rev, idx) => {
        if (rev && rev.id) reviewMap.set(rev.id, rev);
        reviewMap.set(`idx_${idx}`, rev || {});
      });
    }

    // 1. Top Product Genuineness Score & Verdict (1a)
    const genuinenessScore =
      summary.product_genuineness_score != null
        ? Math.round(summary.product_genuineness_score)
        : Math.round(summary.reliability_score ?? 0);
    const clampedScore = Math.max(0, Math.min(100, genuinenessScore));

    let verdict = "Likely Genuine";
    let headlineText = "High Shopper Confidence";
    let descText = "Reviews show authentic linguistic traits, low spam indicators, and verified buyer media.";
    let themeClass = "likely";

    if (clampedScore >= 70) {
      verdict = "Likely Genuine";
      headlineText = "High Shopper Confidence";
      descText = "Reviews show authentic linguistic traits, low spam indicators, and verified buyer media.";
      themeClass = "likely";
    } else if (clampedScore >= 40) {
      verdict = "Moderate Risk";
      headlineText = "Moderate / Mixed Confidence";
      descText = "Mixed credibility signals detected. Review patterns show moderate inconsistencies or mild rating skew.";
      themeClass = "moderate";
    } else {
      verdict = "High Risk";
      headlineText = "High Deception Risk";
      descText = "Critical suspicious signals detected. Pattern anomalies, near-duplicates, or low seller trust indicate high risk.";
      themeClass = "risky";
    }

    // Override if backend returned categorical label
    if (summary.product_genuineness_label) {
      const lbl = summary.product_genuineness_label;
      if (lbl === "Likely Genuine") {
        verdict = "Likely Genuine";
      } else if (lbl === "Moderate / Mixed" || lbl === "Moderate Risk") {
        verdict = "Moderate Risk";
      } else if (lbl === "Potentially Deceptive / Risky" || lbl === "High Risk") {
        verdict = "High Risk";
      }
    }

    if (genuinenessVerdict) {
      genuinenessVerdict.textContent = verdict;
      genuinenessVerdict.className = `genuineness-verdict verdict-${themeClass}`;
    }

    if (genuinenessVerdictBadge) {
      genuinenessVerdictBadge.textContent = verdict;
      genuinenessVerdictBadge.className = `verdict-pill pill-${themeClass}`;
    }

    if (productGenuinenessScore) {
      productGenuinenessScore.textContent = String(clampedScore);
    }

    if (verdictHeadline) verdictHeadline.textContent = headlineText;
    if (verdictDescription) verdictDescription.textContent = descText;

    // Update circular SVG gauge
    if (genuinenessCircle) {
      const circumference = 301.59; // 2 * PI * 48
      const strokeOffset = circumference * (1 - clampedScore / 100);
      genuinenessCircle.style.strokeDasharray = `${circumference}`;
      genuinenessCircle.style.strokeDashoffset = `${strokeOffset}`;
      const strokeColorClass =
        themeClass === "likely" ? "stroke-green" : themeClass === "moderate" ? "stroke-amber" : "stroke-red";
      genuinenessCircle.className = `gauge-circle-fill ${strokeColorClass}`;
    }

    // Update horizontal gauge fill
    if (gaugeFill) {
      gaugeFill.style.width = `${clampedScore}%`;
      gaugeFill.className = "gauge-fill";
      if (themeClass === "likely") {
        gaugeFill.classList.add("gauge-green");
      } else if (themeClass === "moderate") {
        gaugeFill.classList.add("gauge-amber");
      } else {
        gaugeFill.classList.add("gauge-red");
      }
    }

    // 2. Volume Alert Banner (1b)
    const totalCount = summary.total ?? (Array.isArray(rawReviews) ? rawReviews.length : 0);
    const volumeRisk = summary.volume_risk;
    if (volumeRisk && volumeRisk.is_risky) {
      if (volumeRiskBanner) {
        volumeRiskBanner.classList.remove("hidden");
        if (volumeRisk.level === "High") {
          volumeRiskBanner.className = "volume-risk-banner volume-risk-high";
        } else {
          volumeRiskBanner.className = "volume-risk-banner volume-risk-moderate";
        }
      }
      if (volumeRiskText) {
        volumeRiskText.textContent = `⚠️ Risky to Invest: Only ${totalCount} reviews found. Sample size too small for full confidence.`;
      }
    } else {
      if (volumeRiskBanner) volumeRiskBanner.classList.add("hidden");
    }

    // 3. Decision-Support Stats Grid (1c)
    // Media Verified: "X% reviews with photos/videos"
    const mediaPct = Math.round(summary.media_review_pct ?? 0);
    if (statMediaPct) statMediaPct.textContent = `${mediaPct}%`;
    if (statMediaLabel) statMediaLabel.textContent = `${mediaPct}% reviews with photos/videos`;

    // Suspicious Rate: "X% flagged suspicious"
    const suspPct = Math.round(summary.suspicious_pct ?? 0);
    if (statSuspiciousPct) statSuspiciousPct.textContent = `${suspPct}%`;
    if (statSuspiciousLabel) statSuspiciousLabel.textContent = `${suspPct}% flagged suspicious`;

    // Seller Trust: "Seller Rating: X/5.0"
    if (summary.seller_confidence != null) {
      const sRating = (summary.seller_confidence / 20.0).toFixed(1);
      if (statSellerRating) statSellerRating.textContent = `${sRating}/5.0`;
      if (statSellerLabel) statSellerLabel.textContent = `Seller Rating: ${sRating}/5.0`;
    } else {
      if (statSellerRating) statSellerRating.textContent = "N/A";
      if (statSellerLabel) statSellerLabel.textContent = "Seller Rating: Not available";
    }

    // Legacy summary elements
    if (statTotal) statTotal.textContent = String(totalCount);
    if (statSuspicious) statSuspicious.textContent = String(summary.suspicious_count || 0);
    if (reliabilityScore) reliabilityScore.textContent = String(Math.round(summary.reliability_score ?? 0));
    if (reliabilityLabel) {
      reliabilityLabel.textContent = summary.reliability_label || "Moderate";
      reliabilityLabel.className = "badge";
      if (clampedScore >= 70) reliabilityLabel.classList.add("badge-high");
      else if (clampedScore >= 40) reliabilityLabel.classList.add("badge-moderate");
      else reliabilityLabel.classList.add("badge-low");
    }

    // 4. Key Insights Section (up to 3)
    if (insightsContainer && insightsList) {
      insightsList.replaceChildren();
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

    // 5. Render Reviews List
    renderReviewsList(results || [], reviewMap);
  }

  /**
   * Sort and filter reviews, then render list items safely via textContent.
   * @param {Array<object>} results - API results list.
   * @param {Map<string, object>} reviewMap - Map of review IDs to raw review metadata.
   */
  function renderReviewsList(results, reviewMap) {
    if (!reviewsList) return;
    reviewsList.replaceChildren();

    // Sort according to currentSort toggle: "suspicious" vs "media"
    const sorted = [...results].sort((a, b) => {
      const metaA = reviewMap.get(a.id) || {};
      const metaB = reviewMap.get(b.id) || {};
      const hasMediaA = Boolean(metaA.has_images || metaA.has_video);
      const hasMediaB = Boolean(metaB.has_images || metaB.has_video);

      if (currentSort === "media") {
        const mScoreA = hasMediaA ? 1 : 0;
        const mScoreB = hasMediaB ? 1 : 0;
        if (mScoreA !== mScoreB) {
          return mScoreB - mScoreA; // Media-backed reviews first
        }
        return (b.confidence || 0) - (a.confidence || 0);
      } else {
        // Suspicious first
        const isSuspA = a.label === "suspicious" ? 1 : 0;
        const isSuspB = b.label === "suspicious" ? 1 : 0;
        if (isSuspA !== isSuspB) {
          return isSuspB - isSuspA; // Suspicious (1) before others (0)
        }
        return (b.confidence || 0) - (a.confidence || 0);
      }
    });

    // Apply Filter Chips: 'all' | 'media' | 'suspicious' | 'genuine'
    const filtered = sorted.filter((r) => {
      const meta = reviewMap.get(r.id) || {};
      if (currentFilter === "all") return true;
      if (currentFilter === "media") return Boolean(meta.has_images || meta.has_video);
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
        let filterDesc = "";
        if (currentFilter === "suspicious") filterDesc = "suspicious ";
        else if (currentFilter === "genuine") filterDesc = "genuine ";
        else if (currentFilter === "media") filterDesc = "photo/video ";
        reviewsEmpty.textContent = `No ${filterDesc}reviews match the selected filter.`;
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

      const meta = reviewMap.get(r.id) || reviewMap.get(`idx_${idx}`) || {};
      const hasMedia = Boolean(meta.has_images || meta.has_video);
      const isVerified = Boolean(meta.verified_purchase);

      // Header: Badge Group + Confidence
      const header = document.createElement("div");
      header.className = "review-item-header";

      const badgeGroup = document.createElement("div");
      badgeGroup.className = "review-badge-group";

      // 1. Classification Badge
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
      badgeGroup.appendChild(badge);

      // 2. Camera icon / Media Badge (1d)
      if (hasMedia) {
        const mediaBadge = document.createElement("span");
        mediaBadge.className = "badge badge-media";

        const camSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        camSvg.setAttribute("class", "badge-inline-icon");
        camSvg.setAttribute("viewBox", "0 0 24 24");
        camSvg.setAttribute("fill", "none");
        camSvg.setAttribute("stroke", "currentColor");
        camSvg.setAttribute("stroke-width", "2");
        camSvg.setAttribute("stroke-linecap", "round");
        camSvg.setAttribute("stroke-linejoin", "round");
        const camPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
        camPath.setAttribute("d", "M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z");
        const camCircle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        camCircle.setAttribute("cx", "12");
        camCircle.setAttribute("cy", "13");
        camCircle.setAttribute("r", "4");
        camSvg.appendChild(camPath);
        camSvg.appendChild(camCircle);
        mediaBadge.appendChild(camSvg);

        const mediaLabelText = document.createTextNode(meta.has_video ? " Video" : " Photo");
        mediaBadge.appendChild(mediaLabelText);
        badgeGroup.appendChild(mediaBadge);
      }

      // 3. Verified Purchase Tag (1d)
      if (isVerified) {
        const verifiedBadge = document.createElement("span");
        verifiedBadge.className = "badge badge-verified";
        verifiedBadge.textContent = "✓ Verified Purchase";
        badgeGroup.appendChild(verifiedBadge);
      }

      header.appendChild(badgeGroup);

      const confSpan = document.createElement("span");
      confSpan.className = "confidence-val";
      const confPct = Math.round((r.confidence || 0) * 100);
      confSpan.textContent = `${confPct}% confidence`;
      header.appendChild(confSpan);

      item.appendChild(header);

      // 4. Highlight why media-backed reviews are more reliable (1d)
      if (hasMedia) {
        const mediaHighlight = document.createElement("div");
        mediaHighlight.className = "media-trust-highlight";

        const highlightIcon = document.createElement("span");
        highlightIcon.className = "highlight-icon";
        highlightIcon.textContent = "📷";

        const highlightBody = document.createElement("div");
        highlightBody.className = "highlight-body";

        const highlightBold = document.createElement("strong");
        highlightBold.textContent = "Media Verified: ";

        const highlightDesc = document.createTextNode(
          "Customer photo/video evidence physically validates receipt. Media-backed reviews receive higher authenticity weight in the reliability model."
        );

        highlightBody.appendChild(highlightBold);
        highlightBody.appendChild(highlightDesc);
        mediaHighlight.appendChild(highlightIcon);
        mediaHighlight.appendChild(highlightBody);

        item.appendChild(mediaHighlight);
      }

      // Snippet: First 120 characters rendered strictly via textContent
      const snippetEl = document.createElement("p");
      snippetEl.className = "review-snippet";

      const rawText = (meta.text || r.text || "").trim();
      let snippet = rawText
        .replace(/Brief content visible,?\s*double tap to read full content\.?/gi, " ")
        .replace(/Full content visible,?\s*double tap to read brief content\.?/gi, " ")
        .replace(/Double tap to read (?:full|brief) content\.?/gi, " ")
        .replace(/(?:^|\b|(?<=[a-z0-9.,!?]))(?:Read|Show|View|See)\s*(?:more|less)(?:[\s/.,-]*(?:Read|Show|View|See)\s*(?:more|less))*(?:$|\b|(?=[A-Z\s.,!?]))/gi, " ")
        .replace(/\bTranslate review to English\b\.?/gi, " ")
        .replace(/\bTranslated by Amazon\b\.?/gi, " ")
        .replace(/<[^>]*>/g, " ")
        .replace(/&nbsp;/gi, " ")
        .replace(/\s+/g, " ")
        .trim();
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

  // Sort Toggle Click Listener (2)
  if (sortToggleBtn) {
    sortToggleBtn.addEventListener("click", () => {
      if (currentSort === "suspicious") {
        currentSort = "media";
        if (sortToggleText) sortToggleText.textContent = "Media First";
        sortToggleBtn.setAttribute("data-sort", "media");
      } else {
        currentSort = "suspicious";
        if (sortToggleText) sortToggleText.textContent = "Suspicious First";
        sortToggleBtn.setAttribute("data-sort", "suspicious");
      }

      if (currentAnalysisData && currentAnalysisData.results) {
        const reviewMap = new Map();
        if (Array.isArray(currentReviews)) {
          currentReviews.forEach((rev, idx) => {
            if (rev && rev.id) reviewMap.set(rev.id, rev);
            reviewMap.set(`idx_${idx}`, rev || {});
          });
        }
        renderReviewsList(currentAnalysisData.results, reviewMap);
      }
    });
  }

  // Filter Chips Click Listeners (2)
  filterChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      filterChips.forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      currentFilter = chip.getAttribute("data-filter") || "all";

      if (currentAnalysisData && currentAnalysisData.results) {
        const reviewMap = new Map();
        if (Array.isArray(currentReviews)) {
          currentReviews.forEach((rev, idx) => {
            if (rev && rev.id) reviewMap.set(rev.id, rev);
            reviewMap.set(`idx_${idx}`, rev || {});
          });
        }
        renderReviewsList(currentAnalysisData.results, reviewMap);
      }
    });
  });

  // Initialize
  init();
});
