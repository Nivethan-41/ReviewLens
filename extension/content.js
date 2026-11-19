/**
 * ReviewLens Content Script
 * Extracts product reviews from active web pages using schema.org markup,
 * semantic review attributes, and fallback pattern detection.
 */

/**
 * Site-specific selectors dictionary for e-commerce platforms.
 * Centralized repository for platform-specific DOM hooks.
 * TODO: Refine and extend site-specific selectors in the second half.
 */
const SELECTORS = {
  amazon: {
    // Review card container
    reviewCard: "[data-hook='review']",
    // Review content body
    reviewBody: "[data-hook='review-body'], [data-hook='review-collapsed']",
    // Star rating element
    rating: "[data-hook='review-star-rating'], [data-hook='cmps-review-star-rating'], i.a-icon-star",
    // Review submission date
    date: "[data-hook='review-date']",
    // Review title
    title: "[data-hook='review-title']",
    // Reviewer name
    author: "[data-hook='genome-widget'] .a-profile-name",
    // TODO: Refine for Amazon global domains, mobile layouts, and verified purchase badges
  },
  flipkart: {
    // Review card container
    reviewCard: "div._27M-f-, div._1AtVbE",
    // Review content body
    reviewBody: "div.t-ZTKy, div._1135gP, div.ZmyHeo",
    // Star rating badge
    rating: "div._3LWZlK, div.XQDdHH",
    // Review date
    date: "p._2sc7ZR, p._2nQfGI",
    // Reviewer name
    author: "p._2V5RqD, p._330-kX",
    // TODO: Refine for Flipkart infinite-scroll containers and dynamic class hashes
  },
};

/**
 * Helper to clean and normalize whitespace from extracted text.
 * @param {string} str - Raw input string.
 * @returns {string} Trimmed string with collapsed spaces.
 */
function cleanText(str) {
  if (!str) return "";
  return str.replace(/\s+/g, " ").trim();
}

/**
 * Attempt to parse a numerical rating (1.0 to 5.0) from text or element attributes.
 * @param {Element|string} input - DOM element or text string.
 * @returns {number|null} Parsed rating float, or null if unparseable.
 */
function parseRating(input) {
  if (!input) return null;

  let textToScan = "";
  if (typeof input === "string") {
    textToScan = input;
  } else if (input.nodeType) {
    textToScan = [
      input.getAttribute("content"),
      input.getAttribute("aria-label"),
      input.getAttribute("title"),
      input.getAttribute("data-rating"),
      input.textContent,
    ]
      .filter(Boolean)
      .join(" ");
  }

  if (!textToScan) return null;

  // Pattern: "4.0 out of 5 stars", "4.5 out of 5", "5/5"
  const outOfMatch = textToScan.match(/([1-5](?:\.\d+)?)\s*(?:out of|\/)\s*5(?:\.0)?/i);
  if (outOfMatch) {
    const val = parseFloat(outOfMatch[1]);
    if (!isNaN(val) && val >= 1.0 && val <= 5.0) return Math.round(val * 10) / 10;
  }

  // Pattern: "5 stars", "4.5 star"
  const starMatch = textToScan.match(/([1-5](?:\.\d+)?)\s*stars?/i);
  if (starMatch) {
    const val = parseFloat(starMatch[1]);
    if (!isNaN(val) && val >= 1.0 && val <= 5.0) return Math.round(val * 10) / 10;
  }

  // Pattern: Unicode stars count: ★★★★☆
  const filledStars = (textToScan.match(/[★]/g) || []).length;
  if (filledStars >= 1 && filledStars <= 5) {
    return filledStars;
  }

  // Pattern: Single standalone rating number like "4.5" or "5"
  const singleMatch = textToScan.trim().match(/^([1-5](?:\.\d+)?)$/);
  if (singleMatch) {
    const val = parseFloat(singleMatch[1]);
    if (!isNaN(val) && val >= 1.0 && val <= 5.0) return Math.round(val * 10) / 10;
  }

  return null;
}

/**
 * Attempt to parse a date string into an ISO 8601 (YYYY-MM-DD) formatted string.
 * @param {Element|string} input - DOM element or text string.
 * @returns {string|null} ISO date string (YYYY-MM-DD), or null if unparseable.
 */
function parseDate(input) {
  if (!input) return null;

  let raw = "";
  if (typeof input === "string") {
    raw = input;
  } else if (input.nodeType) {
    raw =
      input.getAttribute("datetime") ||
      input.getAttribute("content") ||
      input.textContent ||
      "";
  }

  const text = cleanText(raw);
  if (!text) return null;

  // Direct ISO match (YYYY-MM-DD)
  const isoMatch = text.match(/\b(\d{4})-(\d{2})-(\d{2})\b/);
  if (isoMatch) {
    return `${isoMatch[1]}-${isoMatch[2]}-${isoMatch[3]}`;
  }

  // Remove common prefix phrases like "Reviewed on", "Date:", "on"
  const stripped = text
    .replace(/^(reviewed in [^on]+ on|reviewed on|posted on|date:?)\s*/i, "")
    .trim();

  // Try Date constructor
  const parsed = new Date(stripped);
  if (!isNaN(parsed.getTime())) {
    const year = parsed.getUTCFullYear();
    // Validate reasonable review year boundary
    if (year >= 2000 && year <= 2035) {
      return parsed.toISOString().split("T")[0];
    }
  }

  return null;
}

/**
 * Helper to locate the closest containing container for a review element.
 * @param {Element} el - DOM element representing review text.
 * @returns {Element} Review card container or parent.
 */
function findReviewContainer(el) {
  return (
    el.closest(
      "[itemscope][itemtype*='Review'], [itemprop='review'], [data-hook='review'], [class*='review-item' i], [class*='review-card' i], article, li"
    ) || el.parentElement
  );
}

/**
 * Extract rating and date from an enclosing review container element.
 * @param {Element} container - Container element.
 * @returns {{ rating: number|null, date: string|null }}
 */
function extractMetaFromContainer(container) {
  if (!container) return { rating: null, date: null };

  // Rating search
  let rating = null;
  const ratingEl = container.querySelector(
    "[itemprop='reviewRating'] [itemprop='ratingValue'], [itemprop='ratingValue'], [itemprop='reviewRating'], " +
      "[data-hook*='rating'], [data-hook*='star'], [aria-label*='out of 5' i], [aria-label*='star' i], " +
      "[class*='rating' i], [class*='star' i]"
  );
  if (ratingEl) {
    rating = parseRating(ratingEl);
  }

  // Date search
  let date = null;
  const dateEl = container.querySelector(
    "[itemprop='datePublished'], time, [data-hook*='date'], [class*='date' i]"
  );
  if (dateEl) {
    date = parseDate(dateEl);
  }

  return { rating, date };
}

/**
 * Strategy a: Extract reviews from JSON-LD scripts and schema.org microdata blocks.
 * @returns {Array<{id?: string, text: string, rating: number|null, date: string|null}>}
 */
function extractFromSchema() {
  const reviews = [];

  // a1. JSON-LD scripts
  const scripts = document.querySelectorAll('script[type="application/ld+json"]');
  scripts.forEach((script) => {
    try {
      const data = JSON.parse(script.textContent || "{}");
      const collectFromObj = (obj) => {
        if (!obj || typeof obj !== "object") return;

        // Check if object is a Review
        const type = obj["@type"];
        const isReview =
          type === "Review" || (Array.isArray(type) && type.includes("Review"));

        if (isReview && (obj.reviewBody || obj.description)) {
          const body = cleanText(obj.reviewBody || obj.description);
          if (body.length >= 10) {
            let rating = null;
            if (obj.reviewRating) {
              rating = parseRating(
                String(obj.reviewRating.ratingValue ?? obj.reviewRating)
              );
            }
            let date = parseDate(obj.datePublished);
            reviews.push({
              id: obj["@id"] || obj.identifier || undefined,
              text: body,
              rating,
              date,
            });
          }
        }

        // Recursively inspect 'review' arrays/properties
        if (obj.review) {
          const reviewList = Array.isArray(obj.review) ? obj.review : [obj.review];
          reviewList.forEach((r) => collectFromObj(r));
        }

        // Check '@graph' arrays
        if (Array.isArray(obj["@graph"])) {
          obj["@graph"].forEach((item) => collectFromObj(item));
        }
      };

      if (Array.isArray(data)) {
        data.forEach((item) => collectFromObj(item));
      } else {
        collectFromObj(data);
      }
    } catch (_) {
      // Ignore malformed JSON-LD scripts
    }
  });

  // a2. Schema.org Microdata elements (itemprop="review" or itemtype Review)
  const microdataReviews = document.querySelectorAll(
    "[itemscope][itemtype*='Review'], [itemprop='review']"
  );
  microdataReviews.forEach((card, idx) => {
    const bodyEl = card.querySelector("[itemprop='reviewBody']");
    if (!bodyEl) return;

    const text = cleanText(bodyEl.textContent);
    if (text.length < 10) return;

    const meta = extractMetaFromContainer(card);
    reviews.push({
      id: card.id || `r${idx + 1}`,
      text,
      rating: meta.rating,
      date: meta.date,
    });
  });

  return reviews;
}

/**
 * Strategy b: Extract elements with itemprop='reviewBody', or attributes
 * matching review + (body/text/content) like Amazon's data-hook='review-body'.
 * @returns {Array<{id?: string, text: string, rating: number|null, date: string|null}>}
 */
function extractFromSemanticAttributes() {
  const reviews = [];
  const candidates = document.querySelectorAll(
    "[itemprop='reviewBody'], " +
      "[data-hook*='review'][data-hook*='body'], " +
      "[data-hook*='review'][data-hook*='text'], " +
      "[data-hook*='review'][data-hook*='content'], " +
      "[class*='review' i], " +
      "[id*='review' i], " +
      "[data-hook*='review' i]"
  );

  candidates.forEach((el, idx) => {
    // Check if element qualifies as review body
    const isItemprop = el.getAttribute("itemprop") === "reviewBody";
    let matchesAttr = false;

    for (const attr of ["data-hook", "class", "id"]) {
      const val = (el.getAttribute(attr) || "").toLowerCase();
      if (
        val.includes("review") &&
        (val.includes("body") || val.includes("text") || val.includes("content"))
      ) {
        matchesAttr = true;
        break;
      }
    }

    if (!isItemprop && !matchesAttr) return;

    // Filter out buttons, inputs, titles, headers
    const tag = el.tagName.toLowerCase();
    if (["button", "input", "script", "style", "nav", "header"].includes(tag)) {
      return;
    }

    const text = cleanText(el.textContent);
    if (text.length < 10 || text.length > 5000) return;

    const container = findReviewContainer(el);
    const meta = extractMetaFromContainer(container);

    reviews.push({
      id: container.id || el.id || `r${idx + 1}`,
      text,
      rating: meta.rating,
      date: meta.date,
    });
  });

  return reviews;
}

/**
 * Strategy c (Fallback): Inspect repeated sibling blocks containing
 * a star rating pattern plus 40-3000 characters of text.
 * @returns {Array<{id?: string, text: string, rating: number|null, date: string|null}>}
 */
function extractFromRepeatedSiblings() {
  const reviews = [];
  const starPattern = /([1-5](?:\.\d+)?)\s*(?:out of|\/)\s*5|([1-5](?:\.\d+)?)\s*stars?|[★☆]{1,5}/i;

  // Search all containers with at least 2 children
  const containers = document.querySelectorAll("div, section, ul, ol, main, article");

  for (const container of containers) {
    const children = Array.from(container.children);
    if (children.length < 2) continue;

    const siblingMatches = [];
    for (let i = 0; i < children.length; i++) {
      const child = children[i];
      const text = cleanText(child.textContent);

      // Check text length constraint (40-3000 chars)
      if (text.length < 40 || text.length > 3000) continue;

      // Check star rating pattern
      const hasRatingPattern =
        starPattern.test(child.textContent) ||
        starPattern.test(child.getAttribute("aria-label") || "") ||
        child.querySelector("[aria-label*='star' i], [class*='star' i], [class*='rating' i]") !== null;

      if (hasRatingPattern) {
        const rating = parseRating(child);
        const date = parseDate(child);

        // Find best text subsegment (e.g. paragraph or block with main review content)
        const paragraphs = Array.from(child.querySelectorAll("p, div, span")).filter(
          (p) => {
            const pText = cleanText(p.textContent);
            return pText.length >= 40 && pText.length <= 3000;
          }
        );

        const reviewText = paragraphs.length > 0
          ? cleanText(paragraphs[0].textContent)
          : text;

        siblingMatches.push({
          id: child.id || `r${i + 1}`,
          text: reviewText,
          rating,
          date,
        });
      }
    }

    if (siblingMatches.length >= 2) {
      reviews.push(...siblingMatches);
      break; // Use the first substantial repeating review block
    }
  }

  return reviews;
}

/**
 * Deduplicate reviews by normalized text, cap at 100, and ensure sequential IDs.
 * @param {Array<{id?: string, text: string, rating: number|null, date: string|null}>} rawReviews
 * @returns {Array<{id: string, text: string, rating: number|null, date: string|null}>}
 */
function deduplicateAndFormat(rawReviews) {
  const seenTexts = new Set();
  const result = [];

  for (const item of rawReviews) {
    if (!item || !item.text) continue;
    const clean = cleanText(item.text);
    if (clean.length < 10) continue;

    const key = clean.toLowerCase();
    if (!seenTexts.has(key)) {
      seenTexts.add(key);
      result.push({
        id: item.id || `r${result.length + 1}`,
        text: clean,
        rating: typeof item.rating === "number" ? item.rating : null,
        date: item.date || null,
      });

      if (result.length >= 100) break;
    }
  }

  return result;
}

/**
 * Main review extraction function.
 * Implements a generic 3-tier strategy in exact order:
 * 1. JSON-LD / schema.org Review and AggregateRating blocks
 * 2. Elements with itemprop='reviewBody' or review + (body/text/content) attributes
 * 3. Fallback: repeated sibling blocks with star rating pattern + 40-3000 chars text
 *
 * @returns {Array<{id: string, text: string, rating: number|null, date: string|null}>}
 */
function extractReviews() {
  // Strategy a: JSON-LD and Schema.org Microdata
  let rawReviews = extractFromSchema();

  // Strategy b: Semantic attributes (reviewBody, data-hook review-body, etc.)
  if (rawReviews.length === 0) {
    rawReviews = extractFromSemanticAttributes();
  }

  // Strategy c: Fallback repeated sibling blocks
  if (rawReviews.length === 0) {
    rawReviews = extractFromRepeatedSiblings();
  }

  const finalReviews = deduplicateAndFormat(rawReviews);
  console.log(`[ReviewLens] Extracted ${finalReviews.length} reviews.`);
  return finalReviews;
}

// Attach to window for executeScript in Chrome Extension MV3
if (typeof window !== "undefined") {
  window.extractReviews = extractReviews;
  window.SELECTORS = SELECTORS;
}

// Export for Node.js / Jest / test suites
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    SELECTORS,
    extractReviews,
    cleanText,
    parseRating,
    parseDate,
    extractFromSchema,
    extractFromSemanticAttributes,
    extractFromRepeatedSiblings,
    deduplicateAndFormat,
  };
}
