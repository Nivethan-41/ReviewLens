/**
 * ReviewLens Content Script
 * Extracts product reviews from active web pages using schema.org markup,
 * semantic review attributes, Indian e-commerce platform selectors,
 * and fallback pattern detection.
 */

/**
 * Site-specific selectors dictionary for e-commerce platforms.
 * Explicitly supports Indian e-commerce platforms:
 * Amazon India (amazon.in), Flipkart (flipkart.com), and Meesho / Generic Indian sites.
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
    // Media indicators: .review-image-tile, [data-hook="review-image-tile"], video tags
    mediaImage: ".review-image-tile, [data-hook='review-image-tile'], .review-image-tile-section img, [data-hook='review-image-container'] img",
    mediaVideo: "video, [data-hook='review-video'], .cr-video-desktop",
    // Verified purchase badge: [data-hook="avp-badge"] ("Verified Purchase")
    verifiedPurchase: "[data-hook='avp-badge']",
    // Seller card: #merchant-info, #sellerProfileTriggerId
    sellerCard: "#merchant-info, #sellerProfileTriggerId",
    // Seller rating
    sellerRating: "#seller-rating, .seller-rating, #merchant-info [class*='rating']",
  },
  flipkart: {
    // Review card container: div._27M-vq, div.col._2wzgFH
    reviewCard: "div._27M-vq, div.col._2wzgFH, div._27M-f-, div._1AtVbE",
    // Review content body
    reviewBody: "div.t-ZTKy, div._1135gP, div.ZmyHeo",
    // Star rating badge
    rating: "div._3LWZlK, div.XQDdHH",
    // Review date
    date: "p._2sc7ZR, p._2nQfGI",
    // Reviewer name
    author: "p._2V5RqD, p._330-kX",
    // Media indicators: div._2tNq8e, img._3kidjx, video thumbnails
    mediaImage: "div._2tNq8e, img._3kidjx, div._2103do img",
    mediaVideo: "video, [class*='video' i], ._30kQv0, [data-testid*='video']",
    // Verified purchase: ._2mcSum ("Certified Buyer")
    verifiedPurchase: "._2mcSum, [class*='_2mcSum']",
    // Seller card: #sellerName, ._1RLviY
    sellerCard: "#sellerName, ._1RLviY, div._2d4LTz",
    // Seller rating
    sellerRating: "div._3LWZlK, div.XQDdHH, ._1lRcqv",
  },
  meesho: {
    // Review containers
    reviewCard: "div[class*='ReviewCard'], div[class*='CommentContainer'], div[class*='review-card'], div[class*='SupplierReviewCard']",
    // Review content body
    reviewBody: "p[class*='ReviewText'], span[class*='ReviewText'], div[class*='review-text'], p[class*='CommentText']",
    // Rating element
    rating: "div[class*='RatingBadge'], span[class*='RatingBadge'], [class*='rating']",
    // Date
    date: "span[class*='ReviewDate'], span[class*='DateText'], [class*='date']",
    // Reviewer name
    author: "span[class*='ReviewerName'], span[class*='AuthorName']",
    // Media indicators: .review-image, img tags within review cards, video tags
    mediaImage: ".review-image, img[class*='ReviewImage'], div[class*='ReviewImages'] img, img.review-image, img",
    mediaVideo: "video, div[class*='ReviewVideo'], [class*='video' i]",
    // Verified purchase
    verifiedPurchase: "span[class*='VerifiedBadge'], [class*='verified' i]",
    // Supplier name / card
    sellerCard: "div[class*='SupplierCard'], div[class*='SupplierInfo'], div[class*='ShopNameContainer'], [class*='supplier-name'], #supplierName",
    // Supplier rating
    sellerRating: "span[class*='SupplierRating'], [class*='rating']",
  },
  generic: {
    // Review containers
    reviewCard: "[itemscope][itemtype*='Review'], [itemprop='review'], [class*='review-card' i], [class*='review-item' i], article.review",
    // Review body
    reviewBody: "[itemprop='reviewBody'], .review-text, .review-body, .review-content, p.comment",
    // Rating
    rating: "[itemprop='reviewRating'] [itemprop='ratingValue'], [itemprop='ratingValue'], [class*='rating' i], [class*='star' i]",
    // Date
    date: "[itemprop='datePublished'], time, [class*='date' i]",
    // Media indicators
    mediaImage: ".review-image, img.review-image, [class*='review-image' i], img[class*='review-photo' i]",
    mediaVideo: "video, [class*='review-video' i], [class*='video' i]",
    // Verified purchase
    verifiedPurchase: "[class*='verified' i], [class*='certified' i]",
    // Seller / supplier card
    sellerCard: "#merchant-info, #sellerName, #supplierName, [class*='seller-info' i], [class*='supplier' i]",
    // Seller / supplier rating
    sellerRating: "[class*='seller-rating' i], [class*='supplier-rating' i], [class*='rating' i]",
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
 * @param {Element|string|number} input - DOM element, text string, or number.
 * @returns {number|null} Parsed rating float, or null if unparseable.
 */
function parseRating(input) {
  if (input === null || input === undefined) return null;

  if (typeof input === "number") {
    return input >= 1.0 && input <= 5.0 ? Math.round(input * 10) / 10 : null;
  }

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

  // Pattern: Number with star symbol, e.g. "4.4 ★" or "★ 4.4" or "4.4 *"
  const starWithNumMatch = textToScan.match(/(?:[★*]\s*)?([1-5](?:\.\d+)?)\s*[★*]/);
  if (starWithNumMatch) {
    const val = parseFloat(starWithNumMatch[1]);
    if (!isNaN(val) && val >= 1.0 && val <= 5.0) return Math.round(val * 10) / 10;
  }

  // Pattern: Unicode stars count: ★★★★☆ (only if 2+ stars present)
  const filledStars = (textToScan.match(/[★]/g) || []).length;
  if (filledStars >= 2 && filledStars <= 5) {
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

  // Remove common prefix phrases like "Reviewed in India on", "Reviewed on", "Date:", "on"
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
      "[itemscope][itemtype*='Review'], [itemprop='review'], [data-hook='review'], div._27M-vq, div.col._2wzgFH, [class*='ReviewCard'], [class*='review-item' i], [class*='review-card' i], article, li"
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
      "[data-hook*='rating'], [data-hook*='star'], div._3LWZlK, div.XQDdHH, " +
      "[aria-label*='out of 5' i], [aria-label*='star' i], [class*='rating' i], [class*='star' i]"
  );
  if (ratingEl) {
    rating = parseRating(ratingEl);
  }

  // Date search
  let date = null;
  const dateEl = container.querySelector(
    "[itemprop='datePublished'], time, [data-hook*='date'], p._2sc7ZR, p._2nQfGI, span[class*='Date'], [class*='date' i]"
  );
  if (dateEl) {
    date = parseDate(dateEl);
  }

  return { rating, date };
}

/**
 * Extract media indicators (photos/videos) and verified purchase status from a review card container.
 * Supports Amazon India (.review-image-tile, video tags, avp-badge),
 * Flipkart (div._2tNq8e, img._3kidjx, video thumbnails, ._2mcSum),
 * and Meesho / Generic sites (.review-image, img tags within cards, video tags).
 *
 * @param {Element} container - The DOM element containing the review.
 * @returns {{ has_images: boolean, has_video: boolean, verified_purchase: boolean }}
 */
function extractIndicatorsFromContainer(container) {
  let has_images = false;
  let has_video = false;
  let verified_purchase = false;

  if (!container || !container.querySelector) {
    return { has_images, has_video, verified_purchase };
  }

  // 1. Photo / Image indicators
  // Amazon India: .review-image-tile, [data-hook="review-image-tile"], [data-hook*="image-tile"]
  // Flipkart: div._2tNq8e, img._3kidjx, div._2103do img
  // Meesho / Generic: .review-image, img.review-image, img[class*='ReviewImage'], img[class*='review-image']
  const explicitImageSelectors = [
    ".review-image-tile",
    "[data-hook='review-image-tile']",
    "[data-hook*='image-tile']",
    "div._2tNq8e",
    "img._3kidjx",
    ".review-image",
    "img.review-image",
    "img[class*='ReviewImage']",
    "[class*='review-image' i]",
    "[data-hook='review-image-container'] img",
    "div._2103do img",
    "[class*='ReviewImages'] img",
  ].join(", ");

  if (container.querySelector(explicitImageSelectors)) {
    has_images = true;
  } else {
    // Scan all img tags in the card, filtering out profile avatars and star icons
    const allImgs = container.querySelectorAll("img");
    for (const img of allImgs) {
      const cls = (img.className || "").toLowerCase();
      const alt = (img.getAttribute("alt") || "").toLowerCase();
      const src = (img.getAttribute("src") || "").toLowerCase();
      const parentCls = (img.parentElement?.className || "").toLowerCase();

      const isAvatarOrIcon =
        cls.includes("avatar") ||
        cls.includes("profile") ||
        cls.includes("icon") ||
        cls.includes("star") ||
        parentCls.includes("avatar") ||
        parentCls.includes("profile") ||
        parentCls.includes("rating") ||
        alt.includes("avatar") ||
        alt.includes("profile") ||
        alt.includes("star") ||
        src.includes("avatar");

      if (!isAvatarOrIcon) {
        has_images = true;
        break;
      }
    }
  }

  // 2. Video indicators
  // Amazon India: video tags, [data-hook="review-video"], .cr-video-desktop
  // Flipkart: video thumbnails, div._30kQv0, [data-testid*='video'], video tags
  // Meesho / Generic: video tags, [class*='review-video'], [class*='video' i]
  const explicitVideoSelectors = [
    "video",
    "[data-hook='review-video']",
    ".cr-video-desktop",
    "div._30kQv0",
    "[class*='video-thumbnail' i]",
    "[class*='videoThumbnail' i]",
    "[class*='review-video' i]",
    "[class*='ReviewVideo' i]",
    "[data-testid*='video' i]",
    "div._2tNq8e[data-type*='video']",
  ].join(", ");

  if (container.querySelector(explicitVideoSelectors)) {
    has_video = true;
  }

  // 3. Verified Purchase indicator
  // Amazon India: [data-hook="avp-badge"] ("Verified Purchase")
  // Flipkart: ._2mcSum ("Certified Buyer")
  // Meesho / Generic: span.VerifiedBadge, [class*='verified' i], [class*='certified' i]
  const explicitVerifiedEl = container.querySelector(
    "[data-hook='avp-badge'], ._2mcSum, span[class*='VerifiedBadge'], [class*='verified-purchase' i], [class*='certified-buyer' i]"
  );
  if (explicitVerifiedEl) {
    verified_purchase = true;
  } else {
    // Check text for common verification keywords
    const text = container.textContent || "";
    if (
      /\bverified\s+purchase\b/i.test(text) ||
      /\bcertified\s+buyer\b/i.test(text) ||
      /\bverified\s+buyer\b/i.test(text)
    ) {
      verified_purchase = true;
    }
  }

  return { has_images, has_video, verified_purchase };
}

/**
 * Extract seller/merchant/supplier information from the page.
 * Supports Amazon India (#merchant-info, #sellerProfileTriggerId),
 * Flipkart (#sellerName, ._1RLviY), and Meesho / Generic Indian sites
 * (#supplierName, .SupplierCard, .supplier-name).
 *
 * @param {Document|Element} [root=document] - Root DOM node to search within.
 * @returns {{ name: string, rating: number|null } | null}
 */
function extractSellerInfo(root = (typeof document !== "undefined" ? document : null)) {
  if (!root || !root.querySelector) return null;

  // 1. Amazon India: #merchant-info, #sellerProfileTriggerId
  const amazonSellerTrigger = root.querySelector(SELECTORS.amazon.sellerCard);
  if (amazonSellerTrigger) {
    let name = "";
    const profileTrigger = root.querySelector("#sellerProfileTriggerId");
    if (profileTrigger && cleanText(profileTrigger.textContent)) {
      name = cleanText(profileTrigger.textContent);
    } else {
      const merchantInfo = root.querySelector("#merchant-info") || amazonSellerTrigger;
      const mText = cleanText(merchantInfo.textContent);
      // Pattern: "Sold by X and Fulfilled by Y" or "Ships from and sold by X."
      const soldByMatch = mText.match(/(?:sold by|ships from and sold by)\s+([^.\n,]+?)(?:\s+and\s+|\.|$)/i);
      name = soldByMatch ? cleanText(soldByMatch[1]) : mText;
    }

    // Rating in or near merchant-info
    let rating = null;
    const ratingEl = root.querySelector(SELECTORS.amazon.sellerRating);
    if (ratingEl) {
      rating = parseRating(ratingEl);
    } else {
      const merchantInfo = root.querySelector("#merchant-info");
      if (merchantInfo) {
        rating = parseRating(merchantInfo.textContent);
      }
    }

    if (name) {
      return {
        name,
        rating: typeof rating === "number" ? rating : null,
      };
    }
  }

  // 2. Flipkart: #sellerName, ._1RLviY
  const flipkartSeller = root.querySelector(SELECTORS.flipkart.sellerCard);
  if (flipkartSeller) {
    let rating = null;
    const ratingEl = flipkartSeller.querySelector(SELECTORS.flipkart.sellerRating);
    if (ratingEl) {
      rating = parseRating(ratingEl);
    }

    let name = "";
    // Often #sellerName > span._1RLviY > span (name) + div._3LWZlK (rating)
    const innerNameSpan = flipkartSeller.querySelector("span > span, a");
    if (innerNameSpan && !innerNameSpan.querySelector(SELECTORS.flipkart.sellerRating)) {
      name = cleanText(innerNameSpan.textContent);
    }

    if (!name) {
      const clone = flipkartSeller.cloneNode(true);
      const badges = clone.querySelectorAll(SELECTORS.flipkart.sellerRating + ", svg, path");
      badges.forEach((b) => b.remove());
      name = cleanText(clone.textContent);
    }

    name = name.replace(/^(seller|sold by):?\s*/i, "").trim();
    name = name.replace(/\s+\d+\s+(?:days?|months?)\s+.*$/i, "").trim();

    if (name) {
      return {
        name,
        rating: typeof rating === "number" ? rating : null,
      };
    }
  }

  // 3. Meesho / Generic Indian sites: #supplierName, .SupplierCard, .supplier-name
  const meeshoSeller = root.querySelector(SELECTORS.meesho.sellerCard);
  if (meeshoSeller) {
    let rating = null;
    const ratingEl = meeshoSeller.querySelector(SELECTORS.meesho.sellerRating);
    if (ratingEl) {
      rating = parseRating(ratingEl);
    }

    let name = "";
    const nameEl = meeshoSeller.querySelector(
      "[class*='supplier-name' i], [class*='ShopName' i], span:first-child"
    );
    if (nameEl) {
      name = cleanText(nameEl.textContent);
    } else {
      const clone = meeshoSeller.cloneNode(true);
      const badges = clone.querySelectorAll(SELECTORS.meesho.sellerRating);
      badges.forEach((b) => b.remove());
      name = cleanText(clone.textContent);
    }

    name = name.replace(/^(supplier|seller|sold by|store):?\s*/i, "").trim();

    if (name) {
      return {
        name,
        rating: typeof rating === "number" ? rating : null,
      };
    }
  }

  return null;
}

/**
 * Strategy 0: Extract reviews using platform-specific selectors (Amazon India, Flipkart, Meesho).
 * Captures customer photos, videos, and verified purchase status.
 *
 * @returns {Array<{id?: string, text: string, rating: number|null, date: string|null, has_images: boolean, has_video: boolean, verified_purchase: boolean}>}
 */
function extractFromPlatformSelectors() {
  const reviews = [];

  // Amazon India cards: [data-hook="review"]
  const amazonCards = document.querySelectorAll(SELECTORS.amazon.reviewCard);
  amazonCards.forEach((card, idx) => {
    const bodyEl = card.querySelector(SELECTORS.amazon.reviewBody);
    const text = cleanText(bodyEl ? bodyEl.textContent : card.textContent);
    if (text.length >= 10) {
      const meta = extractMetaFromContainer(card);
      const indicators = extractIndicatorsFromContainer(card);
      reviews.push({
        id: card.id || `amz-${idx + 1}`,
        text,
        rating: meta.rating,
        date: meta.date,
        has_images: indicators.has_images,
        has_video: indicators.has_video,
        verified_purchase: indicators.verified_purchase,
      });
    }
  });

  // Flipkart cards: div._27M-vq, div.col._2wzgFH
  const flipkartCards = document.querySelectorAll(SELECTORS.flipkart.reviewCard);
  flipkartCards.forEach((card, idx) => {
    // If card is an outer container div._27M-vq that contains div.col._2wzgFH, skip outer to avoid duplicate
    if (card.matches && card.matches("div._27M-vq") && card.querySelector("div.col._2wzgFH")) {
      return;
    }
    const bodyEl = card.querySelector(SELECTORS.flipkart.reviewBody) || card;
    const text = cleanText(bodyEl ? bodyEl.textContent : card.textContent);
    if (text.length >= 10) {
      const fullContainer = card.closest("div._27M-vq") || card;
      const meta = extractMetaFromContainer(fullContainer);
      const indicators = extractIndicatorsFromContainer(fullContainer);
      reviews.push({
        id: card.id || fullContainer.id || `fk-${idx + 1}`,
        text,
        rating: meta.rating,
        date: meta.date,
        has_images: indicators.has_images,
        has_video: indicators.has_video,
        verified_purchase: indicators.verified_purchase,
      });
    }
  });

  // Meesho / Generic cards
  const meeshoCards = document.querySelectorAll(SELECTORS.meesho.reviewCard);
  meeshoCards.forEach((card, idx) => {
    // Avoid double counting if already matched by amazon or flipkart
    if (card.closest(SELECTORS.amazon.reviewCard) || card.closest(SELECTORS.flipkart.reviewCard)) {
      return;
    }
    const bodyEl = card.querySelector(SELECTORS.meesho.reviewBody) || card;
    const text = cleanText(bodyEl ? bodyEl.textContent : card.textContent);
    if (text.length >= 10) {
      const meta = extractMetaFromContainer(card);
      const indicators = extractIndicatorsFromContainer(card);
      reviews.push({
        id: card.id || `msh-${idx + 1}`,
        text,
        rating: meta.rating,
        date: meta.date,
        has_images: indicators.has_images,
        has_video: indicators.has_video,
        verified_purchase: indicators.verified_purchase,
      });
    }
  });

  return reviews;
}

/**
 * Strategy a: Extract reviews from JSON-LD scripts and schema.org microdata blocks.
 * @returns {Array<{id?: string, text: string, rating: number|null, date: string|null, has_images: boolean, has_video: boolean, verified_purchase: boolean}>}
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
              has_images: Boolean(obj.image || (Array.isArray(obj.associatedMedia) && obj.associatedMedia.length > 0)),
              has_video: Boolean(obj.video),
              verified_purchase: false,
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
    const indicators = extractIndicatorsFromContainer(card);
    reviews.push({
      id: card.id || `r${idx + 1}`,
      text,
      rating: meta.rating,
      date: meta.date,
      has_images: indicators.has_images,
      has_video: indicators.has_video,
      verified_purchase: indicators.verified_purchase,
    });
  });

  return reviews;
}

/**
 * Strategy b: Extract elements with itemprop='reviewBody', or attributes
 * matching review + (body/text/content) like Amazon's data-hook='review-body'.
 * @returns {Array<{id?: string, text: string, rating: number|null, date: string|null, has_images: boolean, has_video: boolean, verified_purchase: boolean}>}
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
    const indicators = extractIndicatorsFromContainer(container);

    reviews.push({
      id: container.id || el.id || `r${idx + 1}`,
      text,
      rating: meta.rating,
      date: meta.date,
      has_images: indicators.has_images,
      has_video: indicators.has_video,
      verified_purchase: indicators.verified_purchase,
    });
  });

  return reviews;
}

/**
 * Strategy c (Fallback): Inspect repeated sibling blocks containing
 * a star rating pattern plus 40-3000 characters of text.
 * @returns {Array<{id?: string, text: string, rating: number|null, date: string|null, has_images: boolean, has_video: boolean, verified_purchase: boolean}>}
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
        const indicators = extractIndicatorsFromContainer(child);

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
          has_images: indicators.has_images,
          has_video: indicators.has_video,
          verified_purchase: indicators.verified_purchase,
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
 * Deduplicate reviews by normalized text, cap at 100, and ensure sequential IDs and indicators.
 * @param {Array<{id?: string, text: string, rating: number|null, date: string|null, has_images?: boolean, has_video?: boolean, verified_purchase?: boolean}>} rawReviews
 * @returns {Array<{id: string, text: string, rating: number|null, date: string|null, has_images: boolean, has_video: boolean, verified_purchase: boolean}>}
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
        has_images: Boolean(item.has_images),
        has_video: Boolean(item.has_video),
        verified_purchase: Boolean(item.verified_purchase),
      });

      if (result.length >= 100) break;
    }
  }

  return result;
}

/**
 * Main review extraction function.
 * Implements a prioritized extraction strategy:
 * 0. Indian e-commerce platform-specific selectors (Amazon India, Flipkart, Meesho)
 * 1. JSON-LD / schema.org Review and AggregateRating blocks
 * 2. Elements with itemprop='reviewBody' or review + (body/text/content) attributes
 * 3. Fallback: repeated sibling blocks with star rating pattern + 40-3000 chars text
 *
 * @returns {{ reviews: Array<{id: string, text: string, rating: number|null, date: string|null, has_images: boolean, has_video: boolean, verified_purchase: boolean}>, seller_info: { name: string, rating: number|null } | null }}
 */
function extractReviews() {
  // Strategy 0: Platform-specific extraction (Amazon India, Flipkart, Meesho)
  let rawReviews = extractFromPlatformSelectors();

  // Strategy a: JSON-LD and Schema.org Microdata
  if (rawReviews.length === 0) {
    rawReviews = extractFromSchema();
  }

  // Strategy b: Semantic attributes (reviewBody, data-hook review-body, etc.)
  if (rawReviews.length === 0) {
    rawReviews = extractFromSemanticAttributes();
  }

  // Strategy c: Fallback repeated sibling blocks
  if (rawReviews.length === 0) {
    rawReviews = extractFromRepeatedSiblings();
  }

  const finalReviews = deduplicateAndFormat(rawReviews);
  const sellerInfo = extractSellerInfo();

  console.log(`[ReviewLens] Extracted ${finalReviews.length} reviews. Seller info:`, sellerInfo);

  return {
    reviews: finalReviews,
    seller_info: sellerInfo,
  };
}

// Attach to window for executeScript in Chrome Extension MV3
if (typeof window !== "undefined") {
  window.extractReviews = extractReviews;
  window.SELECTORS = SELECTORS;
  window.extractSellerInfo = extractSellerInfo;
  window.extractIndicatorsFromContainer = extractIndicatorsFromContainer;
}

// Export for Node.js / Jest / test suites
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    SELECTORS,
    extractReviews,
    extractSellerInfo,
    extractIndicatorsFromContainer,
    extractFromPlatformSelectors,
    cleanText,
    parseRating,
    parseDate,
    extractFromSchema,
    extractFromSemanticAttributes,
    extractFromRepeatedSiblings,
    deduplicateAndFormat,
  };
}
