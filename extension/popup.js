/**
 * ReviewLens Popup Script
 * Coordinates user actions, triggers review analysis, and renders results/summary UI.
 */

document.addEventListener("DOMContentLoaded", () => {
  const analyzeBtn = document.getElementById("analyze-btn");
  if (analyzeBtn) {
    analyzeBtn.addEventListener("click", () => {
      console.log("Analyze button clicked.");
    });
  }
});

