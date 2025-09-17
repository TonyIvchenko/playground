"use strict";

(function initBrowserMathHelpers() {
  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  window.PlaygroundBrowserMath = Object.freeze({
    clamp,
  });
})();
