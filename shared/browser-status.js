"use strict";

(function initBrowserStatusHelpers() {
  function setStatusState(element, state = "default", message) {
    if (message !== undefined) {
      element.textContent = message;
    }

    element.hidden = state === "hidden";
    element.classList.toggle("is-error", state === "error");
    element.classList.toggle("is-loading", state === "loading");
    element.classList.toggle("is-fallback", state === "fallback");
  }

  function setStateText(
    element,
    message,
    { isError = false, isLoading = false } = {},
  ) {
    setStatusState(
      element,
      isError ? "error" : isLoading ? "loading" : "default",
      message,
    );
  }

  function createStateTextSetter(element) {
    return (message, options = {}) => {
      setStateText(element, message, options);
    };
  }

  window.PlaygroundBrowserStatus = Object.freeze({
    setStatusState,
    setStateText,
    createStateTextSetter,
  });
})();
