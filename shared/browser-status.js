"use strict";

(function initBrowserStatusHelpers() {
  function setStateText(
    element,
    message,
    { isError = false, isLoading = false } = {},
  ) {
    element.textContent = message;
    element.classList.toggle("is-error", isError);
    element.classList.toggle("is-loading", isLoading);
  }

  function createStateTextSetter(element) {
    return (message, options = {}) => {
      setStateText(element, message, options);
    };
  }

  window.PlaygroundBrowserStatus = Object.freeze({
    setStateText,
    createStateTextSetter,
  });
})();
