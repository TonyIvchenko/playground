"use strict";

(function initBrowserActionHelpers() {
  function bindDatasetButtons(buttons, datasetKey, onSelect) {
    for (const button of Array.from(buttons)) {
      button.addEventListener("click", async () => {
        await onSelect(button.dataset[datasetKey], button);
      });
    }
  }

  function bindFirstFileInput(input, onSelect) {
    input.addEventListener("change", async () => {
      const [file] = input.files || [];
      await onSelect(file || null, input);
    });
  }

  async function readTextFile(file) {
    if (!file) {
      return "";
    }
    return (await file.text()).trim();
  }

  window.PlaygroundBrowserActions = Object.freeze({
    bindDatasetButtons,
    bindFirstFileInput,
    readTextFile,
  });
})();
