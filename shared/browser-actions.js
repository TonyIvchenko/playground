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

  function downloadTextFile(
    filename,
    content,
    mimeType = "text/plain;charset=utf-8"
  ) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
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
    downloadTextFile,
    readTextFile,
  });
})();
