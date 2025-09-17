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

  async function copyTextToClipboard(text) {
    if (navigator.clipboard?.writeText && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }

    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    textarea.style.pointerEvents = "none";
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();
    textarea.setSelectionRange(0, textarea.value.length);

    try {
      return document.execCommand("copy");
    } finally {
      textarea.remove();
    }
  }

  function createSoftLimitGuidanceUpdater({
    countElement,
    noteElement,
    limit,
    measure,
    singularUnit,
    pluralUnit,
    underLimitMessage,
    overLimitMessage,
  }) {
    return (text) => {
      const count = measure(text);
      const overLimit = count > limit;
      countElement.textContent = `${count} ${count === 1 ? singularUnit : pluralUnit}`;
      countElement.classList.toggle("is-over", overLimit);
      noteElement.textContent = overLimit
        ? overLimitMessage(limit, count)
        : underLimitMessage(limit, count);
      noteElement.classList.toggle("is-over", overLimit);
    };
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
    copyTextToClipboard,
    createSoftLimitGuidanceUpdater,
    downloadTextFile,
    readTextFile,
  });
})();
