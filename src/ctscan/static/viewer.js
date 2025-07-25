(() => {
  function parseState(root) {
    const stateNode = root.querySelector(".ctscan-state");
    if (!stateNode) {
      return null;
    }
    try {
      return JSON.parse(stateNode.textContent || stateNode.value || "{}");
    } catch (error) {
      console.error("Failed to parse CT Scan viewer state.", error);
      return null;
    }
  }

  function initViewer(root) {
    if (!root || root.dataset.ctscanReady === "1") {
      return;
    }

    const state = parseState(root);
    if (!state) {
      return;
    }

    root.dataset.ctscanReady = "1";

    const overlay = root.querySelector(".ctscan-overlay-select");
    const findingWrap = root.querySelector(".ctscan-finding-wrap");
    const opacity = root.querySelector(".ctscan-opacity");
    const opacityValue = root.querySelector(".ctscan-opacity-value");
    const slice = root.querySelector(".ctscan-slice");
    const sliceLabel = root.querySelector(".ctscan-slice-label");
    const base = root.querySelector(".ctscan-base");
    const lung = root.querySelector(".ctscan-lung");
    const rows = Array.from(root.querySelectorAll(".ctscan-table tbody tr"));
    const findingImages = Object.fromEntries(
      (state.rows || []).map((row) => [
        row.key,
        root.querySelector(`.ctscan-finding[data-key="${row.key}"]`),
      ]),
    );
    const findingChecks = Array.from(
      root.querySelectorAll(".ctscan-finding-wrap input[type='checkbox']"),
    );

    function selectedKeys() {
      return new Set(
        findingChecks.filter((node) => node.checked).map((node) => node.value),
      );
    }

    function render() {
      const index = Number(slice.value || 0);
      const alpha = Number(opacity.value || 0);
      const mode = overlay.value;
      const selected = selectedKeys();

      base.src = `${state.asset_root}/base/${String(index).padStart(4, "0")}.png`;
      lung.src = `${state.asset_root}/lung/${String(index).padStart(4, "0")}.png`;
      lung.style.opacity = mode === "Lungs" ? String(alpha) : "0";
      findingWrap.style.display = mode === "Findings" ? "grid" : "none";
      opacityValue.textContent = alpha.toFixed(2);
      sliceLabel.textContent = `Slice ${index + 1} / ${state.slice_count}`;

      (state.rows || []).forEach((row, rowIndex) => {
        const image = findingImages[row.key];
        if (image) {
          image.src = `${state.asset_root}/findings/${row.key}/${String(index).padStart(4, "0")}.png`;
          image.style.opacity =
            mode === "Findings" && selected.has(row.key) ? String(alpha) : "0";
        }
        const cells = rows[rowIndex]?.querySelectorAll("td") || [];
        if (cells[3]) {
          cells[3].textContent = Number(row.slice_percents[index] || 0).toFixed(4);
        }
      });
    }

    overlay.addEventListener("change", render);
    opacity.addEventListener("input", render);
    slice.addEventListener("input", render);
    findingChecks.forEach((node) => node.addEventListener("change", render));
    render();
  }

  function scan() {
    document.querySelectorAll(".ctscan-viewer-root").forEach(initViewer);
  }

  function boot() {
    scan();
    const observer = new MutationObserver(() => scan());
    observer.observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }
})();
