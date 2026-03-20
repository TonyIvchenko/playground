window.bootstrapDisastersMap = () => {
  const root = document.getElementById("risk-map-shell");
  if (!root || root.dataset.ready === "1") return [];
  root.dataset.ready = "1";

  let cfg = {};
  try {
    cfg = JSON.parse(root.dataset.config || "{}");
  } catch (err) {
    console.error("Failed to parse map config", err);
  }

  const slider = root.querySelector("#risk-time-slider");
  const playBtn = root.querySelector("#risk-play");
  const progressNode = root.querySelector("#risk-time-progress");
  const markerNode = root.querySelector("#risk-now-marker");
  const mapNode = root.querySelector("#risk-map");
  const statusNode = root.querySelector("#risk-map-status");

  const yearTicks = Array.from(root.querySelectorAll(".year-tick"));
  const hazardSelect = root.querySelector("#hazard-select");
  const modelMetricAcc = root.querySelector("#model-metric-acc");
  const modelMetricAuc = root.querySelector("#model-metric-auc");

  const frameCount =
    Array.isArray(cfg.frames) && cfg.frames.length > 0 ? cfg.frames.length : 1;
  const maxFrameIdx = Math.max(1, frameCount - 1);
  if (slider) {
    slider.max = String(frameCount - 1);
  }

  let timer = null;
  let map = null;

  const updateStatus = (text, isError = false) => {
    if (!text || !isError) {
      statusNode.textContent = "";
      statusNode.classList.remove("show", "error");
      return;
    }
    statusNode.textContent = text;
    statusNode.classList.add("show");
    statusNode.classList.toggle("error", Boolean(isError));
  };

  const fmtPct = (value) => {
    if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
    return `${(value * 100).toFixed(2)}%`;
  };

  const renderModelSummary = () => {
    const key = hazardSelect.value;
    const model = (cfg.model_metrics && cfg.model_metrics[key]) || null;
    modelMetricAcc.textContent = model ? fmtPct(model.val_accuracy) : "n/a";
    modelMetricAuc.textContent = model ? fmtPct(model.val_auc) : "n/a";
  };

  const updateTimeline = () => {
    const idx = Math.min(maxFrameIdx, Math.max(0, Number(slider.value) || 0));
    slider.value = String(idx);
    const pct = (idx / maxFrameIdx) * 100.0;
    progressNode.style.width = `${pct}%`;
    markerNode.style.left = `${pct}%`;

    let activeYear = -1;
    yearTicks.forEach((tick, i) => {
      const startIdx = Number(tick.dataset.frameIndex || "0");
      if (idx >= startIdx) {
        activeYear = i;
      }
    });
    yearTicks.forEach((tick, i) => tick.classList.toggle("active", i === activeYear));
  };

  const setPlaying = (on) => {
    if (on && !timer) {
      playBtn.classList.add("playing");
      playBtn.setAttribute("aria-label", "Pause timeline");
      timer = setInterval(() => {
        const next = (Number(slider.value) + 1) % frameCount;
        slider.value = String(next);
        installOverlay();
      }, 900);
      return;
    }
    if (!on && timer) {
      clearInterval(timer);
      timer = null;
    }
    playBtn.classList.remove("playing");
    playBtn.setAttribute("aria-label", "Play timeline");
  };

  const tileUrl = (hazard, frameIdx, z, x, y) =>
    `/tiles/${hazard}/${frameIdx}/${z}/${x}/${y}.png`;

  const clearOverlays = () => {
    if (!map) return;
    map.overlayMapTypes.clear();
  };

  const pushOverlay = (hazard, frameIdx) => {
    if (!map) return;
    const overlay = new google.maps.ImageMapType({
      tileSize: new google.maps.Size(256, 256),
      opacity: 1.0,
      getTileUrl: (coord, zoom) => {
        if (coord.y < 0 || coord.y >= 1 << zoom) return "";
        const wrappedX = ((coord.x % (1 << zoom)) + (1 << zoom)) % (1 << zoom);
        return tileUrl(hazard, frameIdx, zoom, wrappedX, coord.y);
      },
    });
    map.overlayMapTypes.push(overlay);
  };

  const installOverlay = () => {
    updateTimeline();
    renderModelSummary();
    if (!map) return;

    const frameIdx = Number(slider.value);
    clearOverlays();

    const selectedHazard = hazardSelect.value;
    if (selectedHazard === "wildfires" || selectedHazard === "huricaines") {
      pushOverlay(selectedHazard, frameIdx);
      return;
    }
    updateStatus("Unknown layer selection.", true);
  };

  const initGoogleMap = () => {
    map = new google.maps.Map(mapNode, {
      center: {
        lat: Number(cfg.center_lat || 36.0),
        lng: Number(cfg.center_lon || -95.0),
      },
      zoom: Number(cfg.default_zoom || 4),
      minZoom: Number(cfg.zoom_min || 2),
      maxZoom: Number(cfg.zoom_max || 10),
      mapTypeControl: false,
      streetViewControl: false,
      fullscreenControl: false,
      zoomControl: true,
      zoomControlOptions: { position: google.maps.ControlPosition.RIGHT_BOTTOM },
      rotateControl: false,
      scaleControl: false,
      clickableIcons: false,
    });
    setTimeout(() => {
      if (window.google && window.google.maps) {
        google.maps.event.trigger(map, "resize");
      }
    }, 150);
    installOverlay();
  };

  const loadGoogleMaps = () => {
    if (!cfg.api_key) {
      updateStatus("GMAPS_API_KEY is required.", true);
      return;
    }
    if (window.google && window.google.maps) {
      initGoogleMap();
      return;
    }
    const callbackName = `gmapsInit_${cfg.service_id}_${Date.now()}`;
    window[callbackName] = () => {
      delete window[callbackName];
      initGoogleMap();
    };
    const script = document.createElement("script");
    script.src = `https://maps.googleapis.com/maps/api/js?key=${cfg.api_key}&callback=${callbackName}&v=weekly`;
    script.async = true;
    script.defer = true;
    script.onerror = () => {
      updateStatus("Failed to load Google Maps JavaScript API.", true);
    };
    document.head.appendChild(script);
  };

  slider.addEventListener("input", installOverlay);
  hazardSelect.addEventListener("change", installOverlay);
  playBtn.addEventListener("click", () => setPlaying(!timer));

  updateTimeline();
  renderModelSummary();
  loadGoogleMaps();
  return [];
};
