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
  const timelineTicksNode = root.querySelector("#risk-timeline-ticks");
  const timelinePhasesNode = root.querySelector("#risk-timeline-phases");
  const timelineTrackNode = root.querySelector("#risk-timeline-track");
  const frameLabelNode = root.querySelector("#risk-frame-label");

  const hazardSelect = root.querySelector("#hazard-select");
  const metric1LabelNode = root.querySelector("#model-metric-1-label");
  const metric1ValueNode = root.querySelector("#model-metric-1-value");
  const metric2LabelNode = root.querySelector("#model-metric-2-label");
  const metric2ValueNode = root.querySelector("#model-metric-2-value");

  const hazards = cfg.hazards || {};
  if (hazardSelect && hazards[cfg.default_hazard]) {
    hazardSelect.value = cfg.default_hazard;
  }

  let timer = null;
  let map = null;

  const updateStatus = (text, isError = false) => {
    if (!text) {
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

  const currentHazardKey = () => hazardSelect.value;

  const currentHazardCfg = () => hazards[currentHazardKey()] || null;

  const currentFrames = () => {
    const hazardCfg = currentHazardCfg();
    if (!hazardCfg || !Array.isArray(hazardCfg.frames) || hazardCfg.frames.length === 0) {
      return ["Frame 0"];
    }
    return hazardCfg.frames;
  };

  const currentFrameCount = () => currentFrames().length;

  const maxFrameIdx = () => Math.max(1, currentFrameCount() - 1);

  const renderModelSummary = () => {
    const hazardCfg = currentHazardCfg();
    const metrics = Array.isArray(hazardCfg?.metrics) ? hazardCfg.metrics : [];
    const metric1 = metrics[0] || { label: "Metric 1", value: null };
    const metric2 = metrics[1] || { label: "Metric 2", value: null };
    metric1LabelNode.textContent = metric1.label || "Metric 1";
    metric1ValueNode.textContent = fmtPct(metric1.value);
    metric2LabelNode.textContent = metric2.label || "Metric 2";
    metric2ValueNode.textContent = fmtPct(metric2.value);
  };

  const renderTimelineScaffold = () => {
    const hazardCfg = currentHazardCfg();
    const timeline = hazardCfg?.timeline || {};
    const denom = maxFrameIdx();
    const stepPct = Number(timeline.step_pct || 1);
    timelineTrackNode.style.setProperty("--frame-step", `${stepPct}%`);

    const ticks = Array.isArray(timeline.ticks) ? timeline.ticks : [];
    timelineTicksNode.innerHTML = ticks
      .map((tick) => {
        const frameIdx = Number(tick.frame_idx || 0);
        const left = (frameIdx / denom) * 100.0;
        return `<div class="year-tick" data-frame-index="${frameIdx}" style="left:${left.toFixed(6)}%"><span>${tick.label || ""}</span></div>`;
      })
      .join("");

    const phases = Array.isArray(timeline.phases) ? timeline.phases : [];
    timelinePhasesNode.innerHTML = phases
      .map((phase) => {
        const phaseKind = phase.kind || "live";
        const phaseLabel = phase.label ? ` title="${phase.label}"` : "";
        const phaseCount = Math.max(1, Number(phase.count || 1));
        return `<div class="phase-seg ${phaseKind}" style="flex:${phaseCount};"${phaseLabel}></div>`;
      })
      .join("");
  };

  const syncSliderToHazard = (resetToDefault) => {
    const hazardCfg = currentHazardCfg();
    const frameCount = currentFrameCount();
    const currentValue = Number(slider.value) || 0;
    const defaultFrameIdx = Number(hazardCfg?.default_frame_idx || 0);
    slider.max = String(Math.max(0, frameCount - 1));
    slider.value = String(
      resetToDefault
        ? Math.max(0, Math.min(frameCount - 1, defaultFrameIdx))
        : Math.max(0, Math.min(frameCount - 1, currentValue))
    );
  };

  const updateTimeline = () => {
    const frames = currentFrames();
    const idx = Math.min(maxFrameIdx(), Math.max(0, Number(slider.value) || 0));
    slider.value = String(idx);
    const pct = (idx / maxFrameIdx()) * 100.0;
    progressNode.style.width = `${pct}%`;
    markerNode.style.left = `${pct}%`;
    frameLabelNode.textContent = frames[idx] || "";

    const ticks = Array.from(root.querySelectorAll(".year-tick"));
    let activeTick = -1;
    ticks.forEach((tick, i) => {
      const startIdx = Number(tick.dataset.frameIndex || "0");
      if (idx >= startIdx) {
        activeTick = i;
      }
    });
    ticks.forEach((tick, i) => tick.classList.toggle("active", i === activeTick));
  };

  const setPlaying = (on) => {
    if (on && !timer) {
      playBtn.classList.add("playing");
      playBtn.setAttribute("aria-label", "Pause timeline");
      timer = setInterval(() => {
        const next = (Number(slider.value) + 1) % currentFrameCount();
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

  const applyHazardMapView = () => {
    if (!map) return;
    const hazardCfg = currentHazardCfg();
    if (!hazardCfg) return;
    map.setOptions({
      minZoom: Number(hazardCfg.zoom_min || 2),
      maxZoom: Number(hazardCfg.zoom_max || 10),
    });
    map.setCenter({
      lat: Number(hazardCfg.center_lat || 36.0),
      lng: Number(hazardCfg.center_lon || -95.0),
    });
    map.setZoom(Number(hazardCfg.default_zoom || 4));
  };

  const installOverlay = () => {
    updateStatus("");
    renderModelSummary();
    updateTimeline();
    if (!map) return;

    const selectedHazard = currentHazardKey();
    if (!hazards[selectedHazard]) {
      clearOverlays();
      updateStatus("Unknown layer selection.", true);
      return;
    }

    const frameIdx = Number(slider.value);
    clearOverlays();
    pushOverlay(selectedHazard, frameIdx);
  };

  const initGoogleMap = () => {
    const hazardCfg = currentHazardCfg() || {};
    map = new google.maps.Map(mapNode, {
      center: {
        lat: Number(hazardCfg.center_lat || 36.0),
        lng: Number(hazardCfg.center_lon || -95.0),
      },
      zoom: Number(hazardCfg.default_zoom || 4),
      minZoom: Number(hazardCfg.zoom_min || 2),
      maxZoom: Number(hazardCfg.zoom_max || 10),
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
  hazardSelect.addEventListener("change", () => {
    setPlaying(false);
    syncSliderToHazard(true);
    renderTimelineScaffold();
    applyHazardMapView();
    installOverlay();
  });
  playBtn.addEventListener("click", () => setPlaying(!timer));

  syncSliderToHazard(true);
  renderTimelineScaffold();
  renderModelSummary();
  updateTimeline();
  loadGoogleMaps();
  return [];
};
