(() => {
  "use strict";

  const form = document.getElementById("analyze-form");
  const runBtn = document.getElementById("run-btn");
  const textMode = document.getElementById("text-mode");
  const filesMode = document.getElementById("files-mode");
  const modeButtons = [...document.querySelectorAll(".mode-tabs__button")];
  const formStatus = document.getElementById("form-status");

  const contextInput = document.getElementById("context");
  const responseInput = document.getElementById("response");
  const contextCount = document.getElementById("context-count");
  const responseCount = document.getElementById("response-count");

  const scoreValue = document.getElementById("score-value");
  const scoreLabel = document.getElementById("score-label");
  const runtime = document.getElementById("runtime");
  const claims = document.getElementById("claims");
  const gaugeFill = document.getElementById("gauge-fill");

  const statSupported = document.getElementById("stat-supported");
  const statPartial = document.getElementById("stat-partial");
  const statUnsupported = document.getElementById("stat-unsupported");

  const maxTextChars = Number(form.dataset.maxTextChars || 0);
  const circumference = 2 * Math.PI * 52;
  let currentMode = "text";

  gaugeFill.style.strokeDasharray = String(circumference);
  setGauge(0);
  updateCounts();

  for (const input of [contextInput, responseInput]) {
    input.addEventListener("input", updateCounts);
  }

  for (const button of modeButtons) {
    button.addEventListener("click", () => {
      currentMode = button.dataset.mode;
      for (const item of modeButtons) {
        const isActive = item === button;
        item.classList.toggle("active", isActive);
        item.setAttribute("aria-selected", String(isActive));
      }
      textMode.classList.toggle("hidden", currentMode !== "text");
      filesMode.classList.toggle("hidden", currentMode !== "files");
      formStatus.textContent = "";
    });
  }

  document.querySelectorAll(".dropzone").forEach((zone) => {
    const input = zone.querySelector("input[type=file]");
    const hint = zone.querySelector(".dropzone__hint");

    input.addEventListener("change", () => {
      const file = input.files && input.files[0];
      hint.textContent = file ? file.name : "Drop or choose a file";
      zone.classList.toggle("has-file", Boolean(file));
    });

    zone.addEventListener("dragover", (event) => {
      event.preventDefault();
      zone.classList.add("dragover");
    });
    zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
    zone.addEventListener("drop", (event) => {
      event.preventDefault();
      zone.classList.remove("dragover");
      if (event.dataTransfer.files.length) {
        input.files = event.dataTransfer.files;
        input.dispatchEvent(new Event("change"));
      }
    });
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    formStatus.textContent = "";

    const validationError = validateInputs();
    if (validationError) {
      showError(validationError);
      return;
    }

    runBtn.disabled = true;
    runBtn.classList.add("loading");
    clearResults();

    try {
      const response = await fetch("/api/analyze", buildRequest());
      const data = await parseJsonResponse(response);

      if (!response.ok) {
        throw new Error(data.error || `Analysis failed (${response.status})`);
      }

      renderResult(data);
    } catch (error) {
      showError(error.message || "Analysis failed.");
    } finally {
      runBtn.disabled = false;
      runBtn.classList.remove("loading");
    }
  });

  function buildRequest() {
    if (currentMode === "files") {
      const formData = new FormData(form);
      formData.append("model_name", "local");
      return { method: "POST", body: formData };
    }

    return {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        context: contextInput.value,
        response: responseInput.value,
        model_name: "local",
      }),
    };
  }

  async function parseJsonResponse(response) {
    const raw = await response.text();
    try {
      return JSON.parse(raw);
    } catch {
      const snippet = raw.slice(0, 140).replace(/\s+/g, " ").trim();
      throw new Error(`Server returned non-JSON (${response.status}). ${snippet}`);
    }
  }

  function validateInputs() {
    if (currentMode === "text") {
      if (!contextInput.value.trim() || !responseInput.value.trim()) {
        return "Source context and model response are required.";
      }
      if (maxTextChars && contextInput.value.length > maxTextChars) {
        return `Source context exceeds ${maxTextChars} characters.`;
      }
      if (maxTextChars && responseInput.value.length > maxTextChars) {
        return `Model response exceeds ${maxTextChars} characters.`;
      }
      return "";
    }

    const contextFile = document.getElementById("context_file").files[0];
    const responseFile = document.getElementById("response_file").files[0];
    if (!contextFile || !responseFile) {
      return "Choose both a context file and a response file.";
    }
    return "";
  }

  function renderResult(data) {
    const score = Number(data.score || 0);

    animateCounter(scoreValue, score, 2);
    scoreLabel.textContent = classify(score);
    runtime.textContent = `${Number(data.elapsed_ms || 0).toFixed(1)} ms`;
    setGauge(score);

    animateCounter(statSupported, data.counts.supported || 0, 0);
    animateCounter(statPartial, data.counts.partial || 0, 0);
    animateCounter(statUnsupported, data.counts.unsupported || 0, 0);

    claims.innerHTML = "";
    if (!Array.isArray(data.claims) || data.claims.length === 0) {
      claims.append(emptyState("No claims were detected in the response."));
      return;
    }

    data.claims.forEach((item, index) => {
      const card = document.createElement("article");
      card.className = `claim ${item.label}`;
      card.style.animationDelay = `${index * 35}ms`;

      const text = document.createElement("p");
      text.className = "claim__text";
      text.textContent = item.claim;

      const meta = document.createElement("div");
      meta.className = "claim__meta";

      const label = document.createElement("span");
      label.className = "claim__badge";
      label.textContent = item.label;

      const scoreItem = document.createElement("span");
      scoreItem.textContent = `similarity ${Number(item.score || 0).toFixed(2)}`;

      const best = document.createElement("span");
      best.className = "claim__match";
      best.textContent = `best match: ${truncate(item.best_match || "", 110)}`;

      meta.append(label, scoreItem, best);
      card.append(text, meta);
      claims.append(card);
    });
  }

  function showError(message) {
    formStatus.textContent = message;
    claims.innerHTML = "";
    claims.append(emptyState(message));
    scoreLabel.textContent = "Needs attention";
    scoreValue.textContent = "--";
    runtime.textContent = "Not run";
    statSupported.textContent = "0";
    statPartial.textContent = "0";
    statUnsupported.textContent = "0";
    setGauge(0);
  }

  function clearResults() {
    claims.innerHTML = "";
    claims.append(emptyState("Analyzing claims..."));
    scoreLabel.textContent = "Analyzing";
    scoreValue.textContent = "...";
    runtime.textContent = "Running";
    statSupported.textContent = "0";
    statPartial.textContent = "0";
    statUnsupported.textContent = "0";
    setGauge(0);
  }

  function emptyState(text) {
    const element = document.createElement("div");
    element.className = "empty-result";
    element.textContent = text;
    return element;
  }

  function setGauge(value) {
    const bounded = Math.min(1, Math.max(0, Number(value) || 0));
    gaugeFill.style.strokeDashoffset = String(circumference * (1 - bounded));
  }

  function animateCounter(element, target, decimals) {
    const duration = 520;
    const start = performance.now();
    const from = Number(element.textContent) || 0;

    function tick(now) {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const value = from + (Number(target) - from) * eased;
      element.textContent = value.toFixed(decimals);
      if (progress < 1) requestAnimationFrame(tick);
    }

    requestAnimationFrame(tick);
  }

  function classify(score) {
    if (score >= 0.75) return "Strong grounding";
    if (score >= 0.5) return "Mixed grounding";
    return "Low grounding";
  }

  function updateCounts() {
    contextCount.textContent = String(contextInput.value.length);
    responseCount.textContent = String(responseInput.value.length);
    contextCount.classList.toggle("over-limit", maxTextChars && contextInput.value.length > maxTextChars);
    responseCount.classList.toggle("over-limit", maxTextChars && responseInput.value.length > maxTextChars);
  }

  function truncate(value, max) {
    const text = String(value);
    return text.length <= max ? text : `${text.slice(0, max)}...`;
  }
})();
