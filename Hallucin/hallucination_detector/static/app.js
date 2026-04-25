/* ─────────────────────────────────────────────
   Hallucin Studio — App Logic
   ───────────────────────────────────────────── */

(() => {
  "use strict";

  // ── DOM handles ──
  const form         = document.getElementById("analyze-form");
  const runBtn       = document.getElementById("run-btn");
  const textMode     = document.getElementById("text-mode");
  const filesMode    = document.getElementById("files-mode");
  const modeToggle   = document.getElementById("mode-toggle");
  const modeButtons  = [...document.querySelectorAll(".mode-toggle__btn")];

  const scoreValue   = document.getElementById("score-value");
  const scoreLabel   = document.getElementById("score-label");
  const runtime      = document.getElementById("runtime");
  const claims       = document.getElementById("claims");
  const gaugeFill    = document.getElementById("gauge-fill");

  const statSupported   = document.getElementById("stat-supported");
  const statPartial     = document.getElementById("stat-partial");
  const statUnsupported = document.getElementById("stat-unsupported");

  let currentMode = "text";

  // ── Inject SVG gradient for gauge (needs to live inside <svg>) ──
  const gaugeRingSvg = document.querySelector(".gauge__ring");
  if (gaugeRingSvg) {
    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    defs.innerHTML = `
      <linearGradient id="gauge-gradient" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#2dd4bf"/>
        <stop offset="100%" stop-color="#0284c7"/>
      </linearGradient>
    `;
    gaugeRingSvg.prepend(defs);
  }

  // ── Mode toggle ──
  for (const button of modeButtons) {
    button.addEventListener("click", () => {
      currentMode = button.dataset.mode;
      modeButtons.forEach(btn => btn.classList.toggle("active", btn === button));
      modeToggle.dataset.active = currentMode;
      textMode.classList.toggle("hidden", currentMode !== "text");
      filesMode.classList.toggle("hidden", currentMode !== "files");
    });
  }

  // ── File dropzone interaction ──
  document.querySelectorAll(".dropzone").forEach(zone => {
    const input = zone.querySelector("input[type=file]");
    const hint  = zone.querySelector(".dropzone__hint");

    if (input) {
      input.addEventListener("change", () => {
        if (input.files.length) {
          hint.textContent = input.files[0].name;
          zone.classList.add("has-file");
        } else {
          hint.textContent = "Drop or click to upload";
          zone.classList.remove("has-file");
        }
      });
    }

    zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("dragover"); });
    zone.addEventListener("dragleave", ()=> { zone.classList.remove("dragover"); });
    zone.addEventListener("drop", e => {
      e.preventDefault();
      zone.classList.remove("dragover");
      if (input && e.dataTransfer.files.length) {
        input.files = e.dataTransfer.files;
        input.dispatchEvent(new Event("change"));
      }
    });
  });

  // ── Form submit ──
  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    runBtn.disabled = true;
    runBtn.classList.add("loading");
    clearResults();

    try {
      let resp;
      if (currentMode === "files") {
        const formData = new FormData(form);
        formData.append("model_name", "local");
        resp = await fetch("/api/analyze", {
          method: "POST",
          body: formData,
        });
      } else {
        const payload = {
          context: document.getElementById("context").value,
          response: document.getElementById("response").value,
          model_name: "local",
        };
        resp = await fetch("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
      }

      const raw = await resp.text();
      let data = null;
      try {
        data = JSON.parse(raw);
      } catch {
        const snippet = raw.slice(0, 140).replace(/\s+/g, " ").trim();
        throw new Error(`Server returned non-JSON (${resp.status}). ${snippet}`);
      }

      if (!resp.ok) {
        throw new Error(data.error || `Analysis failed (${resp.status})`);
      }

      renderResult(data);
    } catch (error) {
      scoreLabel.textContent = error.message;
      scoreValue.textContent = "--";
      setGauge(0);
    } finally {
      runBtn.disabled = false;
      runBtn.classList.remove("loading");
    }
  });

  // ── Render result ──
  function renderResult(data) {
    const score = Number(data.score);

    // animate score counter
    animateCounter(scoreValue, score, 2);
    scoreLabel.textContent = classify(score);
    runtime.textContent = `${Number(data.elapsed_ms).toFixed(1)} ms`;

    setGauge(score);

    // stat counters
    animateCounter(statSupported,   data.counts.supported, 0);
    animateCounter(statPartial,     data.counts.partial, 0);
    animateCounter(statUnsupported, data.counts.unsupported, 0);

    // render claims with stagger
    data.claims.forEach((item, i) => {
      const card = document.createElement("article");
      card.className = `claim ${item.label}`;
      card.style.animationDelay = `${i * 60}ms`;
      card.innerHTML = `
        <div class="claim__text">${escapeHtml(item.claim)}</div>
        <div class="claim__meta">
          <span class="claim__badge">${item.label}</span>
          <span>sim ${Number(item.score).toFixed(2)}</span>
          <span>· best: ${escapeHtml(truncate(item.best_match, 80))}</span>
        </div>
      `;
      claims.append(card);
    });
  }

  // ── Score gauge ──
  const CIRCUMFERENCE = 2 * Math.PI * 62; // r=62

  function setGauge(pct /* 0-1 */) {
    const offset = CIRCUMFERENCE * (1 - Math.min(1, Math.max(0, pct)));
    gaugeFill.style.strokeDashoffset = offset;
  }

  // ── Animated counter ──
  function animateCounter(el, target, decimals) {
    const duration = 800;
    const start    = performance.now();
    const from     = 0;

    function tick(now) {
      const t = Math.min((now - start) / duration, 1);
      const ease = 1 - Math.pow(1 - t, 3);
      const val = from + (target - from) * ease;
      el.textContent = val.toFixed(decimals);
      if (t < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  // ── Classify score ──
  function classify(score) {
    if (score >= 0.75) return "Strong grounding";
    if (score >= 0.5)  return "Mixed grounding";
    return "Low grounding";
  }

  // ── Clear results ──
  function clearResults() {
    claims.innerHTML = "";
    scoreLabel.textContent = "Analyzing…";
    scoreValue.textContent = "···";
    runtime.textContent = "— ms";
    statSupported.textContent   = "0";
    statPartial.textContent     = "0";
    statUnsupported.textContent = "0";
    setGauge(0);
  }

  // ── Helpers ──
  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  function truncate(str, max) {
    if (str.length <= max) return str;
    return str.slice(0, max) + "…";
  }

})();
