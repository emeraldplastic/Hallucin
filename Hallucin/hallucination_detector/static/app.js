const form = document.getElementById("analyze-form");
const runBtn = document.getElementById("run-btn");
const textMode = document.getElementById("text-mode");
const filesMode = document.getElementById("files-mode");
const modeButtons = [...document.querySelectorAll(".mode-btn")];

const scoreValue = document.getElementById("score-value");
const scoreLabel = document.getElementById("score-label");
const runtime = document.getElementById("runtime");
const claims = document.getElementById("claims");
const supportedChip = document.getElementById("supported-chip");
const partialChip = document.getElementById("partial-chip");
const unsupportedChip = document.getElementById("unsupported-chip");

let currentMode = "text";

for (const button of modeButtons) {
  button.addEventListener("click", () => {
    currentMode = button.dataset.mode;
    modeButtons.forEach((btn) => btn.classList.toggle("active", btn === button));
    textMode.classList.toggle("hidden", currentMode !== "text");
    filesMode.classList.toggle("hidden", currentMode !== "files");
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  runBtn.disabled = true;
  runBtn.textContent = "Analyzing...";
  clearResults();

  try {
    let response;
    if (currentMode === "files") {
      const formData = new FormData(form);
      response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });
    } else {
      const payload = {
        context: document.getElementById("context").value,
        response: document.getElementById("response").value,
      };
      response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    }

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Analysis failed");
    }

    renderResult(data);
  } catch (error) {
    scoreLabel.textContent = error.message;
    scoreValue.textContent = "--";
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Analyze Grounding";
  }
});

function renderResult(data) {
  scoreValue.textContent = Number(data.score).toFixed(2);
  scoreLabel.textContent = classify(data.score);
  runtime.textContent = `Runtime ${Number(data.elapsed_ms).toFixed(1)} ms`;

  supportedChip.textContent = `Supported: ${data.counts.supported}`;
  partialChip.textContent = `Partial: ${data.counts.partial}`;
  unsupportedChip.textContent = `Unsupported: ${data.counts.unsupported}`;

  for (const item of data.claims) {
    const card = document.createElement("article");
    card.className = `claim ${item.label}`;
    card.innerHTML = `
      <div>${escapeHtml(item.claim)}</div>
      <div class="meta">${item.label.toUpperCase()} · similarity ${Number(item.score).toFixed(2)}</div>
      <div class="meta">Best match: ${escapeHtml(item.best_match)}</div>
    `;
    claims.append(card);
  }
}

function classify(score) {
  if (score >= 0.75) return "Strong grounding";
  if (score >= 0.5) return "Mixed grounding";
  return "Low grounding confidence";
}

function clearResults() {
  claims.innerHTML = "";
  scoreLabel.textContent = "Running analysis...";
  scoreValue.textContent = "...";
  runtime.textContent = "Runtime -- ms";
  supportedChip.textContent = "Supported: 0";
  partialChip.textContent = "Partial: 0";
  unsupportedChip.textContent = "Unsupported: 0";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}
