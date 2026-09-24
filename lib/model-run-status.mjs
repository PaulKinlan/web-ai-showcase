// Honest inference activity UI. Runtimes often expose no percent-complete signal, so this component
// shows real phases, elapsed time, heartbeat, and token throughput without inventing a percentage.

export function nextPaint() {
  return new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
}

export function elapsedText(ms) {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${String(seconds % 60).padStart(2, "0")}s`;
}

// Count generated token IDs, not decoded text callbacks (which may contain several tokens).
// The denominator includes prefill; this is not decode-only throughput.
export function tokenMetrics(tokens, ms) {
  if (!Number.isSafeInteger(tokens) || tokens < 0 || !Number.isFinite(ms) || ms <= 0) {
    return "Token rate unavailable";
  }
  return `${tokens} token${tokens === 1 ? "" : "s"} · ${
    (tokens * 1000 / ms).toFixed(1)
  } tok/s average (including prefill)`;
}

const TEMPLATE = `
  <style>
    :host { display:block; margin:.55rem 0; }
    :host([hidden]) { display:none; }
    .box { border:1px solid var(--border, #8885); border-radius:8px; padding:.65rem .75rem;
      background:var(--bg-secondary, Canvas); }
    .top { display:flex; flex-wrap:wrap; justify-content:space-between; gap:.35rem 1rem; align-items:baseline; }
    strong { font-size:.88rem; }
    .elapsed, .metrics { color:var(--muted, #666); font: .75rem/1.4 var(--font-mono, monospace); }
    progress { display:block; width:100%; height:.45rem; margin:.5rem 0; accent-color:var(--accent, Highlight); }
    p { margin:.2rem 0 0; color:var(--muted, #666); font-size:.78rem; }
    .sr-live { position:absolute; inline-size:1px; block-size:1px; padding:0; margin:-1px;
      overflow:hidden; clip-path:inset(50%); white-space:nowrap; border:0; }
    :host([data-state="complete"]) progress { accent-color:var(--good, green); }
    :host([data-state="error"]) .box { border-color:var(--bad, red); }
    @media (prefers-reduced-motion: reduce) { progress:indeterminate { opacity:.75; } }
  </style>
  <div class="box">
    <div class="top"><strong id="phase">Preparing input…</strong><span id="elapsed" class="elapsed">0s</span></div>
    <progress id="progress" aria-label="Model inference in progress"></progress>
    <p id="detail">The model is running locally on this device.</p>
    <p id="metrics" class="metrics"></p>
    <p id="live" role="status" aria-live="polite" class="sr-live"></p>
  </div>`;

const HTMLElementBase = globalThis.HTMLElement ?? class {};

export class ModelRunStatus extends HTMLElementBase {
  constructor() {
    super();
    this.attachShadow({ mode: "open" }).innerHTML = TEMPLATE;
    this._timer = null;
    this._startedAt = 0;
    this._tokens = null;
    this._tokenMs = 0;
    this._lastHeartbeat = -1;
  }

  connectedCallback() {
    this.$ = (selector) => this.shadowRoot.querySelector(selector);
    // Custom-element constructors must not add attributes (document.createElement would fail).
    if (!this.dataset.state) this.hidden = true;
  }

  async start(
    { phase = "Preparing input…", detail = "The model is running locally on this device." } = {},
  ) {
    this.stopTimer();
    this.hidden = false;
    this.dataset.state = "running";
    this._startedAt = performance.now();
    this._tokens = null;
    this._tokenMs = 0;
    this._lastHeartbeat = -1;
    this.$("#progress").removeAttribute("value");
    this.$("#progress").setAttribute("aria-label", phase);
    this.$("#phase").textContent = phase;
    this.$("#detail").textContent = detail;
    this.$("#elapsed").textContent = "0s";
    this.$("#metrics").textContent = "";
    this.$("#live").textContent = phase;
    this._timer = setInterval(() => this.tick(), 250);
    await nextPaint();
  }

  phase(label, detail = "") {
    if (this.hidden) return;
    this.$("#phase").textContent = label;
    this.$("#progress").setAttribute("aria-label", label);
    if (detail) this.$("#detail").textContent = detail;
    this.$("#live").textContent = label;
  }

  token(count = (this._tokens ?? 0) + 1, ms = performance.now() - this._startedAt) {
    if (this.hidden || this.dataset.state !== "running") return;
    this._tokens = count;
    this._tokenMs = ms;
    // tick() paints at most four times per second; token callbacks need not trigger a DOM write.
  }

  complete(
    { label = "Complete", tokens = this._tokens, ms = performance.now() - this._startedAt } = {},
  ) {
    this.stopTimer();
    this.dataset.state = "complete";
    this.$("#phase").textContent = label;
    this.$("#elapsed").textContent = elapsedText(ms);
    this.$("#progress").value = 1;
    this.$("#progress").max = 1;
    this.$("#progress").setAttribute("aria-label", label);
    this.$("#detail").textContent = "Output is ready.";
    const metrics = tokens === null ? "" : tokenMetrics(tokens, ms);
    this.$("#metrics").textContent = metrics;
    this.$("#live").textContent = metrics ? `${label}. ${metrics}.` : label;
  }

  fail(message) {
    this.stopTimer();
    this.dataset.state = "error";
    this.$("#phase").textContent = "Inference failed";
    this.$("#progress").removeAttribute("value");
    this.$("#detail").textContent = message;
    this.$("#live").textContent = `Inference failed: ${message}`;
  }

  tick() {
    const elapsed = performance.now() - this._startedAt;
    this.$("#elapsed").textContent = elapsedText(elapsed);
    if (this._tokens !== null) {
      this.$("#metrics").textContent = tokenMetrics(this._tokens, this._tokenMs);
    }
    const heartbeat = Math.floor(elapsed / 10_000);
    if (heartbeat > 0 && heartbeat !== this._lastHeartbeat) {
      this._lastHeartbeat = heartbeat;
      this.$("#live").textContent = `Still working locally — ${elapsedText(elapsed)} elapsed.`;
    }
  }

  stopTimer() {
    if (this._timer) clearInterval(this._timer);
    this._timer = null;
  }

  disconnectedCallback() {
    this.stopTimer();
  }
}

if (globalThis.customElements && !customElements.get("model-run-status")) {
  customElements.define("model-run-status", ModelRunStatus);
}
