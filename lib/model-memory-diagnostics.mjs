// Visible, honest memory snapshots for local-model demos.
// measureUserAgentSpecificMemory() estimates this page and its dedicated workers, not model-only
// allocation or GPU/process memory. It can wait minutes for GC; never let it wedge model controls.

export function formatMemoryBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return "unavailable";
  const mib = bytes / (1024 * 1024);
  return `${mib >= 100 ? mib.toFixed(0) : mib.toFixed(1)} MiB`;
}

export function memorySupport() {
  if (!globalThis.crossOriginIsolated) {
    return { supported: false, reason: "This page is not cross-origin isolated." };
  }
  if (typeof globalThis.performance?.measureUserAgentSpecificMemory !== "function") {
    return {
      supported: false,
      reason: "This Chrome version or device does not expose measureUserAgentSpecificMemory().",
    };
  }
  return { supported: true, reason: "" };
}

export async function measureOriginMemory() {
  const support = memorySupport();
  if (!support.supported) throw new DOMException(support.reason, "NotSupportedError");
  const result = await performance.measureUserAgentSpecificMemory();
  if (!Number.isFinite(result.bytes) || result.bytes < 0) {
    throw new Error("The browser returned no usable memory estimate.");
  }
  return {
    bytes: result.bytes,
    breakdownEntries: Array.isArray(result.breakdown) ? result.breakdown.length : 0,
    types: [...new Set((result.breakdown || []).flatMap((entry) => entry.types || []))],
    measuredAt: new Date().toISOString(),
  };
}

const TEMPLATE = `
  <style>
    :host { display:block; margin-top:.75rem; color:var(--color, CanvasText); }
    details { border-top:1px solid var(--border, #8885); padding-top:.55rem; }
    summary { cursor:pointer; font-size:.875rem; min-block-size:44px; align-content:center; color:var(--muted, #666); }
    summary strong { color:var(--color, CanvasText); font-weight:600; }
    .body { padding:.55rem 0 .1rem; font-size:.875rem; overflow-wrap:anywhere; }
    p { margin:.25rem 0; }
    ol { margin:.45rem 0; padding-inline-start:1.3rem; }
    li { margin:.2rem 0; }
    button { font:inherit; min-block-size:44px; padding:.3rem .65rem; border:1px solid var(--border-strong, #555);
      border-radius:7px; background:var(--bg-raised, Canvas); color:var(--color, CanvasText); cursor:pointer; }
    button:disabled { opacity:.55; cursor:default; }
    .muted { color:var(--muted, #666); }
    :is(summary, button):focus-visible { outline:2px solid var(--accent, Highlight); outline-offset:3px; }
  </style>
  <details>
    <summary>Memory diagnostics: <strong id="headline">checking support…</strong></summary>
    <div class="body">
      <p id="explanation" class="muted"></p>
      <ol id="snapshots"></ol>
      <button id="measure" type="button">Measure memory now</button>
      <p id="status" role="status" aria-live="polite"></p>
    </div>
  </details>`;

const HTMLElementBase = globalThis.HTMLElement ?? class {};

export class ModelMemoryDiagnostics extends HTMLElementBase {
  constructor() {
    super();
    this.attachShadow({ mode: "open" }).innerHTML = TEMPLATE;
    this.snapshots = [];
    this.measuring = null;
  }

  connectedCallback() {
    if (this._connected) return;
    this._connected = true;
    this.$ = (selector) => this.shadowRoot.querySelector(selector);
    this.$("#measure").addEventListener(
      "click",
      // An explicit request is not a phase label, so a slow answer is still the answer: the measured
      // native floor is ~19s idle / ~60s with a WASM model resident (web-ai-showcase-cgr), so race it
      // against a bound that can actually be met and RETAIN the result when it lands.
      () => this.capture("Manual snapshot", { timeoutMs: 90000, retainLate: true }),
    );
    const support = memorySupport();
    this.$("#measure").disabled = !support.supported;
    this.$("#headline").textContent = support.supported ? "ready to measure" : "unavailable";
    this.$("#explanation").textContent = support.supported
      ? "Browser estimate for this page and its dedicated workers, not model-only allocation, GPU memory, or disk cache. Deltas compare observations; garbage collection and other page activity affect them. The browser answers only after a garbage-collection pass — measured at roughly 20 seconds with nothing loaded and a minute with a model resident — so Measure memory can take that long."
      : `${support.reason} Use the cross-origin-isolated Deno deployment in a supported Chromium browser.`;
  }

  async capture(label = "Snapshot", { timeoutMs = 1000, retainLate = false } = {}) {
    const support = memorySupport();
    if (!support.supported || this.measuring) return null;
    label = String(label);
    let expired = false, timer;
    const startedAt = performance.now();
    this.$("#measure").disabled = true;
    if (!this.snapshots.length) this.$("#headline").textContent = "measuring…";
    this.$("#status").textContent = retainLate
      ? `Measuring ${label.toLowerCase()}… the browser waits for a garbage-collection pass before it answers, which can take up to a minute. Model controls are unaffected.`
      : `Measuring ${label.toLowerCase()}…`;
    this.measuring = measureOriginMemory().then((measurement) => {
      // A late measurement cannot truthfully be labelled "before load" or "before release" — that is
      // why lifecycle captures discard it. An explicit request (retainLate) carries a user-supplied
      // label, so the slow answer is kept rather than thrown away.
      if (!this.isConnected) return null;
      if (expired && !retainLate) return null;
      const previous = this.snapshots.at(-1);
      const snapshot = {
        ...measurement,
        label,
        measurementMs: Math.round(performance.now() - startedAt),
        delta: previous ? measurement.bytes - previous.bytes : null,
      };
      this.snapshots.push(snapshot);
      if (this.snapshots.length > 8) this.snapshots.shift();
      this.renderSnapshots();
      this.$("#status").textContent = `${label} measured.`;
      return snapshot;
    }).catch((error) => {
      if (!expired && this.isConnected) {
        if (!this.snapshots.length) this.$("#headline").textContent = "unavailable";
        this.$("#status").textContent = `Memory measurement unavailable: ${error.message}`;
      }
      return null;
    }).finally(() => {
      this.measuring = null;
      this.$("#measure").disabled = false;
      if (expired && !retainLate && this.isConnected) {
        if (!this.snapshots.length) this.$("#headline").textContent = "ready to retry";
        this.$("#status").textContent =
          "The late memory request finished without a retained estimate. Try Measure memory now.";
      }
    });
    try {
      return await Promise.race([
        this.measuring,
        new Promise((resolve) => {
          timer = setTimeout(() => {
            expired = true;
            if (!this.snapshots.length) this.$("#headline").textContent = "measurement delayed";
            this.$("#status").textContent =
              retainLate
                ? "Still measuring — the browser waits for garbage collection before it answers, which can take up to a minute. The result will appear here; model controls are unaffected."
                : "Memory measurement took too long; model controls can continue. Retry when the browser finishes this request.";
            resolve(null);
          }, timeoutMs);
        }),
      ]);
    } finally {
      clearTimeout(timer);
      // The native API cannot be cancelled. Keep its single-flight lock until it settles instead
      // of queuing more GC requests or recording a later estimate under an earlier phase label.
    }
  }

  renderSnapshots() {
    const latest = this.snapshots.at(-1);
    this.$("#headline").textContent = latest
      ? `last ${formatMemoryBytes(latest.bytes)} page + workers`
      : "ready to measure";
    const list = this.$("#snapshots");
    list.replaceChildren();
    for (const snapshot of this.snapshots) {
      const item = document.createElement("li");
      const delta = snapshot.delta == null
        ? "baseline"
        : `${snapshot.delta >= 0 ? "+" : "−"}${formatMemoryBytes(Math.abs(snapshot.delta))}`;
      item.textContent = `${snapshot.label}: ${formatMemoryBytes(snapshot.bytes)} (${delta})`;
      item.title = `${snapshot.measuredAt} · sample took ${snapshot.measurementMs} ms`;
      list.append(item);
    }
  }
}

if (globalThis.customElements && !customElements.get("model-memory-diagnostics")) {
  customElements.define("model-memory-diagnostics", ModelMemoryDiagnostics);
}
