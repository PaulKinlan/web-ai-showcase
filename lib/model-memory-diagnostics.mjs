// Visible, honest memory snapshots for local-model demos.
// measureUserAgentSpecificMemory() is origin-wide and limited-availability: never label its bytes as
// model allocation. Snapshots and deltas are useful when taken before/after model lifecycle phases.

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
    summary { cursor:pointer; font-size:.82rem; color:var(--muted, #666); }
    summary strong { color:var(--color, CanvasText); font-weight:600; }
    .body { padding:.55rem 0 .1rem; font-size:.78rem; }
    p { margin:.25rem 0; }
    ol { margin:.45rem 0; padding-inline-start:1.3rem; }
    li { margin:.2rem 0; }
    button { font:inherit; padding:.3rem .65rem; border:1px solid var(--border-strong, #555);
      border-radius:7px; background:var(--bg-raised, Canvas); color:var(--color, CanvasText); cursor:pointer; }
    button:disabled { opacity:.55; cursor:default; }
    .muted { color:var(--muted, #666); }
    .delta.up { color:var(--warn, #996b00); }
    .delta.down { color:var(--good, #187a35); }
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
    this.$("#measure").addEventListener("click", () => this.capture("Manual snapshot"));
    const support = memorySupport();
    this.$("#measure").disabled = !support.supported;
    this.$("#headline").textContent = support.supported ? "ready to measure" : "unavailable";
    this.$("#explanation").textContent = support.supported
      ? "Chrome's origin-wide estimate includes this page, workers, model data, and runtime overhead. Deltas are labelled observations, not model-only allocation."
      : `${support.reason} Use the cross-origin-isolated Deno deployment in a supported Chromium browser.`;
  }

  async capture(label = "Snapshot") {
    const support = memorySupport();
    if (!support.supported) return null;
    if (this.measuring) {
      await this.measuring;
      return this.capture(label);
    }
    this.$("#measure").disabled = true;
    this.$("#status").textContent = `Measuring ${label.toLowerCase()}…`;
    this.measuring = measureOriginMemory().then((measurement) => {
      const previous = this.snapshots.at(-1);
      const snapshot = {
        ...measurement,
        label,
        delta: previous ? measurement.bytes - previous.bytes : null,
      };
      this.snapshots.push(snapshot);
      if (this.snapshots.length > 8) this.snapshots.shift();
      this.renderSnapshots();
      this.$("#status").textContent = `${label} measured.`;
      return snapshot;
    }).catch((error) => {
      this.$("#status").textContent = `Memory measurement unavailable: ${error.message}`;
      return null;
    }).finally(() => {
      this.measuring = null;
      this.$("#measure").disabled = false;
    });
    return this.measuring;
  }

  renderSnapshots() {
    const latest = this.snapshots.at(-1);
    this.$("#headline").textContent = latest
      ? `${formatMemoryBytes(latest.bytes)} origin-wide`
      : "ready to measure";
    const list = this.$("#snapshots");
    list.replaceChildren();
    for (const snapshot of this.snapshots) {
      const item = document.createElement("li");
      const delta = snapshot.delta == null
        ? "baseline"
        : `${snapshot.delta >= 0 ? "+" : "−"}${formatMemoryBytes(Math.abs(snapshot.delta))}`;
      item.textContent = `${snapshot.label}: ${formatMemoryBytes(snapshot.bytes)} (${delta})`;
      if (snapshot.delta !== null) {
        const span = document.createElement("span");
        span.className = `delta ${snapshot.delta >= 0 ? "up" : "down"}`;
        span.textContent = "";
        item.append(span);
      }
      list.append(item);
    }
  }
}

if (globalThis.customElements && !customElements.get("model-memory-diagnostics")) {
  customElements.define("model-memory-diagnostics", ModelMemoryDiagnostics);
}
