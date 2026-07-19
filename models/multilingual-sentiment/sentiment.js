// Shared front-end helpers for the multilingual-sentiment pages. Keeps each page thin: it owns the
// worker handshake and the renderers (3-class probability bars, word-level occlusion attribution, a
// compact verdict pill). All inference lives in worker.js (off the main thread).

const WORKER_URL = "/web-ai-showcase/models/multilingual-sentiment/worker.js";

export const CLASS_META = {
  negative: { label: "Negative", emoji: "🙁", varName: "--bad" },
  neutral: { label: "Neutral", emoji: "😐", varName: "--warn" },
  positive: { label: "Positive", emoji: "🙂", varName: "--good" },
};
export const CLASS_ORDER = ["negative", "neutral", "positive"];

export class SentimentEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this._loadWaiters = [];
    this._pending = new Map();
    this._id = 0;
    this.onProgress = null;
    this.device = "wasm";
    this.worker.addEventListener("message", (e) => this._onMessage(e.data));
    this.worker.addEventListener("error", (e) => {
      const err = new Error(e.message || "Worker failed to start");
      for (const w of this._loadWaiters) w.reject(err);
      this._loadWaiters = [];
      for (const [, p] of this._pending) p.reject(err);
      this._pending.clear();
    });
  }

  _onMessage(msg) {
    if (msg.type === "progress") {
      this.onProgress?.(msg.p);
    } else if (msg.type === "ready") {
      this.ready = true;
      this.device = msg.device;
      for (const w of this._loadWaiters) w.resolve(msg.device);
      this._loadWaiters = [];
    } else if (msg.type === "result" || msg.type === "attr") {
      const p = this._pending.get(msg.id);
      if (p) {
        this._pending.delete(msg.id);
        p.resolve(msg);
      }
    } else if (msg.type === "error") {
      if (msg.id != null && this._pending.has(msg.id)) {
        this._pending.get(msg.id).reject(new Error(msg.message));
        this._pending.delete(msg.id);
      } else {
        const err = new Error(msg.message);
        for (const w of this._loadWaiters) w.reject(err);
        this._loadWaiters = [];
      }
    }
  }

  load(onProgress) {
    if (onProgress) this.onProgress = onProgress;
    if (this.ready) return Promise.resolve(this.device);
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load" });
    });
  }

  /** → { text, dist:{negative,neutral,positive}, label, ms, device } */
  classify(text) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, text });
    });
  }

  /** → { text, words[], attributions[], dist, label, target, capped, ms, device } */
  attribute(text) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "attribute", id, text });
    });
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

export function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

/** A compact verdict pill (emoji + label + confidence). */
export function renderVerdict(container, dist, label) {
  const meta = CLASS_META[label];
  const conf = dist[label];
  container.className = "senti-verdict is-" + label;
  container.innerHTML = `<span class="senti-emoji" aria-hidden="true">${meta.emoji}</span>` +
    `<span class="senti-vlabel">${meta.label}</span>` +
    `<span class="senti-vconf">${(conf * 100).toFixed(1)}%</span>`;
}

/** Render the full 3-class probability distribution as labelled bars (the "see inside" surface). */
export function renderProbs(container, dist, label) {
  container.replaceChildren(...CLASS_ORDER.map((c) => {
    const meta = CLASS_META[c];
    const pct = dist[c] * 100;
    const row = document.createElement("div");
    row.className = "prob-row" + (c === label ? " is-top" : "");
    row.innerHTML =
      `<span class="prob-name"><span aria-hidden="true">${meta.emoji}</span> ${meta.label}</span>` +
      `<span class="prob-track"><span class="prob-fill" style="inline-size:${pct.toFixed(1)}%;` +
      `background:var(${meta.varName})"></span></span>` +
      `<span class="prob-val">${pct.toFixed(1)}%</span>`;
    return row;
  }));
}

/**
 * Render word-level occlusion attribution: each word tinted by how much removing it moved the winning
 * class (in log-odds). Words that PUSH the verdict are tinted in the class colour; words that OPPOSE it
 * are tinted the opposite way. Intensity encodes magnitude.
 */
export function renderAttribution(container, words, attributions, target) {
  const max = Math.max(1e-6, ...attributions.map((a) => Math.abs(a)));
  const meta = CLASS_META[target];
  container.replaceChildren(...words.map((w, i) => {
    const a = attributions[i] || 0;
    const t = Math.min(1, Math.abs(a) / max);
    const span = document.createElement("span");
    span.className = "attr-word";
    // Positive attribution → the word supports the winning class; colour it in the class hue.
    // Negative → it pulls the other way; use a neutral slate so the contrast reads.
    const hueVar = a >= 0 ? meta.varName : "--muted";
    span.style.background = `color-mix(in srgb, var(${hueVar}) ${
      (t * 55).toFixed(0)
    }%, transparent)`;
    span.textContent = w;
    const sign = a >= 0 ? "+" : "−";
    span.title = `"${w}" ${sign}${
      Math.abs(a).toFixed(2)
    } log-odds toward ${meta.label.toLowerCase()}`;
    return span;
  }));
}

export const SENTI_CSS = `
.senti-verdict { display: inline-flex; align-items: center; gap: .5rem; border: 1px solid var(--border);
  border-radius: 999px; padding: .35rem .9rem; background: var(--bg-raised); font-family: var(--font-display);
  font-size: 1.15rem; }
.senti-verdict.is-negative { border-color: var(--bad); }
.senti-verdict.is-neutral { border-color: var(--warn); }
.senti-verdict.is-positive { border-color: var(--good); }
.senti-emoji { font-size: 1.4rem; line-height: 1; }
.senti-vconf { font-family: var(--font-mono); font-size: .85rem; color: var(--muted); }
.prob-list { display: flex; flex-direction: column; gap: .5rem; margin-top: .6rem; }
.prob-row { display: grid; grid-template-columns: minmax(6.5rem, auto) 1fr auto; gap: .6rem; align-items: center; }
.prob-name { font-size: .9rem; white-space: nowrap; }
.prob-row.is-top .prob-name { font-weight: 600; }
.prob-track { block-size: .7rem; background: var(--bg-secondary); border: 1px solid var(--border);
  border-radius: 999px; overflow: hidden; min-inline-size: 0; }
.prob-fill { display: block; block-size: 100%; border-radius: 999px; transition: inline-size .25s ease; }
.prob-val { font-family: var(--font-mono); font-size: .78rem; color: var(--muted); white-space: nowrap; }
.attr-strip { display: flex; flex-wrap: wrap; gap: 4px; margin-top: .5rem; line-height: 2; }
.attr-word { padding: .1rem .35rem; border-radius: 6px; font-size: .95rem; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.chip { font: inherit; font-size: .78rem; padding: .3rem .7rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer;
  min-block-size: 2.2rem; }
.chip:hover { border-color: var(--accent); }
.chip .chip-lang { font-family: var(--font-mono); color: var(--muted); font-size: .72rem; margin-inline-end: .3rem; }
.senti-input { inline-size: 100%; font: inherit; padding: .5rem .6rem; border: 1px solid var(--border);
  border-radius: 8px; background: var(--bg-raised); color: var(--color); resize: vertical; }
.sample-row { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0; }
`;
