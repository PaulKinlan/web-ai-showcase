// Front-end helpers for the formality-detection pages. Keeps each page thin: it owns the worker
// handshake and the shared renderers (formality meter, register-cue highlights). All inference lives in
// worker.js (off the main thread).

const WORKER_URL = "/web-ai-showcase/models/formality-detection/worker.js";

export class FormalityEngine {
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

  /** Classify one text → { formal, informal, label, ms, device }. */
  classify(text) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, text });
    });
  }

  /** Occlusion attribution → { words, attributions, formal, label, ms, device }. */
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

/** Render a formal/informal verdict + a formality meter (informal ← → formal) into `els`. */
export function renderVerdict(els, { formal, label }) {
  const pct = (formal * 100).toFixed(1);
  els.label.textContent = label === "formal" ? "Formal" : "Informal";
  els.label.className = "verdict-label " + (label === "formal" ? "formal" : "informal");
  els.conf.textContent = pct + "% formal";
  els.needle.style.insetInlineStart = pct + "%";
}

/**
 * Render occlusion attributions as coloured word chips. Blue = pushed FORMAL, amber = pushed INFORMAL,
 * brightness ∝ magnitude. `words`/`attributions` are aligned arrays from engine.attribute().
 */
export function renderAttribution(container, words, attributions) {
  const maxAbs = Math.max(1e-6, ...attributions.map(Math.abs));
  container.replaceChildren(...words.map((w, i) => {
    const a = attributions[i];
    const t = Math.abs(a) / maxAbs;
    const hue = a >= 0 ? "var(--formal)" : "var(--informal)";
    const span = document.createElement("span");
    span.className = "attr-word";
    span.textContent = w;
    span.style.background = `color-mix(in srgb, ${hue} ${(t * 62).toFixed(0)}%, transparent)`;
    span.title = `${a >= 0 ? "+" : ""}${a.toFixed(2)} log-odds toward formal`;
    return span;
  }));
}

export const FORMALITY_CSS = `
/* The at-a-glance model id (Deepchecks/roberta_base_formality_ranker_onnx) is a long unbreakable token;
   let it wrap so the table never forces horizontal overflow on a 360px viewport. */
.inside-table td, .inside-table th { overflow-wrap: anywhere; }
:root { --formal: #3a6ea5; --informal: #c07a1a; }
@media (prefers-color-scheme: dark) { :root { --formal: #6ea0d8; --informal: #d9a441; } }
.verdict-row { display: flex; align-items: baseline; gap: .8rem; flex-wrap: wrap; margin-top: .6rem; }
.verdict-label { font-family: var(--font-display); font-size: 1.8rem; }
.verdict-label.formal { color: var(--formal); }
.verdict-label.informal { color: var(--informal); }
.verdict-conf { font-family: var(--font-mono); color: var(--muted); font-size: .9rem; }
.fmeter { position: relative; block-size: .85rem; border: 1px solid var(--border); border-radius: 999px;
  overflow: hidden; margin-top: .5rem; max-inline-size: 560px;
  background: linear-gradient(90deg, var(--informal), var(--bg-secondary) 50%, var(--formal)); }
.fneedle { position: absolute; inset-block: -3px; inline-size: 3px; background: var(--color);
  border-radius: 2px; transition: inset-inline-start .25s ease; }
.fmeter-labels { display: flex; justify-content: space-between; max-inline-size: 560px;
  font-family: var(--font-mono); font-size: .72rem; color: var(--muted); margin-top: .2rem; }
.attr-wrap { line-height: 2.1; margin-top: .5rem; }
.attr-word { padding: .12rem .28rem; border-radius: 5px; margin: 0 1px; white-space: pre-wrap; }
.attr-legend { display: flex; flex-wrap: wrap; gap: 1rem; font-size: .78rem; color: var(--muted);
  font-family: var(--font-mono); margin-top: .6rem; }
.attr-legend .swatch { display: inline-block; inline-size: .8rem; block-size: .8rem; border-radius: 3px;
  margin-inline-end: .3rem; vertical-align: -1px; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.chip { font: inherit; font-size: .78rem; padding: .3rem .7rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; min-block-size: 2.2rem; }
.chip:hover { border-color: var(--accent); }
.chip-row { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0; }
.sent-list { display: flex; flex-direction: column; gap: .5rem; margin-top: .5rem; }
.sent-row { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised);
  padding: .5rem .7rem; display: flex; justify-content: space-between; gap: .6rem; align-items: center; flex-wrap: wrap; }
.sent-row.formal { border-inline-start: 4px solid var(--formal); }
.sent-row.informal { border-inline-start: 4px solid var(--informal); }
.sent-text { flex: 1 1 14rem; min-inline-size: 0; }
.sent-meta { font-family: var(--font-mono); font-size: .74rem; color: var(--muted); white-space: nowrap; }
.badge { font-family: var(--font-mono); font-size: .68rem; padding: .1rem .45rem; border-radius: 999px; border: 1px solid var(--border); }
.badge.formal { color: var(--formal); border-color: var(--formal); }
.badge.informal { color: var(--informal); border-color: var(--informal); }
.pair-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 16rem), 1fr)); gap: 1rem; margin-top: .6rem; }
.pair { border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-raised); padding: .7rem; }
.pair h4 { margin: 0 0 .4rem; font-size: .85rem; }
`;
