// Shared front-end helpers for the DistilBERT sentiment pages. Keeps each page thin: it owns the worker
// handshake and the highlight renderer. All inference (classification + occlusion) lives in worker.js.

const WORKER_URL = "/web-ai-showcase/models/sentiment-classifier/worker.js";

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

  /** Classify one text → { text, pos, neg, label, ms, device }. */
  classify(text) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, text });
    });
  }

  /** Occlusion attribution → { text, words, attributions, pos, label, ms, device }. */
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

/** Simple debounce for "as you type" surfaces. */
export function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

/** Render a POSITIVE/NEGATIVE verdict + confidence meter into `els`. */
export function renderVerdict(els, { pos, neg, label }) {
  const conf = Math.max(pos, neg);
  els.label.textContent = label;
  els.label.className = "verdict-label " + (label === "POSITIVE" ? "pos" : "neg");
  els.conf.textContent = (conf * 100).toFixed(1) + "%";
  els.fillPos.style.inlineSize = (pos * 100).toFixed(1) + "%";
  els.fillNeg.style.inlineSize = (neg * 100).toFixed(1) + "%";
}

/**
 * Render occlusion attributions as coloured word chips. Green = pushed POSITIVE, red = pushed NEGATIVE,
 * brightness ∝ magnitude. `words`/`attributions` are aligned arrays from engine.attribute().
 */
export function renderAttribution(container, words, attributions) {
  const maxAbs = Math.max(1e-6, ...attributions.map(Math.abs));
  container.replaceChildren(...words.map((w, i) => {
    const a = attributions[i];
    const t = Math.abs(a) / maxAbs;
    const hue = a >= 0 ? "var(--good)" : "var(--bad)";
    const span = document.createElement("span");
    span.className = "attr-word";
    span.textContent = w;
    span.style.background = `color-mix(in srgb, ${hue} ${(t * 60).toFixed(0)}%, transparent)`;
    span.title = `${a >= 0 ? "+" : ""}${a.toFixed(2)} log-odds toward POSITIVE`;
    return span;
  }));
}

export const SENTIMENT_CSS = `
.verdict-row { display: flex; align-items: baseline; gap: .8rem; flex-wrap: wrap; margin-top: .6rem; }
.verdict-label { font-family: var(--font-display); font-size: 1.8rem; }
.verdict-label.pos { color: var(--good); }
.verdict-label.neg { color: var(--bad); }
.verdict-conf { font-family: var(--font-mono); color: var(--muted); font-size: .9rem; }
.meter-dual { display: flex; block-size: .8rem; border: 1px solid var(--border); border-radius: 999px;
  overflow: hidden; margin-top: .5rem; max-inline-size: 520px; background: var(--bg-raised); }
.meter-neg { background: var(--bad); block-size: 100%; }
.meter-pos { background: var(--good); block-size: 100%; }
.meter-labels { display: flex; justify-content: space-between; max-inline-size: 520px;
  font-family: var(--font-mono); font-size: .72rem; color: var(--muted); margin-top: .2rem; }
.attr-wrap { line-height: 2.1; margin-top: .5rem; }
.attr-word { padding: .12rem .28rem; border-radius: 5px; margin: 0 1px; white-space: pre-wrap; }
.attr-legend { display: flex; flex-wrap: wrap; gap: 1rem; font-size: .78rem; color: var(--muted);
  font-family: var(--font-mono); margin-top: .6rem; }
.attr-legend .swatch { display: inline-block; inline-size: .8rem; block-size: .8rem; border-radius: 3px;
  margin-inline-end: .3rem; vertical-align: -1px; }
.review-row { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised);
  padding: .5rem .7rem; display: flex; justify-content: space-between; gap: .6rem; align-items: center; }
.review-row.pos { border-inline-start: 4px solid var(--good); }
.review-row.neg { border-inline-start: 4px solid var(--bad); }
.review-text { flex: 1 1 auto; }
.review-meta { font-family: var(--font-mono); font-size: .76rem; color: var(--muted); white-space: nowrap; }
.badge { font-family: var(--font-mono); font-size: .68rem; padding: .1rem .45rem; border-radius: 999px;
  border: 1px solid var(--border); }
.badge.pos { color: var(--good); border-color: var(--good); }
.badge.neg { color: var(--bad); border-color: var(--bad); }
.sent-list { display: flex; flex-direction: column; gap: .5rem; margin-top: .5rem; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.chip:hover { border-color: var(--accent); }
`;
