// Front-end helpers for the SPLADE sparse-retrieval page: the worker handshake, sparse dot-product, and
// term-cloud rendering. All inference lives in worker.js (off the main thread).

export class SpladeEngine {
  constructor() {
    this.worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
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
    } else if (msg.type === "result") {
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
        for (const w of this._loadWaiters) w.reject(new Error(msg.message));
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
  /** Encode text → { terms:[{term,weight,expansion}], sparse:{id:weight}, nonZero, ms }. */
  encode(text) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "encode", id, text });
    });
  }
}

/** Sparse dot-product of two {id: weight} maps — the SPLADE relevance score. */
export function sparseDot(a, b) {
  let s = 0;
  const [small, big] = Object.keys(a).length < Object.keys(b).length ? [a, b] : [b, a];
  for (const id in small) {
    if (id in big) s += small[id] * big[id];
  }
  return s;
}

export function escapeHTML(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]),
  );
}

/** Render terms as a weighted cloud: font-size ∝ weight; expansion terms get the accent style. */
export function renderCloud(container, terms) {
  const max = terms.length ? terms[0].weight : 1;
  container.innerHTML = terms.map((t) => {
    const size = (0.8 + (t.weight / max) * 1.1).toFixed(2);
    const cls = t.expansion ? "sp-term sp-exp" : "sp-term";
    return `<span class="${cls}" style="font-size:${size}rem" title="${t.weight.toFixed(2)}${
      t.expansion ? " · expansion" : ""
    }">${escapeHTML(t.term)}</span>`;
  }).join(" ");
}

export const SPLADE_CSS = `
  .sp-field { margin: 0.5rem 0; }
  .sp-field label { display: block; font-size: 0.85rem; margin-bottom: 0.2rem; color: var(--muted, #888); }
  .sp-field input { width: 100%; box-sizing: border-box; padding: 0.55rem 0.65rem; border-radius: 8px; border: 1px solid #8886; font: inherit; font-size: 0.98rem; }
  .sp-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.5rem 0; }
  .sp-cloud { display: flex; flex-wrap: wrap; gap: 0.3rem 0.55rem; align-items: baseline; line-height: 1.8; margin: 0.6rem 0; }
  .sp-term { color: var(--fg, inherit); border-bottom: 2px solid #2bb59a; padding: 0 0.1em; }
  .sp-term.sp-exp { color: #c77dff; border-bottom-color: #c77dff; font-style: italic; }
  .sp-term.sp-exp::after { content: "+"; font-size: 0.6em; vertical-align: super; opacity: 0.7; }
  .sp-legend { display: flex; flex-wrap: wrap; gap: 0.4rem 1rem; font-size: 0.8rem; margin: 0.3rem 0; }
  .sp-legend b { border-bottom: 2px solid #2bb59a; }
  .sp-legend i { color: #c77dff; border-bottom: 2px solid #c77dff; font-style: normal; }
  .sp-match { font-size: 1.05rem; font-weight: 600; margin: 0.6rem 0 0.2rem; }
  .sp-shared { display: flex; flex-wrap: wrap; gap: 0.3rem; margin: 0.3rem 0; }
  .sp-shared span { font-size: 0.8rem; padding: 0.1rem 0.4rem; border-radius: 999px; background: #2bb59a22; border: 1px solid #2bb59a55; }
  .sp-status { font-family: var(--font-mono, monospace); font-size: 0.9rem; margin: 0.4rem 0; }
`;
