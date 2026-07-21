// Front-end helpers for the ColBERT late-interaction page: the worker handshake + alignment rendering.
// All inference lives in worker.js (off the main thread).

export class ColbertEngine {
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
  /** Score a query against a document → { docTokens, align, score, ms }. */
  score(query, document) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "score", id, query, document });
    });
  }
}

export function escapeHTML(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]),
  );
}

// Similarity → teal intensity. ColBERT cosine sims cluster high, so stretch the 0.4–1.0 band.
function simColor(sim) {
  const t = Math.max(0, Math.min(1, (sim - 0.4) / 0.6));
  return `hsl(168 60% ${88 - t * 40}% / ${0.25 + t * 0.75})`;
}

/** Render the per-query-token alignment: each query word → its best document word + similarity. */
export function renderAlignment(container, align) {
  container.innerHTML = align.map((a) => {
    const pct = Math.round(a.sim * 100);
    return `<div class="cb-pair">` +
      `<span class="cb-q">${escapeHTML(a.q)}</span>` +
      `<span class="cb-arrow" aria-label="matches">→</span>` +
      `<span class="cb-d" style="background:${simColor(a.sim)}">${escapeHTML(a.d || "·")}</span>` +
      `<span class="cb-sim">${pct}%</span></div>`;
  }).join("");
}

/** Render the document with each token tinted by how strongly the query matched it. */
export function renderDoc(container, docTokens) {
  container.innerHTML = docTokens.map((t) =>
    `<span class="cb-dtok" style="background:${simColor(t.match)}" title="best query match ${
      (t.match * 100) | 0
    }%">${escapeHTML(t.str)}</span>`
  ).join(" ");
}

export const COLBERT_CSS = `
  .cb-field { margin: 0.5rem 0; }
  .cb-field label { display: block; font-size: 0.85rem; margin-bottom: 0.2rem; color: var(--muted, #888); }
  .cb-field input, .cb-field textarea { width: 100%; box-sizing: border-box; padding: 0.55rem 0.65rem; border-radius: 8px; border: 1px solid #8886; font: inherit; font-size: 0.95rem; }
  .cb-field textarea { min-height: 3.5rem; resize: vertical; }
  .cb-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.5rem 0; }
  .cb-score { font-size: 1.1rem; font-weight: 600; margin: 0.7rem 0 0.3rem; }
  .cb-doc { line-height: 2.1; margin: 0.4rem 0 0.8rem; }
  .cb-dtok { padding: 0.05em 0.15em; border-radius: 3px; }
  .cb-align { display: flex; flex-direction: column; gap: 0.25rem; max-width: 30rem; }
  .cb-pair { display: grid; grid-template-columns: 1fr auto 1fr 2.6rem; align-items: center; gap: 0.4rem; font-size: 0.86rem; }
  .cb-q { font-weight: 600; text-align: right; }
  .cb-arrow { color: var(--muted, #888); }
  .cb-d { padding: 0.05em 0.35em; border-radius: 4px; font-family: var(--font-mono, monospace); }
  .cb-sim { font-family: var(--font-mono, monospace); text-align: right; color: var(--muted, #888); }
  .cb-status { font-family: var(--font-mono, monospace); font-size: 0.9rem; margin: 0.4rem 0; }
`;
