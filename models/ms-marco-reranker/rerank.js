// Shared front-end helpers for the MS MARCO cross-encoder reranker pages. Keeps each page thin: it owns
// the worker handshake and the renderers (ranked passage list, cross-encoder-vs-lexical contrast).
// All inference lives in worker.js (off the main thread).

const WORKER_URL = "/web-ai-showcase/models/ms-marco-reranker/worker.js";

export class RerankEngine {
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

  /** Rerank → { query, results:[{idx,passage,logit,prob,lexical}] (sorted), ms, device }. */
  rerank(query, passages) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, query, passages });
    });
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

export function parseLines(text) {
  return [...new Set(text.split(/\n/).map((s) => s.trim()).filter(Boolean))];
}

/**
 * Render the reranked passages. `results` is already sorted by cross-encoder logit (descending).
 * The bar encodes the sigmoid relevance probability; the raw logit is shown numerically.
 */
export function renderResults(container, results) {
  // MS MARCO logits are unbounded and usually negative, so a sigmoid bar would be near-invisible. Bar
  // width is a min-max scaling of the logit across THIS candidate set (best fills, worst empties) purely
  // for legibility; the honest raw logit and its sigmoid stay as numbers.
  const logits = results.map((r) => r.logit);
  const lo = Math.min(...logits), hi = Math.max(...logits);
  const span = hi - lo || 1;
  container.replaceChildren(...results.map((r, rank) => {
    const row = document.createElement("div");
    row.className = "rr-row" + (rank === 0 ? " top" : "");
    const w = ((r.logit - lo) / span) * 100;
    row.innerHTML = `<div class="rr-head"><span class="rr-rank">#${rank + 1}</span>` +
      `<span class="rr-logit">logit ${r.logit.toFixed(2)}</span></div>` +
      `<div class="rr-text">${escapeHTML(r.passage)}</div>` +
      `<div class="rr-bar"><span class="rr-fill" style="inline-size:${
        w.toFixed(1)
      }%"></span></div>` +
      `<div class="rr-meta">relative relevance (rank-scaled) · sigmoid ${
        (r.prob * 100).toFixed(1)
      }% · lexical overlap ${(r.lexical * 100).toFixed(0)}%</div>`;
    return row;
  }));
}

/**
 * Render the "see inside" contrast: the same passages ranked by the cross-encoder vs by naive lexical
 * overlap, side by side, so the reordering (semantic vs word-matching) is visible.
 */
export function renderContrast(container, results) {
  const byLex = [...results].sort((a, b) => b.lexical - a.lexical || b.logit - a.logit);
  const shorten = (s) => (s.length > 46 ? s.slice(0, 45) + "…" : s);
  const col = (title, list, scoreFn) =>
    `<div class="ct-col"><h4>${title}</h4>` +
    list.map((r, i) =>
      `<div class="ct-item"><span class="ct-rank">${i + 1}</span>` +
      `<span class="ct-text">${escapeHTML(shorten(r.passage))}</span>` +
      `<span class="ct-score">${scoreFn(r)}</span></div>`
    ).join("") + `</div>`;
  container.innerHTML = col("Cross-encoder rank", results, (r) => r.logit.toFixed(1)) +
    col("Lexical-overlap rank", byLex, (r) => (r.lexical * 100).toFixed(0) + "%");
}

export const RERANK_CSS = `
.rr-list { display: flex; flex-direction: column; gap: .5rem; margin-top: .5rem; }
.rr-row { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised); padding: .5rem .7rem; }
.rr-row.top { border-color: var(--accent); border-inline-start: 4px solid var(--accent); }
.rr-head { display: flex; justify-content: space-between; align-items: baseline; gap: .5rem; }
.rr-rank { font-family: var(--font-mono); color: var(--muted); font-size: .8rem; font-weight: 600; }
.rr-logit { font-family: var(--font-mono); font-size: .78rem; color: var(--muted); white-space: nowrap; }
.rr-text { margin: .3rem 0; font-size: .92rem; }
.rr-bar { block-size: .5rem; background: var(--bg-secondary); border: 1px solid var(--border);
  border-radius: 999px; overflow: hidden; }
.rr-fill { display: block; block-size: 100%; background: var(--accent); }
.rr-meta { font-family: var(--font-mono); font-size: .72rem; color: var(--muted); margin-top: .25rem; }
.ct-wrap { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; margin-top: .5rem; }
.ct-col h4 { margin: 0 0 .4rem; font-family: var(--font-mono); font-size: .8rem; color: var(--muted); }
.ct-item { display: grid; grid-template-columns: auto 1fr auto; gap: .5rem; align-items: center;
  border: 1px solid var(--border); border-radius: 6px; background: var(--bg-raised); padding: .35rem .5rem; margin-bottom: .35rem; }
.ct-rank { font-family: var(--font-mono); color: var(--muted); font-size: .76rem; }
.ct-text { font-size: .82rem; }
.ct-score { font-family: var(--font-mono); font-size: .76rem; color: var(--muted); white-space: nowrap; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.chip:hover { border-color: var(--accent); }
`;
