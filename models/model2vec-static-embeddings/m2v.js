// Shared front-end helpers for the Model2Vec static-embedding pages. Keeps each page thin: it owns the
// worker handshake, the light client-side maths (cosine similarity), and the matrix / result renderers.
// All model inference (tokenize → gather → mean-pool) lives in worker.js, off the main thread.
//
// The whole story here is SPEED: because Model2Vec has no transformer to run — just a table lookup and
// an average — embedding is microsecond-scale on a CPU. The pages surface the real per-sentence timing
// so you can feel the contrast with an attention-based encoder like MiniLM.

const WORKER_URL = "/web-ai-showcase/models/model2vec-static-embeddings/worker.js";

export class M2VEngine {
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

  /**
   * Embed texts → { texts, embeddings:number[][] (256-d unit), dim, tokenCounts, tokens?, ms, device }.
   * `ms` is a fractional-millisecond wall time for the whole batch (embedding is sub-ms per sentence).
   */
  embed(texts, { withTokens = false } = {}) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, texts, withTokens });
    });
  }
}

/** Cosine similarity. Worker vectors are unit-length, so this is a dot product. */
export function cosine(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

/** Full pairwise similarity matrix for a set of embeddings. */
export function simMatrix(embeddings) {
  const n = embeddings.length;
  const m = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const s = i === j ? 1 : cosine(embeddings[i], embeddings[j]);
      m[i][j] = s;
      m[j][i] = s;
    }
  }
  return m;
}

/** Parse a textarea of one-item-per-line into a clean, de-duped, non-empty list. */
export function parseLines(text) {
  return [...new Set(text.split(/\n/).map((s) => s.trim()).filter(Boolean))];
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

/** Map a cosine value to an accent-over-neutral heatmap fill. */
export function simColor(sim) {
  const t = Math.max(0, Math.min(1, sim));
  return `color-mix(in srgb, var(--accent) ${(t * 100).toFixed(0)}%, transparent)`;
}

/** Render an n×n similarity heatmap into `container`. */
export function renderMatrix(container, matrix, labels) {
  const n = matrix.length;
  const short = labels.map((_, i) => `S${i + 1}`);
  const table = document.createElement("table");
  table.className = "sim-matrix";
  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  hr.appendChild(document.createElement("th"));
  for (let j = 0; j < n; j++) {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = short[j];
    th.title = labels[j];
    hr.appendChild(th);
  }
  thead.appendChild(hr);
  table.appendChild(thead);
  const tb = document.createElement("tbody");
  for (let i = 0; i < n; i++) {
    const tr = document.createElement("tr");
    const rh = document.createElement("th");
    rh.scope = "row";
    rh.textContent = short[i];
    rh.title = labels[i];
    tr.appendChild(rh);
    for (let j = 0; j < n; j++) {
      const td = document.createElement("td");
      const s = matrix[i][j];
      td.textContent = s.toFixed(2);
      td.style.background = simColor(s);
      td.style.color = s > 0.6 ? "var(--accent-ink)" : "var(--color)";
      td.title = `${labels[i]} ↔ ${labels[j]}: cosine ${s.toFixed(4)}`;
      if (i === j) td.classList.add("diag");
      tr.appendChild(td);
    }
    tb.appendChild(tr);
  }
  table.appendChild(tb);
  container.replaceChildren(table);
}

/** Render ranked search results ([{text, score}]) into `container`. */
export function renderResults(container, ranked, { query = "" } = {}) {
  container.replaceChildren(...ranked.map((r, i) => {
    const pct = Math.max(0, Math.min(100, r.score * 100));
    const row = document.createElement("div");
    row.className = "result-row" + (i === 0 ? " result-top" : "");
    row.innerHTML = `
      <div class="result-head"><span>${escapeHTML(r.text)}</span>
        <span class="result-score">${r.score.toFixed(3)}</span></div>
      <div class="result-bar"><div class="result-fill" style="inline-size:${
      pct.toFixed(1)
    }%"></div></div>`;
    return row;
  }));
}

export const M2V_CSS = `
.sim-matrix { border-collapse: collapse; font-family: var(--font-mono); font-size: .8rem; margin-top: .5rem; }
.sim-matrix th, .sim-matrix td {
  border: 1px solid var(--border); padding: .3rem .45rem; text-align: center; min-inline-size: 3rem;
}
.sim-matrix thead th, .sim-matrix tbody th { color: var(--muted); background: var(--bg-raised); font-weight: 600; }
.sim-matrix td.diag { outline: 2px solid var(--border-strong); outline-offset: -2px; }
.matrix-scroll { overflow-x: auto; }
.vec-strip { display: flex; flex-wrap: wrap; gap: 2px; margin-top: .4rem; }
.vec-cell { inline-size: 12px; block-size: 20px; border-radius: 2px; border: 1px solid var(--border); }
.tok-chips { display: flex; flex-wrap: wrap; gap: .3rem; margin: .4rem 0; }
.tok-chip { font-family: var(--font-mono); font-size: .74rem; padding: .12rem .45rem; border-radius: 6px;
  border: 1px solid var(--border); background: var(--bg-raised); }
.tok-arrow { color: var(--muted); }
.result-row { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised);
  padding: .55rem .7rem; margin-top: .5rem; }
.result-row.result-top { border-color: var(--accent); }
.result-head { display: flex; justify-content: space-between; gap: .5rem; align-items: baseline; }
.result-score { font-family: var(--font-mono); color: var(--muted); font-size: .8rem; white-space: nowrap; }
.result-bar { block-size: .45rem; background: var(--bg-secondary); border: 1px solid var(--border);
  border-radius: 999px; overflow: hidden; margin-top: .3rem; }
.result-fill { block-size: 100%; background: var(--accent); }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.big-num { font-family: var(--font-display); font-size: 1.6rem; color: var(--accent); }
.speed-card { display: flex; flex-wrap: wrap; gap: 1.2rem; align-items: baseline; margin-top: .4rem; }
.grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; align-items: start; }
.field-row { display: flex; flex-wrap: wrap; gap: .6rem; align-items: end; margin: .6rem 0; }
`;
