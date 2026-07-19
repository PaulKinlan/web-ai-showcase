// Shared front-end helpers for the nomic-embed-text-v1 pages. Keeps each page thin: it owns the worker
// handshake, the Nomic TASK-PREFIX helpers, the light client-side maths (cosine, ranking, clustering),
// and the renderers. All model inference lives in worker.js (off the main thread).
//
// The one discipline that matters for Nomic: it is trained with INSTRUCTION PREFIXES that you prepend to
// the raw text. The prefix is part of what the model reads, so it changes the embedding. Use the right
// one for the job:
//   search_query:    a short query you are searching WITH (asymmetric retrieval)
//   search_document: a passage you are searching OVER / storing in the index
//   classification:  single-text labelling / symmetric similarity
//   clustering:      grouping / dedup / paraphrase mining
// Mismatch them (e.g. embed everything as search_document) and retrieval quality drops.

const WORKER_URL = "/web-ai-showcase/models/nomic-embeddings/worker.js";

export const PREFIXES = {
  query: "search_query: ",
  document: "search_document: ",
  classification: "classification: ",
  clustering: "clustering: ",
};
export const asQuery = (t) => PREFIXES.query + t;
export const asDocument = (t) => PREFIXES.document + t;
export const asClassification = (t) => PREFIXES.classification + t;
export const asClustering = (t) => PREFIXES.clustering + t;

export class EmbedEngine {
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

  /** Embed texts (already prefixed) → { texts, embeddings:number[][], norms, dim, ms, device } */
  embed(texts) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, texts });
    });
  }
}

/** Cosine similarity. Vectors from the worker are unit-length, so this is a dot product. */
export function cosine(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

/** Rank candidate embeddings against a query embedding → [{index, score}] high→low. */
export function rank(queryEmb, candidateEmbs) {
  return candidateEmbs
    .map((v, index) => ({ index, score: cosine(queryEmb, v) }))
    .sort((a, b) => b.score - a.score);
}

/** Full pairwise similarity matrix. */
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

/** All pairs above a threshold, high→low — the core of paraphrase mining / dedup. */
export function pairsAbove(embeddings, threshold) {
  const out = [];
  for (let i = 0; i < embeddings.length; i++) {
    for (let j = i + 1; j < embeddings.length; j++) {
      const score = cosine(embeddings[i], embeddings[j]);
      if (score >= threshold) out.push({ i, j, score });
    }
  }
  return out.sort((a, b) => b.score - a.score);
}

/** Greedy single-link clustering by a cosine threshold (union-find). Returns a cluster id per item. */
export function clusterByThreshold(embeddings, threshold) {
  const n = embeddings.length;
  const parent = Array.from({ length: n }, (_, i) => i);
  const find = (x) => (parent[x] === x ? x : (parent[x] = find(parent[x])));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (cosine(embeddings[i], embeddings[j]) >= threshold) parent[find(i)] = find(j);
    }
  }
  const roots = new Map();
  return Array.from({ length: n }, (_, i) => {
    const r = find(i);
    if (!roots.has(r)) roots.set(r, roots.size);
    return roots.get(r);
  });
}

export function parseLines(text) {
  return [...new Set(text.split(/\n/).map((s) => s.trim()).filter(Boolean))];
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

export function simColor(sim) {
  const t = Math.max(0, Math.min(1, sim));
  return `color-mix(in srgb, var(--accent) ${(t * 100).toFixed(0)}%, transparent)`;
}

/** Render a ranked result list. `items` = [{text, score, tag?}] high→low. */
export function renderRanked(container, items, { max = 999 } = {}) {
  const wrap = document.createElement("div");
  wrap.className = "sent-list";
  items.slice(0, max).forEach((it, r) => {
    const row = document.createElement("div");
    row.className = "result-row";
    const head = document.createElement("div");
    head.className = "result-head";
    const label = document.createElement("span");
    label.innerHTML = `<strong>#${r + 1}</strong> ${escapeHTML(it.text)}` +
      (it.tag ? ` <span class="tone">${escapeHTML(it.tag)}</span>` : "");
    const score = document.createElement("span");
    score.className = "result-score";
    score.textContent = it.score.toFixed(3);
    head.append(label, score);
    const bar = document.createElement("div");
    bar.className = "result-bar";
    const fill = document.createElement("div");
    fill.className = "result-fill";
    fill.style.inlineSize = `${(Math.max(0, Math.min(1, it.score)) * 100).toFixed(0)}%`;
    bar.append(fill);
    row.append(head, bar);
    wrap.append(row);
  });
  container.replaceChildren(wrap);
}

/** Render an n×n cosine heatmap. */
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

/** Render the first N dims of a vector as a coloured strip. */
export function renderVecStrip(container, vec, n = 96) {
  const maxAbs = Math.max(...vec.map(Math.abs)) || 1;
  container.replaceChildren(
    ...vec.slice(0, n).map((val) => {
      const cell = document.createElement("span");
      cell.className = "vec-cell";
      const t = Math.abs(val) / maxAbs;
      const hue = val >= 0 ? "var(--accent)" : "var(--warn)";
      cell.style.background = `color-mix(in srgb, ${hue} ${
        (t * 100).toFixed(0)
      }%, var(--bg-raised))`;
      cell.title = val.toFixed(4);
      return cell;
    }),
  );
}

export const NOMIC_CSS = `
.sim-matrix { border-collapse: collapse; font-family: var(--font-mono); font-size: .8rem; margin-top: .5rem; }
.sim-matrix th, .sim-matrix td { border: 1px solid var(--border); padding: .3rem .45rem; text-align: center; min-inline-size: 3rem; }
.sim-matrix thead th, .sim-matrix tbody th { color: var(--muted); background: var(--bg-raised); font-weight: 600; }
.sim-matrix td.diag { outline: 2px solid var(--border-strong); outline-offset: -2px; }
.matrix-scroll { overflow-x: auto; }
.vec-strip { display: flex; flex-wrap: wrap; gap: 2px; margin-top: .4rem; }
.vec-cell { inline-size: 12px; block-size: 20px; border-radius: 2px; border: 1px solid var(--border); }
.sent-list { display: flex; flex-direction: column; gap: .5rem; margin-top: .5rem; }
.result-row { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised); padding: .55rem .7rem; }
.result-head { display: flex; justify-content: space-between; gap: .5rem; align-items: baseline; }
.result-score { font-family: var(--font-mono); color: var(--muted); font-size: .8rem; white-space: nowrap; }
.result-bar { block-size: .45rem; background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 999px; overflow: hidden; margin-top: .3rem; }
.result-fill { block-size: 100%; background: var(--accent); }
.tone { display: inline-block; font-family: var(--font-mono); font-size: .68rem; padding: .1rem .45rem; border-radius: 999px; border: 1px solid var(--border); margin-inline-start: .4rem; color: var(--muted); }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; align-items: start; }
.seg { display:inline-flex; border:1px solid var(--border-strong); border-radius:8px; overflow:hidden; }
.seg button { border:0; border-radius:0; background:var(--bg-raised); color:var(--color); }
.seg button[aria-pressed=true] { background:var(--accent); color:var(--accent-ink); }
.pill { display:inline-block; font-family:var(--font-mono); font-size:.7rem; padding:.1rem .45rem; border-radius:999px; border:1px solid var(--border); color:var(--muted); }
`;
