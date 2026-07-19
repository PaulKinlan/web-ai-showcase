// Front-end helpers for the Jina embeddings pages. Keeps each page thin: it owns the worker handshake,
// the light client-side maths (cosine, PCA-2D, k-means), and the matrix / projection / search renderers.
// All inference lives in the workers (off the main thread); the maths here runs on a handful of
// already-computed vectors, so it is cheap enough for the main thread.
//
// What makes Jina distinct: jina-embeddings-v2-base-en is a LONG-CONTEXT embedder. Instead of the usual
// 512-token cap, it uses ALiBi (attention with linear biases) so a single vector can represent up to
// 8192 tokens — whole documents, not just snippets. That is the strength these pages lean into: embed
// long passages in full where a 512-token model (MiniLM) would silently truncate the tail.

/** Generic client for any feature-extraction worker sharing our message protocol. */
export class EmbedClient {
  constructor(workerUrl) {
    this.worker = new Worker(workerUrl, { type: "module" });
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

  /** Embed texts → { texts, embeddings:number[][] (unit vectors), norms, tokenCounts, dim, ms, device } */
  embed(texts) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, texts });
    });
  }
}

const BASE = "/web-ai-showcase/models";
export const WORKERS = {
  jina: `${BASE}/jina-embeddings/worker.js`,
  minilm: `${BASE}/minilm-embeddings/worker.js`,
};
export const MODEL_LABELS = {
  jina: "Jina v2 base (8192-token context)",
  minilm: "MiniLM-L6 (512-token context)",
};

/** The Jina engine used by the main "Run it" control on every Jina page. */
export class JinaEngine extends EmbedClient {
  constructor() {
    super(WORKERS.jina);
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

/** PCA to 2D via the small centered Gram matrix (n ≪ d). Returns points + fraction of variance explained. */
export function pca2d(embeddings) {
  const n = embeddings.length;
  const d = embeddings[0]?.length ?? 0;
  if (n < 2) return { points: embeddings.map(() => ({ x: 0, y: 0 })), explained: 0 };
  const mean = new Array(d).fill(0);
  for (const v of embeddings) for (let k = 0; k < d; k++) mean[k] += v[k] / n;
  const X = embeddings.map((v) => v.map((val, k) => val - mean[k]));
  const G = Array.from({ length: n }, () => new Array(n).fill(0));
  let totalVar = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let s = 0;
      for (let k = 0; k < d; k++) s += X[i][k] * X[j][k];
      G[i][j] = s;
      G[j][i] = s;
      if (i === j) totalVar += s;
    }
  }
  const [u1, l1] = topEigen(G, n);
  deflate(G, u1, l1, n);
  const [u2, l2] = topEigen(G, n);
  const s1 = Math.sqrt(Math.max(l1, 0));
  const s2 = Math.sqrt(Math.max(l2, 0));
  const points = [];
  for (let i = 0; i < n; i++) points.push({ x: u1[i] * s1, y: u2[i] * s2 });
  const explained = totalVar > 0 ? (l1 + l2) / totalVar : 0;
  return { points, explained };
}

function topEigen(M, n) {
  let v = new Array(n).fill(0).map(() => Math.random() - 0.5);
  normalizeVec(v);
  let lambda = 0;
  for (let iter = 0; iter < 200; iter++) {
    const w = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += M[i][j] * v[j];
      w[i] = s;
    }
    const norm = Math.sqrt(w.reduce((a, x) => a + x * x, 0));
    if (norm < 1e-12) break;
    for (let i = 0; i < n; i++) w[i] /= norm;
    const diff = w.reduce((a, x, i) => a + Math.abs(x - v[i]), 0);
    v = w;
    lambda = norm;
    if (diff < 1e-9) break;
  }
  return [v, lambda];
}
function deflate(M, u, lambda, n) {
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) M[i][j] -= lambda * u[i] * u[j];
}
function normalizeVec(v) {
  const norm = Math.sqrt(v.reduce((a, x) => a + x * x, 0)) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= norm;
}

/** Simple k-means over unit vectors (cosine = dot). Returns an assignment array of length n. */
export function kmeans(embeddings, k, iters = 25) {
  const n = embeddings.length;
  if (n === 0) return [];
  k = Math.min(k, n);
  // k-means++ style seeding: first random, rest farthest from chosen.
  const centers = [embeddings[Math.floor(Math.random() * n)].slice()];
  while (centers.length < k) {
    let best = -1, bestD = -Infinity;
    for (let i = 0; i < n; i++) {
      let nearest = Infinity;
      for (const c of centers) nearest = Math.min(nearest, 1 - cosine(embeddings[i], c));
      if (nearest > bestD) {
        bestD = nearest;
        best = i;
      }
    }
    centers.push(embeddings[best].slice());
  }
  let assign = new Array(n).fill(0);
  for (let it = 0; it < iters; it++) {
    let moved = false;
    for (let i = 0; i < n; i++) {
      let bestC = 0, bestS = -Infinity;
      for (let c = 0; c < centers.length; c++) {
        const s = cosine(embeddings[i], centers[c]);
        if (s > bestS) {
          bestS = s;
          bestC = c;
        }
      }
      if (assign[i] !== bestC) {
        assign[i] = bestC;
        moved = true;
      }
    }
    for (let c = 0; c < centers.length; c++) {
      const dim = embeddings[0].length;
      const sum = new Array(dim).fill(0);
      let cnt = 0;
      for (let i = 0; i < n; i++) {
        if (assign[i] === c) {
          for (let d = 0; d < dim; d++) sum[d] += embeddings[i][d];
          cnt++;
        }
      }
      if (cnt > 0) {
        normalizeVec(sum);
        centers[c] = sum;
      }
    }
    if (!moved && it > 0) break;
  }
  return assign;
}

/** Parse a textarea of one-per-line into a clean, de-duped, non-empty list. */
export function parseLines(text) {
  return [...new Set(text.split(/\n/).map((s) => s.trim()).filter(Boolean))];
}

/** Split a blob of text into paragraphs (blank-line separated), trimmed and non-empty. */
export function parseParagraphs(text) {
  return text.split(/\n\s*\n/).map((s) => s.replace(/\s+/g, " ").trim()).filter(Boolean);
}

export function simColor(sim) {
  const t = Math.max(0, Math.min(1, sim));
  return `color-mix(in srgb, var(--accent) ${(t * 100).toFixed(0)}%, transparent)`;
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Render an n×n similarity heatmap. `labels` are shown as S1…Sn with the full text in the title. */
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

/** Render a 2D PCA scatter into an SVG. `groups` optionally colours points by cluster. */
export function renderProjection(container, points, labels, { groups = null } = {}) {
  const W = 460, H = 320, pad = 34;
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const spanX = maxX - minX || 1, spanY = maxY - minY || 1;
  const sx = (x) => pad + ((x - minX) / spanX) * (W - 2 * pad);
  const sy = (y) => H - pad - ((y - minY) / spanY) * (H - 2 * pad);
  const palette = [
    "var(--accent)",
    "var(--good)",
    "var(--bad)",
    "var(--warn)",
    "#0e7490",
    "#9333ea",
    "#be185d",
    "#0369a1",
  ];
  const dots = points.map((p, i) => {
    const cx = sx(p.x), cy = sy(p.y);
    const fill = groups ? palette[groups[i] % palette.length] : "var(--accent)";
    const label = (labels[i] || "").replace(/[<>&]/g, "");
    const clipped = label.length > 22 ? label.slice(0, 21) + "…" : label;
    return `<g><circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="6" fill="${fill}"></circle>
      <text x="${(cx + 9).toFixed(1)}" y="${(cy + 4).toFixed(1)}" class="proj-label">S${
      i + 1
    } ${clipped}</text></g>`;
  }).join("");
  container.innerHTML =
    `<svg viewBox="0 0 ${W} ${H}" class="proj-svg" role="img" aria-label="2D PCA projection of the embeddings">
      <line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" class="proj-axis"></line>
      <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${H - pad}" class="proj-axis"></line>
      <text x="${W - pad}" y="${
      H - pad + 20
    }" text-anchor="end" class="proj-axis-label">PC1 →</text>
      <text x="${pad - 6}" y="${pad - 10}" class="proj-axis-label">PC2 ↑</text>
      ${dots}
    </svg>`;
}

/** Render a ranked semantic-search result list. `hits` = [{text, score, tokens?}]. */
export function renderRanked(container, hits, { showBar = true } = {}) {
  container.replaceChildren(...hits.map((h, rank) => {
    const row = document.createElement("div");
    row.className = "result-row" + (rank === 0 ? " top" : "");
    const head = document.createElement("div");
    head.className = "result-head";
    const t = document.createElement("span");
    t.textContent = `${rank + 1}. ${h.text.length > 160 ? h.text.slice(0, 159) + "…" : h.text}`;
    const sc = document.createElement("span");
    sc.className = "result-score";
    sc.textContent = h.score.toFixed(3) + (h.tokens != null ? ` · ${h.tokens} tok` : "");
    head.append(t, sc);
    row.append(head);
    if (showBar) {
      const bar = document.createElement("div");
      bar.className = "result-bar";
      const fill = document.createElement("div");
      fill.className = "result-fill";
      fill.style.inlineSize = `${Math.max(0, Math.min(1, h.score)) * 100}%`;
      bar.append(fill);
      row.append(bar);
    }
    return row;
  }));
}

export const JINA_CSS = `
.sim-matrix { border-collapse: collapse; font-family: var(--font-mono); font-size: .8rem; margin-top: .5rem; }
.sim-matrix th, .sim-matrix td { border: 1px solid var(--border); padding: .3rem .45rem; text-align: center; min-inline-size: 3rem; }
.sim-matrix thead th, .sim-matrix tbody th { color: var(--muted); background: var(--bg-raised); font-weight: 600; }
.sim-matrix td.diag { outline: 2px solid var(--border-strong); outline-offset: -2px; }
.matrix-scroll { overflow-x: auto; }
.proj-svg { inline-size: 100%; max-inline-size: 100%; block-size: auto; background: var(--bg-raised); border: 1px solid var(--border); border-radius: 8px; }
.proj-axis { stroke: var(--border-strong); stroke-width: 1; }
.proj-axis-label { fill: var(--muted); font-family: var(--font-mono); font-size: 11px; }
.proj-label { fill: var(--color); font-family: var(--font-mono); font-size: 10px; }
.vec-strip { display: flex; flex-wrap: wrap; gap: 2px; margin-top: .4rem; }
.vec-cell { inline-size: 12px; block-size: 20px; border-radius: 2px; border: 1px solid var(--border); }
.sent-list { display: flex; flex-direction: column; gap: .5rem; margin-top: .5rem; }
.result-row { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised); padding: .55rem .7rem; }
.result-row.top { border-color: var(--accent); border-inline-start: 4px solid var(--accent); }
.result-head { display: flex; justify-content: space-between; gap: .5rem; align-items: baseline; }
.result-score { font-family: var(--font-mono); color: var(--muted); font-size: .8rem; white-space: nowrap; }
.result-bar { block-size: .45rem; background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 999px; overflow: hidden; margin-top: .3rem; }
.result-fill { block-size: 100%; background: var(--accent); }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; align-items: start; }
.cmp-table { border-collapse: collapse; font-family: var(--font-mono); font-size: .82rem; margin-top: .6rem; inline-size: 100%; }
.cmp-table th, .cmp-table td { border: 1px solid var(--border); padding: .4rem .55rem; text-align: center; }
.cmp-table th:first-child, .cmp-table td:first-child { text-align: start; }
.cmp-table thead th { background: var(--bg-raised); color: var(--muted); }
.chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px; border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.chip:hover, .chip:focus-visible { border-color: var(--accent); }
.ctx-input { font: inherit; inline-size: 100%; padding: .6rem .7rem; border-radius: 8px; border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); resize: vertical; }
.ctx-input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.tokbar { display: flex; block-size: 1.1rem; border-radius: 6px; overflow: hidden; border: 1px solid var(--border); margin: .3rem 0; }
.tokbar .seen { background: var(--good); }
.tokbar .cut { background: color-mix(in srgb, var(--bad) 55%, transparent); }
.cluster-group { border: 1px solid var(--border); border-radius: 10px; background: var(--bg-raised); padding: .5rem .7rem; margin-top: .5rem; }
.cluster-group h4 { margin: 0 0 .3rem; font-size: .82rem; }
.cluster-dot { display: inline-block; inline-size: .7rem; block-size: .7rem; border-radius: 50%; margin-inline-end: .4rem; vertical-align: -1px; }
`;
