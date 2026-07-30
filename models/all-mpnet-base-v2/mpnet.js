// Front-end helpers for the all-mpnet-base-v2 embedding pages. Keeps each page thin.
//
// Engines are built on the shared, versioned lib/worker-protocol.js (WorkerClient): one in-flight
// inference request, a bounded queue, cooperative cancellation (AbortSignal), stale-response
// suppression, latest-wins channels, terminal fatal errors, and deterministic dispose (terminate).
// The pooled 768-d vectors come back as a TRANSFERRED Float64Array (ownership moves, no clone).
// All inference + k-means clustering runs off the main thread so INP stays low.

import { SupersededError, WorkerClient } from "/web-ai-showcase/lib/worker-protocol.js";

const WORKER = "/web-ai-showcase/models/all-mpnet-base-v2/worker.js";
const RERANKER_WORKER = "/web-ai-showcase/models/all-mpnet-base-v2/reranker-worker.js";

/** Re-export so callers can recognise latest-wins drops where expected. */
export { SupersededError };

/** MPNet embedding engine over a bounded WorkerClient. */
export class MpNetEngine {
  constructor() {
    this.client = null;
    this.device = "wasm";
  }
  /** Load the pinned q8 model. onProgress forwards per-file download progress to the shared loader. */
  async load(onProgress) {
    if (this.client) return this.device;
    const client = new WorkerClient({
      url: WORKER,
      name: "mpnet",
      maxInFlight: 1, // single-flight: one inference at a time
      maxQueue: 8, // bounded backpressure; overflow rejects deterministically
    });
    try {
      await client.ready; // init is light; the heavy download is the `load` request below
      const { result } = await client.request("load", {}, { onProgress });
      this.client = client;
      this.device = result.device;
      return this.device;
    } catch (err) {
      await client.terminate().catch(() => {});
      throw err;
    }
  }
  /** Embed texts → { texts, embeddings:number[][] (unit vectors), norms, dim, ms, device }. */
  async embed(texts, opts = {}) {
    if (!this.client) throw new Error("Model not loaded");
    const { result } = await this.client.request(
      "embed",
      { texts },
      { channel: opts.channel, signal: opts.signal },
    );
    const { flat, dim, n } = result;
    const embeddings = [];
    for (let i = 0; i < n; i++) {
      embeddings.push(Array.from(flat.subarray(i * dim, (i + 1) * dim)));
    }
    return {
      texts,
      embeddings,
      norms: result.norms,
      dim,
      ms: result.ms,
      device: result.device,
    };
  }
  /** Cluster texts into k groups off the main thread → { assign, k, central, cohesion, stability, ms }. */
  async cluster(texts, k, opts = {}) {
    if (!this.client) throw new Error("Model not loaded");
    const { result } = await this.client.request("cluster", { texts, k }, { signal: opts.signal });
    return result;
  }
  /** Deterministic teardown — dispose the worker's model + terminate it. Idempotent. */
  async close() {
    if (this.client) {
      await this.client.terminate();
      this.client = null;
    }
  }
}

/** Family-specific PINNED MS MARCO cross-encoder reranker engine (revision a09144355…). */
export class PinnedRerankEngine {
  constructor() {
    this.client = null;
    this.device = "wasm";
  }
  async load(onProgress) {
    if (this.client) return this.device;
    const client = new WorkerClient({
      url: RERANKER_WORKER,
      name: "mpnet-reranker",
      maxInFlight: 1,
      maxQueue: 8,
    });
    try {
      await client.ready;
      const { result } = await client.request("load", {}, { onProgress });
      this.client = client;
      this.device = result.device;
      return this.device;
    } catch (err) {
      await client.terminate().catch(() => {});
      throw err;
    }
  }
  /** rerank(query, passages) → { results:[{idx,passage,logit,prob,lexical}], ms, device }. */
  async rerank(query, passages, opts = {}) {
    if (!this.client) throw new Error("Reranker not loaded");
    const { result } = await this.client.request(
      "rerank",
      { query, passages },
      { signal: opts.signal },
    );
    return result;
  }
  async close() {
    if (this.client) {
      await this.client.terminate();
      this.client = null;
    }
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

/**
 * PCA down to 2 dimensions via the (small) centered Gram matrix, with a DETERMINISTIC sign convention
 * (the largest-magnitude component of each axis is forced positive) so the projection is identical on
 * every run. For n points in d dims with n ≪ d this is far cheaper than a d×d covariance. Returns
 * [{x,y}] plus the fraction of variance the two axes explain and a residual ratio (≈0 ⇒ converged).
 */
export function pca2d(embeddings) {
  const n = embeddings.length;
  const d = embeddings[0]?.length ?? 0;
  if (n < 2) return { points: embeddings.map(() => ({ x: 0, y: 0 })), explained: 0, residual: 0 };

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

  const [u1, l1, r1] = topEigen(G, n, 1);
  deflate(G, u1, l1, n);
  const [u2, l2, r2] = topEigen(G, n, 2);
  // Deterministic sign: flip each axis so its largest-magnitude component is positive.
  fixSign(u1, n);
  fixSign(u2, n);

  const s1 = Math.sqrt(Math.max(l1, 0));
  const s2 = Math.sqrt(Math.max(l2, 0));
  const points = [];
  for (let i = 0; i < n; i++) points.push({ x: u1[i] * s1, y: u2[i] * s2 });
  const explained = totalVar > 0 ? (l1 + l2) / totalVar : 0;
  return { points, explained, residual: Math.max(r1, r2) };
}

function topEigen(M, n, seed) {
  // A centered Gram matrix annihilates the all-ones vector, so an all-equal deterministic start is
  // invalid. Use a deterministic, centered non-uniform seed for each axis instead.
  let v = Array.from(
    { length: n },
    (_, i) => Math.sin((i + 1) * (seed + 0.5)) + 0.05 * (i + 1) * seed,
  );
  const mean = v.reduce((sum, value) => sum + value, 0) / n;
  v = v.map((value) => value - mean);
  normalizeVec(v);
  let lambda = 0;
  let residual = Infinity;
  for (let iter = 0; iter < 300; iter++) {
    const y = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) y[i] += M[i][j] * v[j];
    }
    const norm = Math.sqrt(y.reduce((sum, value) => sum + value * value, 0));
    if (norm < 1e-12) return [new Array(n).fill(0), 0, 0];
    const w = y.map((value) => value / norm);
    fixSign(w, n);

    const mw = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) mw[i] += M[i][j] * w[j];
    }
    lambda = w.reduce((sum, value, i) => sum + value * mw[i], 0);
    const residualNorm = Math.sqrt(
      mw.reduce((sum, value, i) => sum + (value - lambda * w[i]) ** 2, 0),
    );
    residual = residualNorm / Math.max(Math.abs(lambda), 1e-12);
    const delta = Math.sqrt(w.reduce((sum, value, i) => sum + (value - v[i]) ** 2, 0));
    v = w;
    if (residual < 1e-9 && delta < 1e-8) break;
  }
  return [v, lambda, residual];
}

function fixSign(v, n) {
  let bi = 0, bm = 0;
  for (let i = 0; i < n; i++) {
    if (Math.abs(v[i]) > bm) {
      bm = Math.abs(v[i]);
      bi = i;
    }
  }
  if (v[bi] < 0) { for (let i = 0; i < n; i++) v[i] = -v[i]; }
}

function deflate(M, u, lambda, n) {
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) M[i][j] -= lambda * u[i] * u[j];
  }
}

function normalizeVec(v) {
  const norm = Math.sqrt(v.reduce((a, x) => a + x * x, 0)) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= norm;
}

/** Parse a textarea of one-per-line into a clean, de-duped, non-empty list. */
export function parseLines(text) {
  return [...new Set(text.split(/\n/).map((s) => s.trim()).filter(Boolean))];
}

/**
 * HONEST framing for a single cosine score. MPNet cosine is an UNCALIBRATED, model-specific
 * relatedness score (higher = more semantically similar) — NOT a universal threshold, calibrated
 * probability, or fixed semantic verdict. Returns a neutral descriptor without invented bands.
 */
export function uncalibratedNote(score) {
  if (!Number.isFinite(score)) return "—";
  // A single, clearly-labelled qualitative bucket driven by THIS model's observed geometry, not a
  // universal rule; unrelated pairs land near 0, paraphrases higher.
  let band;
  if (score >= 0.6) band = "strongly related";
  else if (score >= 0.35) band = "related";
  else if (score >= 0.1) band = "weakly related";
  else band = "largely unrelated";
  return `${band} (uncalibrated, model-specific — not a universal threshold)`;
}

/** Map a cosine value in [-1,1] to accent opacity over a neutral track for a heatmap cell. */
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

/** Render a 2D PCA scatter into an SVG with COMPACT numbered markers (S1…Sn) only — long labels live in
 *  the separate legend to avoid overlap. `groups` optionally colours points. */
export function renderProjection(container, points, { groups = null } = {}) {
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
    "var(--accent)",
    "var(--good)",
    "var(--bad)",
    "var(--warn)",
  ];
  const dots = points.map((p, i) => {
    const cx = sx(p.x), cy = sy(p.y);
    const fill = groups ? palette[groups[i] % palette.length] : "var(--accent)";
    return `<g><circle cx="${cx.toFixed(1)}" cy="${
      cy.toFixed(1)
    }" r="6" fill="${fill}"></circle><text x="${(cx + 10).toFixed(1)}" y="${
      (cy + 4).toFixed(1)
    }" class="proj-label">S${i + 1}</text></g>`;
  }).join("");
  container.innerHTML =
    `<svg viewBox="0 0 ${W} ${H}" class="proj-svg" role="img" aria-label="2D PCA projection of the sentence embeddings (numbered points; see legend)">
      <line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" class="proj-axis"></line>
      <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${H - pad}" class="proj-axis"></line>
      <text x="${W - pad}" y="${
      H - pad + 20
    }" text-anchor="end" class="proj-axis-label">PC1 →</text>
      <text x="${pad - 6}" y="${pad - 10}" class="proj-axis-label">PC2 ↑</text>
      ${dots}
    </svg>`;
}

/** Render the numbered legend that accompanies the PCA scatter (S1… → full text). */
export function renderLegend(container, labels, { groups = null, groupNames = null } = {}) {
  const palette = [
    "var(--accent)",
    "var(--good)",
    "var(--bad)",
    "var(--warn)",
    "var(--accent)",
    "var(--good)",
    "var(--bad)",
    "var(--warn)",
  ];
  container.replaceChildren(
    ...labels.map((label, i) => {
      const row = document.createElement("div");
      row.className = "legend-row";
      const sw = document.createElement("span");
      sw.className = "swatch";
      sw.style.background = groups ? palette[groups[i] % palette.length] : "var(--accent)";
      const t = document.createElement("span");
      t.className = "legend-text";
      let prefix = `S${i + 1} — `;
      if (groups != null && groupNames) prefix += `[${groupNames[groups[i]] ?? groups[i]}] `;
      t.textContent = prefix + label;
      row.append(sw, t);
      return row;
    }),
  );
}

/** Render a ranked semantic-search result list into `container`. `hits` = [{text, score, group?}].
 *  `tags` (if provided) sets a per-row tag; an explicit `tagClass` makes abstention visible. */
export function renderRanked(
  container,
  hits,
  { showBar = true, tags = null, tagClass = null } = {},
) {
  container.replaceChildren(
    ...hits.map((h, rank) => {
      const row = document.createElement("div");
      row.className = "result-row";
      const head = document.createElement("div");
      head.className = "result-head";
      const t = document.createElement("span");
      t.textContent = `${rank + 1}. ${h.text}`;
      const sc = document.createElement("span");
      sc.className = "result-score";
      sc.textContent = h.score.toFixed(3);
      if (tags && tags[rank] != null) {
        const tag = document.createElement("span");
        tag.className = "tone" + (tagClass ? " " + tagClass[rank] : "");
        tag.textContent = tags[rank];
        t.append(tag);
      }
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
    }),
  );
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Shared inline styles for the embedding widgets. Injected once per page. */
export const EMBEDDING_CSS = `
.sim-matrix { border-collapse: collapse; font-family: var(--font-mono); font-size: .8rem; margin-top: .5rem; }
.sim-matrix th, .sim-matrix td {
  border: 1px solid var(--border); padding: .3rem .45rem; text-align: center; min-inline-size: 3rem;
}
.sim-matrix thead th, .sim-matrix tbody th { color: var(--muted); background: var(--bg-raised); font-weight: 600; }
.sim-matrix td.diag { outline: 2px solid var(--border-strong); outline-offset: -2px; }
.matrix-scroll { overflow-x: auto; }
.proj-svg { inline-size: 100%; max-inline-size: 100%; block-size: auto; background: var(--bg-raised);
  border: 1px solid var(--border); border-radius: 8px; }
.proj-axis { stroke: var(--border-strong); stroke-width: 1; }
.proj-axis-label { fill: var(--muted); font-family: var(--font-mono); font-size: 11px; }
.proj-label { fill: var(--color); font-family: var(--font-mono); font-size: 11px; font-weight: 600; }
.legend { display: flex; flex-direction: column; gap: .2rem; font-size: .78rem; color: var(--muted);
  font-family: var(--font-mono); margin-top: .4rem; }
.legend-row { display: flex; gap: .4rem; align-items: baseline; }
.legend-row .swatch { display: inline-block; inline-size: .7rem; block-size: .7rem; border-radius: 50%; flex: none; }
.legend-text { word-break: break-word; }
.vec-strip { display: flex; flex-wrap: wrap; gap: 2px; margin-top: .4rem; }
.vec-cell { inline-size: 12px; block-size: 20px; border-radius: 2px; border: 1px solid var(--border); }
.sent-list { display: flex; flex-direction: column; gap: .5rem; margin-top: .5rem; }
.result-row { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised); padding: .55rem .7rem; }
.result-head { display: flex; justify-content: space-between; gap: .5rem; align-items: baseline; }
.result-score { font-family: var(--font-mono); color: var(--muted); font-size: .8rem; white-space: nowrap; }
.result-bar { block-size: .45rem; background: var(--bg-secondary); border: 1px solid var(--border);
  border-radius: 999px; overflow: hidden; margin-top: .3rem; }
.result-fill { block-size: 100%; background: var(--accent); }
.tone { display: inline-block; font-family: var(--font-mono); font-size: .68rem; padding: .1rem .45rem;
  border-radius: 999px; border: 1px solid var(--border); margin-inline-start: .4rem; color: var(--muted); }
.tone-pos { color: var(--good); border-color: var(--good); }
.tone-neg { color: var(--bad); border-color: var(--bad); }
.tone-abstain { color: var(--warn); border-color: var(--warn); }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; align-items: start; }
.cmp-table { border-collapse: collapse; font-family: var(--font-mono); font-size: .82rem; margin-top: .6rem; inline-size: 100%; }
.cmp-table th, .cmp-table td { border: 1px solid var(--border); padding: .35rem .55rem; text-align: center; }
.cmp-table th:first-child, .cmp-table td:first-child { text-align: start; }
.cmp-table thead th { background: var(--bg-raised); color: var(--muted); }
.cmp-pair { color: var(--color); }
/* Coarse-pointer-friendly tap targets (~44px). */
.chip { font: inherit; font-size: .8rem; padding: .45rem .7rem; border-radius: 999px; border: 1px solid var(--border);
  background: var(--bg-raised); color: var(--color); cursor: pointer; min-block-size: 40px; display: inline-flex;
  align-items: center; }
.chip:hover { border-color: var(--accent); }
.corpus-area { inline-size: 100%; box-sizing: border-box; }
.cluster { border:1px solid var(--border); border-inline-start:4px solid var(--accent); border-radius:8px;
  background:var(--bg-raised); padding:.5rem .7rem; margin:.5rem 0; }
.cluster h4 { margin:.1rem 0 .3rem; font-family:var(--font-mono); font-size:.8rem; color:var(--muted); }
.cluster ul { margin:.2rem 0; padding-inline-start:1.1rem; } .cluster li { margin:.15rem 0; }
`;
