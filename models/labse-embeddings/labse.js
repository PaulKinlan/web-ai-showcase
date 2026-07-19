// Front-end helpers for the LaBSE pages. Keeps each page thin: it owns the worker handshake, the light
// client-side maths (cosine, similarity matrix, PCA-2D, k-means, bitext mining) and the renderers
// (heatmap, 2D map coloured by language, ranked list, bitext-alignment table). All actual inference
// lives in worker.js (off the main thread); the maths here runs on a handful of already-computed 768-d
// vectors, so it is cheap enough for the main thread.

/** Client for the LaBSE feature-extraction worker. */
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

  /** Embed texts → { texts, embeddings:number[][] (unit vectors), norms, dim, ms, device } */
  embed(texts) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, texts });
    });
  }
}

const BASE = "/web-ai-showcase/models/labse-embeddings";
export class LabseEngine extends EmbedClient {
  constructor() {
    super(`${BASE}/worker.js`);
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

/** Cross similarity: every source vector vs every candidate vector → rows=sources, cols=candidates. */
export function crossSim(sources, candidates) {
  return sources.map((s) => candidates.map((c) => cosine(s, c)));
}

/**
 * Bitext mining: for each source, pick the best-matching candidate by cosine (greedy argmax with a
 * uniqueness pass so two sources don't grab the same candidate). Returns per-source
 * { srcIndex, bestCand, best, runnerUp, margin } where margin = best − runnerUp (LaBSE's retrieval
 * confidence — a large margin means the true translation stands clear of every distractor).
 */
export function bitextMine(sources, candidates) {
  const S = crossSim(sources, candidates);
  const n = sources.length;
  const taken = new Set();
  // Assign in order of each source's top score (most confident first) so strong matches win their pick.
  const order = [...Array(n).keys()].sort((a, b) => Math.max(...S[b]) - Math.max(...S[a]));
  const out = new Array(n);
  for (const i of order) {
    const ranked = S[i]
      .map((score, j) => ({ j, score }))
      .sort((a, b) => b.score - a.score);
    const pick = ranked.find((r) => !taken.has(r.j)) ?? ranked[0];
    taken.add(pick.j);
    const runnerUp = ranked.find((r) => r.j !== pick.j)?.score ?? 0;
    out[i] = {
      srcIndex: i,
      bestCand: pick.j,
      best: pick.score,
      runnerUp,
      margin: pick.score - runnerUp,
    };
  }
  return out;
}

/** PCA down to 2 dims via the (small) centered Gram matrix (cheap for n ≪ d). */
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
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) M[i][j] -= lambda * u[i] * u[j];
  }
}

function normalizeVec(v) {
  const norm = Math.sqrt(v.reduce((a, x) => a + x * x, 0)) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= norm;
}

/** k-means over unit vectors (cosine = dot on the sphere); a few restarts, returns cluster ids. */
export function kmeans(embeddings, k, iters = 25) {
  const n = embeddings.length;
  if (n === 0) return [];
  k = Math.min(k, n);
  const idx = [...Array(n).keys()];
  for (let i = idx.length - 1; i > 0; i--) {
    const j = (Math.random() * (i + 1)) | 0;
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  let centroids = idx.slice(0, k).map((i) => embeddings[i].slice());
  let assign = new Array(n).fill(0);
  for (let it = 0; it < iters; it++) {
    let moved = false;
    for (let i = 0; i < n; i++) {
      let best = 0, bestSim = -Infinity;
      for (let c = 0; c < k; c++) {
        const s = cosine(embeddings[i], centroids[c]);
        if (s > bestSim) {
          bestSim = s;
          best = c;
        }
      }
      if (assign[i] !== best) {
        assign[i] = best;
        moved = true;
      }
    }
    const sums = Array.from({ length: k }, () => new Array(embeddings[0].length).fill(0));
    const counts = new Array(k).fill(0);
    for (let i = 0; i < n; i++) {
      counts[assign[i]]++;
      const v = embeddings[i];
      const s = sums[assign[i]];
      for (let d = 0; d < v.length; d++) s[d] += v[d];
    }
    centroids = sums.map((s, c) => {
      if (counts[c] === 0) return embeddings[(Math.random() * n) | 0].slice();
      normalizeVec(s);
      return s;
    });
    if (!moved && it > 0) break;
  }
  return assign;
}

/** A tiny, dependency-free language guesser — script ranges first, then stop-word hints for a few Latin
 *  languages. It only drives the flag/badge UI (it is NOT the model); the embeddings do the real work. */
const LATIN_HINTS = [
  [
    "fr",
    /\b(le|la|les|des|une|est|dans|pour|avec|aujourd|vous|nous|je|il|elle|c'est|pas|être|voudrais|réserver|où)\b/i,
  ],
  [
    "es",
    /\b(el|la|los|las|una|est[aá]|para|con|pero|c[oó]mo|hoy|muy|m[aá]s|qu[eé]|ni[ñn]os|cu[aá]ndo|d[oó]nde|contrase[ñn]a|pedido|env[ií]o|d[ií]as|gratis|producto|devolver)\b/i,
  ],
  [
    "de",
    /\b(der|die|das|und|ist|nicht|mit|für|ein|eine|auch|heute|wetter|kinder|wenn|sie|ger[aä]t|schön)\b/i,
  ],
  ["pt", /\b(os|as|uma|está|para|com|não|hoje|muito|você|obrigado|crianças)\b/i],
  ["it", /\b(lo|gli|una|è|per|con|non|oggi|molto|questo|bambini|grazie)\b/i],
];
export function guessLang(text) {
  const t = String(text);
  if (/[぀-ヿ]/.test(t)) return { code: "ja", flag: "🇯🇵", name: "Japanese" };
  if (/[가-힣]/.test(t)) return { code: "ko", flag: "🇰🇷", name: "Korean" };
  if (/[一-鿿]/.test(t)) return { code: "zh", flag: "🇨🇳", name: "Chinese" };
  if (/[Ѐ-ӿ]/.test(t)) return { code: "ru", flag: "🇷🇺", name: "Russian" };
  if (/[؀-ۿ]/.test(t)) return { code: "ar", flag: "🇸🇦", name: "Arabic" };
  if (/[ऀ-ॿ]/.test(t)) return { code: "hi", flag: "🇮🇳", name: "Hindi" };
  if (/[Ͱ-Ͽ]/.test(t)) return { code: "el", flag: "🇬🇷", name: "Greek" };
  if (/[¿¡ñ]/.test(t)) return { code: "es", flag: "🇪🇸", name: "Spanish" };
  if (/ß/.test(t) || /[äöü].*\b(der|die|das|und|ist|nicht|mit|für)\b/i.test(t)) {
    return { code: "de", flag: "🇩🇪", name: "German" };
  }
  for (const [code, re] of LATIN_HINTS) {
    if (re.test(t)) {
      const map = {
        fr: "🇫🇷 French",
        es: "🇪🇸 Spanish",
        de: "🇩🇪 German",
        pt: "🇵🇹 Portuguese",
        it: "🇮🇹 Italian",
      };
      const [flag, name] = map[code].split(" ");
      return { code, flag, name };
    }
  }
  return { code: "en", flag: "🇬🇧", name: "English" };
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Parse a textarea of one-per-line into a clean, de-duped, non-empty list. */
export function parseLines(text) {
  return [...new Set(text.split(/\n/).map((s) => s.trim()).filter(Boolean))];
}

/** Map a cosine value in [-1,1] to accent opacity over a neutral track for a heatmap cell. */
export function simColor(sim) {
  const t = Math.max(0, Math.min(1, sim));
  return `color-mix(in srgb, var(--accent) ${(t * 100).toFixed(0)}%, transparent)`;
}

/** Render an n×n (or n×m) similarity heatmap. Optionally mark a cell per row as the argmax pick. */
export function renderMatrix(container, matrix, rowLabels, colLabels, { picks = null } = {}) {
  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 0;
  const table = document.createElement("table");
  table.className = "sim-matrix";
  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  hr.appendChild(document.createElement("th"));
  for (let j = 0; j < cols; j++) {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = colLabels[j];
    th.title = colLabels[j];
    hr.appendChild(th);
  }
  thead.appendChild(hr);
  table.appendChild(thead);
  const tb = document.createElement("tbody");
  for (let i = 0; i < rows; i++) {
    const tr = document.createElement("tr");
    const rh = document.createElement("th");
    rh.scope = "row";
    rh.textContent = rowLabels[i];
    rh.title = rowLabels[i];
    tr.appendChild(rh);
    for (let j = 0; j < cols; j++) {
      const td = document.createElement("td");
      const s = matrix[i][j];
      td.textContent = s.toFixed(2);
      td.style.background = simColor(s);
      td.style.color = s > 0.6 ? "var(--accent-ink)" : "var(--color)";
      td.title = `${rowLabels[i]} ↔ ${colLabels[j]}: cosine ${s.toFixed(4)}`;
      if (rows === cols && i === j) td.classList.add("diag");
      if (picks && picks[i] === j) td.classList.add("pick");
      tr.appendChild(td);
    }
    tb.appendChild(tr);
  }
  table.appendChild(tb);
  container.replaceChildren(table);
}

/** Render a 2D PCA scatter into an SVG. `groups` colours points (e.g. by language or cluster). */
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
    const clipped = label.length > 24 ? label.slice(0, 23) + "…" : label;
    return `<g>
      <circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="6" fill="${fill}"></circle>
      <text x="${(cx + 9).toFixed(1)}" y="${
      (cy + 4).toFixed(1)
    }" class="proj-label">${clipped}</text>
    </g>`;
  }).join("");
  container.innerHTML =
    `<svg viewBox="0 0 ${W} ${H}" class="proj-svg" role="img" aria-label="2D PCA projection of the sentence embeddings">
      <line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" class="proj-axis"></line>
      <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${H - pad}" class="proj-axis"></line>
      <text x="${W - pad}" y="${
      H - pad + 20
    }" text-anchor="end" class="proj-axis-label">PC1 →</text>
      <text x="${pad - 6}" y="${pad - 10}" class="proj-axis-label">PC2 ↑</text>
      ${dots}
    </svg>`;
}

/** Render a ranked cross-lingual search result list. `hits` = [{text, score, tag?, langCode?}]. */
export function renderRanked(container, hits, { queryLang = null } = {}) {
  container.replaceChildren(
    ...hits.map((h, rank) => {
      const row = document.createElement("div");
      row.className = "result-row" + (rank === 0 ? " top" : "");
      const head = document.createElement("div");
      head.className = "result-head";
      const t = document.createElement("span");
      t.textContent = `${rank + 1}. ${h.text}`;
      if (h.tag) {
        const tag = document.createElement("span");
        tag.className = "tone";
        if (queryLang && h.langCode && h.langCode !== queryLang) tag.classList.add("xling");
        tag.textContent = h.tag;
        t.append(" ");
        t.append(tag);
      }
      const sc = document.createElement("span");
      sc.className = "result-score";
      sc.textContent = h.score.toFixed(3);
      head.append(t, sc);
      row.append(head);
      const bar = document.createElement("div");
      bar.className = "result-bar";
      const fill = document.createElement("div");
      fill.className = "result-fill";
      fill.style.inlineSize = `${Math.max(0, Math.min(1, h.score)) * 100}%`;
      bar.append(fill);
      row.append(bar);
      return row;
    }),
  );
}

/**
 * Render a bitext-mining alignment table: each source row shows the source sentence, the candidate the
 * model matched it to, whether that is the CORRECT translation (when a gold map is provided), the
 * cosine score, and the margin over the best distractor. `mined` from bitextMine(); `gold[i]` = the
 * correct candidate index for source i (optional).
 */
export function renderBitext(container, sources, candidates, mined, { gold = null } = {}) {
  const table = document.createElement("table");
  table.className = "bitext-table";
  table.innerHTML =
    `<thead><tr><th scope="col">Source</th><th scope="col">Matched translation</th><th scope="col">cosine</th><th scope="col">margin</th></tr></thead>`;
  const tb = document.createElement("tbody");
  for (const m of mined) {
    const tr = document.createElement("tr");
    const srcL = guessLang(sources[m.srcIndex]);
    const candText = candidates[m.bestCand] ?? "";
    const candL = guessLang(candText);
    const correct = gold ? gold[m.srcIndex] === m.bestCand : null;

    const srcTd = document.createElement("td");
    srcTd.innerHTML = `<span class="pill-lang">${srcL.flag}</span> ${
      escapeHTML(sources[m.srcIndex])
    }`;

    const matchTd = document.createElement("td");
    matchTd.innerHTML = `<span class="pill-lang">${candL.flag}</span> ${escapeHTML(candText)}`;
    if (correct === true) matchTd.classList.add("match-ok");
    if (correct === false) matchTd.classList.add("match-bad");
    if (correct === true) {
      matchTd.insertAdjacentHTML("beforeend", ' <span class="badge ok">✓</span>');
    }
    if (correct === false) {
      matchTd.insertAdjacentHTML("beforeend", ' <span class="badge bad">✗</span>');
    }

    const scTd = document.createElement("td");
    scTd.className = "num";
    scTd.textContent = m.best.toFixed(3);

    const mgTd = document.createElement("td");
    mgTd.className = "num";
    mgTd.innerHTML = `<span class="margin-bar"><span style="inline-size:${
      Math.max(0, Math.min(1, m.margin * 2)) * 100
    }%"></span></span> ${m.margin.toFixed(3)}`;

    tr.append(srcTd, matchTd, scTd, mgTd);
    tb.append(tr);
  }
  table.append(tb);
  container.replaceChildren(table);
}

/** Shared inline styles for the LaBSE widgets. Injected once per page. */
export const LABSE_CSS = `
.sim-matrix { border-collapse: collapse; font-family: var(--font-mono); font-size: .8rem; margin-top: .5rem; }
.sim-matrix th, .sim-matrix td {
  border: 1px solid var(--border); padding: .3rem .45rem; text-align: center; min-inline-size: 3rem;
}
.sim-matrix thead th, .sim-matrix tbody th { color: var(--muted); background: var(--bg-raised); font-weight: 600; }
.sim-matrix td.diag { outline: 2px solid var(--border-strong); outline-offset: -2px; }
.sim-matrix td.pick { outline: 2px solid var(--accent); outline-offset: -2px; font-weight: 700; }
.matrix-scroll { overflow-x: auto; }
.proj-svg { inline-size: 100%; max-inline-size: 100%; block-size: auto; background: var(--bg-raised);
  border: 1px solid var(--border); border-radius: 8px; }
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
.result-bar { block-size: .45rem; background: var(--bg-secondary); border: 1px solid var(--border);
  border-radius: 999px; overflow: hidden; margin-top: .3rem; }
.result-fill { block-size: 100%; background: var(--accent); }
.tone { display: inline-block; font-family: var(--font-mono); font-size: .68rem; padding: .1rem .45rem;
  border-radius: 999px; border: 1px solid var(--border); margin-inline-start: .3rem; color: var(--muted); }
.tone.xling { color: var(--good); border-color: var(--good); }
.legend { display: flex; flex-wrap: wrap; gap: .8rem; font-size: .78rem; color: var(--muted);
  font-family: var(--font-mono); margin-top: .5rem; }
.legend .swatch { display: inline-block; inline-size: .8rem; block-size: .8rem; border-radius: 50%;
  margin-inline-end: .3rem; vertical-align: -1px; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; align-items: start; }
.bitext-table { border-collapse: collapse; inline-size: 100%; font-size: .82rem; margin-top: .5rem; }
.bitext-table th, .bitext-table td { border: 1px solid var(--border); padding: .4rem .55rem; text-align: start; vertical-align: top; }
.bitext-table thead th { background: var(--bg-raised); color: var(--muted); font-family: var(--font-mono); font-size: .72rem;
  text-transform: uppercase; letter-spacing: .04em; }
.bitext-table td.num { font-family: var(--font-mono); text-align: end; white-space: nowrap; }
.bitext-table td.match-ok { background: color-mix(in srgb, var(--good) 12%, transparent); }
.bitext-table td.match-bad { background: color-mix(in srgb, var(--bad) 12%, transparent); }
.badge { font-family: var(--font-mono); font-size: .72rem; font-weight: 700; }
.badge.ok { color: var(--good); } .badge.bad { color: var(--bad); }
.margin-bar { display: inline-block; inline-size: 3rem; block-size: .5rem; background: var(--bg-secondary);
  border: 1px solid var(--border); border-radius: 999px; overflow: hidden; vertical-align: middle; margin-inline-end: .3rem; }
.margin-bar span { display: block; block-size: 100%; background: var(--accent); }
.stat-row { display: grid; gap: .6rem; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); margin-top: .6rem; }
.stat { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-secondary); padding: .5rem .7rem; }
.stat .k { font-family: var(--font-mono); font-size: .68rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.stat .v { font-family: var(--font-display); font-size: 1.4rem; }
.chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px; border: 1px solid var(--border);
  background: var(--bg-raised); color: var(--color); cursor: pointer; }
.chip:hover { border-color: var(--accent); }
.pill-lang { font-family: var(--font-mono); font-size: .9rem; }
`;
