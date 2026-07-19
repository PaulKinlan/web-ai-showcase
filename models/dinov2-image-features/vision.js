// Shared front-end helpers for the DINOv2 image-feature pages. Keeps each page thin: worker handshake,
// light client-side maths (cosine, similarity matrix, clustering), the renderers (thumbnail matrix,
// patch-feature heatmap), image loading (samples + upload → data URL), and the injected CSS.
// All model inference lives in worker.js (off the main thread).

const WORKER_URL = "/web-ai-showcase/models/dinov2-image-features/worker.js";

export class VisionEngine {
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

  /** Embed one image (URL or data URL) → { clsEmb, dim, clsPreNorm, patchSims, gridSize, ms, device } */
  embed(imageURL) {
    // Resolve page-relative sample paths to ABSOLUTE against the page's base URL. The worker lives at a
    // different URL, so a relative path (e.g. "../samples/cat.jpg") would otherwise resolve against the
    // worker's location inside RawImage.read and 404. data:/blob:/http(s) URLs pass through unchanged.
    const image = /^(data:|blob:|https?:)/.test(imageURL)
      ? imageURL
      : new URL(imageURL, document.baseURI).href;
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, image });
    });
  }
}

/** Cosine similarity between two unit vectors → dot product. */
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

/** Nearest neighbour of each item (excluding itself) → [{index, score}]. */
export function nearest(embeddings, i) {
  let best = -1, bestS = -Infinity;
  for (let j = 0; j < embeddings.length; j++) {
    if (j === i) continue;
    const s = cosine(embeddings[i], embeddings[j]);
    if (s > bestS) {
      bestS = s;
      best = j;
    }
  }
  return { index: best, score: bestS };
}

/** Greedy single-link clustering by cosine threshold (union-find) → cluster id per item. */
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

function simColor(sim) {
  const t = Math.max(0, Math.min(1, sim));
  return `color-mix(in srgb, var(--accent) ${(t * 100).toFixed(0)}%, transparent)`;
}

/**
 * Render an n×n image-similarity matrix with thumbnails on the axes. `items` = [{src, name}].
 */
export function renderImageMatrix(container, matrix, items) {
  const n = matrix.length;
  const table = document.createElement("table");
  table.className = "img-matrix";
  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  hr.appendChild(document.createElement("th"));
  for (let j = 0; j < n; j++) {
    const th = document.createElement("th");
    th.scope = "col";
    th.innerHTML = `<img src="${items[j].src}" alt="${
      escapeAttr(items[j].name)
    }" class="mx-thumb" />`;
    hr.appendChild(th);
  }
  thead.appendChild(hr);
  table.appendChild(thead);
  const tb = document.createElement("tbody");
  for (let i = 0; i < n; i++) {
    const tr = document.createElement("tr");
    const rh = document.createElement("th");
    rh.scope = "row";
    rh.innerHTML = `<img src="${items[i].src}" alt="${
      escapeAttr(items[i].name)
    }" class="mx-thumb" />`;
    tr.appendChild(rh);
    for (let j = 0; j < n; j++) {
      const td = document.createElement("td");
      const s = matrix[i][j];
      td.textContent = s.toFixed(2);
      td.style.background = simColor(s);
      td.style.color = s > 0.6 ? "var(--accent-ink)" : "var(--color)";
      td.title = `${items[i].name} ↔ ${items[j].name}: cosine ${s.toFixed(4)}`;
      if (i === j) td.classList.add("diag");
      tr.appendChild(td);
    }
    tb.appendChild(tr);
  }
  table.appendChild(tb);
  container.replaceChildren(table);
}

/**
 * Render the patch-feature heatmap: a gridSize×gridSize grid of each patch's cosine to the CLS token,
 * overlaid semi-transparently. Returns an element the caller can place over the source thumbnail.
 */
export function renderPatchGrid(container, patchSims, gridSize) {
  const lo = Math.min(...patchSims), hi = Math.max(...patchSims);
  const span = hi - lo || 1;
  const cells = patchSims.map((s) => {
    const t = (s - lo) / span; // normalize within this image for contrast
    return `<div class="pg-cell" style="background:color-mix(in srgb, var(--accent) ${
      (t * 100).toFixed(0)
    }%, transparent)" title="cosine to CLS: ${s.toFixed(3)}"></div>`;
  }).join("");
  container.innerHTML =
    `<div class="pg-grid" role="img" aria-label="Patch feature heatmap: each cell is a patch's similarity to the global descriptor" style="grid-template-columns:repeat(${gridSize},1fr)">${cells}</div>`;
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}
function escapeAttr(s) {
  return String(s).replace(
    /["&<>]/g,
    (c) => ({ '"': "&quot;", "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]),
  );
}

/** Read a File → data URL (works across the main thread → worker boundary). */
export function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result);
    fr.onerror = () => reject(fr.error);
    fr.readAsDataURL(file);
  });
}

/** The bundled sample images (shipped alongside the pages), grouped for the similarity demos. */
export const SAMPLES = [
  { src: "../samples/cat-1.jpg", name: "cat A", group: "cat" },
  { src: "../samples/cat-2.jpg", name: "cat B", group: "cat" },
  { src: "../samples/car-1.jpg", name: "car A", group: "car" },
  { src: "../samples/car-2.jpg", name: "car B", group: "car" },
  { src: "../samples/mountain-1.jpg", name: "mountain", group: "landscape" },
  { src: "../samples/flower-1.jpg", name: "flower", group: "flower" },
];

export const VISION_CSS = `
.img-matrix { border-collapse: collapse; font-family: var(--font-mono); font-size: .78rem; margin-top: .5rem; }
.img-matrix th, .img-matrix td { border: 1px solid var(--border); padding: .25rem; text-align: center; }
.img-matrix td { min-inline-size: 3rem; }
.img-matrix td.diag { outline: 2px solid var(--border-strong); outline-offset: -2px; }
.mx-thumb { inline-size: 44px; block-size: 44px; object-fit: cover; border-radius: 6px; display: block; }
.matrix-scroll { overflow-x: auto; }
.img-tray { display: flex; flex-wrap: wrap; gap: .6rem; margin-top: .5rem; }
.img-tile { position: relative; inline-size: 96px; border: 1px solid var(--border); border-radius: 8px;
  overflow: hidden; background: var(--bg-raised); }
.img-tile img { inline-size: 96px; block-size: 96px; object-fit: cover; display: block; }
.img-tile figcaption { font-size: .7rem; color: var(--muted); text-align: center; padding: .2rem;
  font-family: var(--font-mono); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.img-tile .rm { position: absolute; inset-block-start: 2px; inset-inline-end: 2px; inline-size: 20px;
  block-size: 20px; border-radius: 50%; border: none; background: rgb(0 0 0 / .55); color: #fff;
  cursor: pointer; font-size: .8rem; line-height: 1; }
.dropzone { border: 2px dashed var(--border-strong); border-radius: 10px; padding: 1rem; text-align: center;
  color: var(--muted); font-size: .85rem; cursor: pointer; margin-top: .6rem; }
.dropzone.drag { border-color: var(--accent); color: var(--color); background: var(--bg-raised); }
.pg-grid { display: grid; gap: 1px; inline-size: 100%; aspect-ratio: 1; border: 1px solid var(--border);
  border-radius: 8px; overflow: hidden; }
.pg-cell { inline-size: 100%; block-size: 100%; }
.pg-wrap { position: relative; inline-size: 224px; max-inline-size: 100%; }
.pg-wrap > img { inline-size: 100%; block-size: auto; border-radius: 8px; display: block; }
.pg-wrap > .pg-over { position: absolute; inset: 0; opacity: .68; }
.sent-list { display: flex; flex-direction: column; gap: .5rem; margin-top: .5rem; }
.result-row { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised); padding: .55rem .7rem; }
.result-head { display: flex; justify-content: space-between; gap: .5rem; align-items: center; }
.result-score { font-family: var(--font-mono); color: var(--muted); font-size: .8rem; white-space: nowrap; }
.tone { display: inline-block; font-family: var(--font-mono); font-size: .68rem; padding: .1rem .45rem;
  border-radius: 999px; border: 1px solid var(--border); margin-inline-start: .4rem; color: var(--muted); }
.tone-neg { color: var(--bad); border-color: var(--bad); }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.vec-strip { display: flex; flex-wrap: wrap; gap: 2px; margin-top: .4rem; }
.vec-cell { inline-size: 12px; block-size: 20px; border-radius: 2px; border: 1px solid var(--border); }
.grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; align-items: start; }
`;
