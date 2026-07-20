// Front-end helpers shared by every Nomic multimodal-embedding page. Owns the worker handshake, turns
// files/samples into data URLs, embeds images + text into a SHARED space, and renders ranked results.
// All inference lives in worker.js (off the main thread); embeddings come back L2-normalised so a
// cosine similarity is just a dot product on the main thread (cheap, 768-d).
//
// Models: nomic-ai/nomic-embed-vision-v1.5 (image tower) + nomic-ai/nomic-embed-text-v1.5 (text tower),
// aligned into one 768-d space — CLIP-style cross-modal retrieval from the Nomic family.

const WORKER_URL = "/web-ai-showcase/models/nomic-embed-vision-v1-5/worker.js";

export class NomicEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.onProgress = null;
    this._loadWaiters = [];
    this._pending = new Map();
    this._id = 0;
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
    } else if (msg.type === "imageEmbedding" || msg.type === "textEmbedding") {
      const p = this._pending.get(msg.id);
      if (p) {
        this._pending.delete(msg.id);
        p.resolve({
          embedding: new Float32Array(msg.embedding),
          dim: msg.dim,
          ms: msg.ms,
          key: msg.key,
          device: msg.device,
        });
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

  embedImage(url, key) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "embedImage", id, url, key });
    });
  }

  embedText(text, isQuery = true) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "embedText", id, text, isQuery });
    });
  }
}

/** Cosine similarity between two L2-normalised vectors = dot product. */
export function cosine(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}

export async function urlToDataURL(src) {
  const blob = await (await fetch(src)).blob();
  return fileToDataURL(new File([blob], "sample", { type: blob.type }));
}

export function escapeHTML(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]),
  );
}

export const NOMIC_CSS = `
.dropzone { border: 2px dashed var(--border-strong); border-radius: var(--radius); background: var(--bg-raised); padding: .8rem; text-align: center; cursor: pointer; transition: border-color .15s, background .15s; }
.dropzone.drag { border-color: var(--accent); background: var(--bg-secondary); }
.dropzone:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.query-row { display: flex; flex-wrap: wrap; gap: .6rem; align-items: end; margin: .6rem 0; }
.query-row label { display: flex; flex-direction: column; gap: .25rem; font-size: .82rem; flex: 1 1 260px; }
.query-row input[type=text] { inline-size: 100%; font: inherit; padding: .45rem .55rem; border-radius: 8px; border: 1px solid var(--border-strong); background: var(--bg-raised); color: var(--color); }
.chip { font: inherit; font-size: .78rem; padding: .3rem .7rem; min-block-size: 40px; border-radius: 999px; border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.chip:hover { border-color: var(--accent); }
.chip:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.chip-wrap { display: flex; flex-wrap: wrap; gap: .4rem; margin: .4rem 0; }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: .7rem; margin: .6rem 0; }
.tile { position: relative; border: 2px solid transparent; border-radius: 10px; overflow: hidden; background: var(--bg-raised); cursor: pointer; padding: 0; display: block; inline-size: 100%; }
.tile img { display: block; inline-size: 100%; block-size: 120px; object-fit: cover; }
.tile:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.tile.query { border-color: var(--accent); }
.tile.top { border-color: var(--accent); }
.tile .score { position: absolute; inset-block-end: 0; inset-inline: 0; background: color-mix(in srgb, var(--bg) 72%, transparent); font-family: var(--font-mono); font-size: .72rem; padding: .12rem .3rem; text-align: right; }
.tile .rank { position: absolute; inset-block-start: .25rem; inset-inline-start: .25rem; background: var(--accent); color: #fff; font-family: var(--font-mono); font-size: .68rem; inline-size: 1.3rem; block-size: 1.3rem; border-radius: 50%; display: grid; place-items: center; }
.tile .cap { position: absolute; inset-block-start: .25rem; inset-inline-end: .25rem; background: color-mix(in srgb, var(--bg) 72%, transparent); font-size: .66rem; padding: .05rem .3rem; border-radius: 5px; }
.bars { display: flex; flex-direction: column; gap: .5rem; margin-top: .4rem; }
.bar-head { display: flex; justify-content: space-between; gap: .5rem; font-size: .82rem; }
.bar-val { font-family: var(--font-mono); color: var(--muted); }
.bar-track { block-size: .65rem; background: var(--bg-raised); border: 1px solid var(--border); border-radius: 999px; overflow: hidden; margin-top: .12rem; }
.bar-fill { block-size: 100%; background: var(--muted); border-radius: 999px; transition: inline-size .35s ease; }
.bar-top .bar-fill { background: var(--accent); } .bar-top .bar-label { font-weight: 600; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.emb-strip { display: flex; block-size: 34px; border-radius: 6px; overflow: hidden; border: 1px solid var(--border); margin-top: .3rem; }
.emb-strip span { flex: 1 1 0; }
.credit { font-size: .72rem; color: var(--muted); margin: .3rem 0 0; }
.inside-table { inline-size: 100%; max-inline-size: 100%; table-layout: fixed; border-collapse: collapse; font-size: .82rem; margin-top: .5rem; }
.inside-table th, .inside-table td { text-align: left; padding: .3rem .5rem; border-bottom: 1px solid var(--border); font-family: var(--font-mono); overflow-wrap: anywhere; word-break: break-word; }
.inside-table th { color: var(--muted); font-weight: 600; }
.glance-table th:first-child { inline-size: 7.5rem; }
.seg { display: inline-flex; border: 1px solid var(--border-strong); border-radius: 999px; overflow: hidden; }
.seg button { font: inherit; font-size: .8rem; padding: .4rem .8rem; min-block-size: 40px; border: 0; background: var(--bg-raised); color: var(--color); cursor: pointer; }
.seg button[aria-pressed=true] { background: var(--accent); color: #fff; }
`;

/**
 * Render a signed embedding vector as a compact diverging colour strip (blue negative / warm positive).
 * A quick "what does the vector look like" glance — no meaning per dimension, just a fingerprint.
 */
export function embStripHTML(vec, bins = 64) {
  const step = Math.max(1, Math.floor(vec.length / bins));
  let html = "";
  for (let i = 0; i < vec.length; i += step) {
    let m = 0;
    for (let j = i; j < Math.min(i + step, vec.length); j++) m += vec[j];
    m /= step;
    const v = Math.max(-1, Math.min(1, m * 8));
    const col = v >= 0 ? `hsl(28 80% ${60 - v * 22}%)` : `hsl(215 70% ${60 + v * 22}%)`;
    html += `<span style="background:${col}"></span>`;
  }
  return html;
}
