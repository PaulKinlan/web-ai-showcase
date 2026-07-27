// Front-end helpers for the OPUS-MT Dutch→English pages. Owns the worker handshake, streaming, and
// render helpers. All inference lives in worker.js (off the main thread).

const WORKER_URL = "/web-ai-showcase/models/opus-mt-nl-en/worker.js";

// The direction this demo ships: one dedicated bilingual model, Dutch → English. The reverse pair
// exists in the worker too (for the round-trip wild demo) but isn't the headline model.
export const PAIR_NAME = {
  "nl-en": "Dutch → English",
  "en-nl": "English → Dutch",
};
export const DEFAULT_PAIR = "nl-en";

export class NlEnEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.onProgress = null; // loader progress — ONLY used for the primary model, before ready
    this.onPairLoading = null; // (pair) => void — a pair started downloading
    this.onPairProgress = null; // (p, pair) => void — on-demand pair download progress (post-ready)
    this.loadedPairs = new Set();
    this._loadWaiters = [];
    this._pending = new Map();
    this._streams = new Map();
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
      // Before the primary model is ready, progress drives the shared loader UI. After ready, an
      // on-demand pair download must NOT overwrite the loader's "ready" status — route it elsewhere.
      if (!this.ready) this.onProgress?.(msg.p, msg.pair);
      else this.onPairProgress?.(msg.p, msg.pair);
    } else if (msg.type === "pair-loading") {
      this.onPairLoading?.(msg.pair);
    } else if (msg.type === "pair-ready") {
      this.loadedPairs.add(msg.pair);
    } else if (msg.type === "ready") {
      this.ready = true;
      this.device = msg.device;
      for (const w of this._loadWaiters) w.resolve(msg.device);
      this._loadWaiters = [];
    } else if (msg.type === "ensured") {
      this._pending.get(msg.id)?.resolve(msg);
      this._pending.delete(msg.id);
    } else if (msg.type === "stream") {
      this._streams.get(msg.id)?.(msg.text);
    } else if (msg.type === "result") {
      const p = this._pending.get(msg.id);
      if (p) {
        this._pending.delete(msg.id);
        this._streams.delete(msg.id);
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

  load(onProgress, pair = DEFAULT_PAIR) {
    if (onProgress) this.onProgress = onProgress;
    if (this.ready) return Promise.resolve(this.device);
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load", pair });
    });
  }

  /** Ensure a pair's model is loaded (downloads on demand). Resolves when ready. */
  ensurePair(pair) {
    if (this.loadedPairs.has(pair)) return Promise.resolve();
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "ensure", id, pair });
    });
  }

  /** Translate text with a given pair. opts: { numBeams }. onStream(partial). */
  translate(text, pair, opts = {}, onStream) {
    const id = ++this._id;
    if (onStream) this._streams.set(id, onStream);
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, text, pair, opts });
    });
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Render the pair/token/timing readout for a completed translation. */
export function renderStats(els, r) {
  if (els.pair) els.pair.textContent = PAIR_NAME[r.pair] || r.pair;
  if (els.tok) els.tok.textContent = `${r.inTokens} → ${r.outTokens}`;
  if (els.inTok) els.inTok.textContent = r.inTokens;
  if (els.outTok) els.outTok.textContent = r.outTokens;
  if (els.backend) els.backend.textContent = r.device.toUpperCase();
  if (els.ms) els.ms.textContent = (r.ms / 1000).toFixed(2) + " s";
  if (els.toksec) {
    const tps = r.ms > 0 ? (r.outTokens / (r.ms / 1000)) : 0;
    els.toksec.textContent = tps ? tps.toFixed(1) + " tok/s" : "–";
  }
}

export const NLMT_CSS = `
.tr-io { display:grid; gap:1rem; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); }
.tr-io textarea { inline-size:100%; min-block-size:130px; resize:vertical; font-family:var(--font-body); }
.tr-out { border:1px solid var(--border); border-radius:var(--radius); background:var(--bg-raised);
  padding:.8rem 1rem; min-block-size:130px; white-space:pre-wrap; font-size:1.05rem; }
.tr-out:empty::before { content:"The translation will stream in here."; color:var(--muted); font-size:.95rem; }
.controls-grid { display:grid; gap:.9rem 1.2rem; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); align-items:end; margin:.6rem 0; }
.controls-grid label { display:flex; flex-direction:column; gap:.3rem; font-size:.82rem; }
.stat-row { display:grid; gap:.6rem; grid-template-columns:repeat(auto-fit, minmax(120px, 1fr)); margin-top:.6rem; }
.stat { border:1px solid var(--border); border-radius:8px; background:var(--bg-secondary); padding:.5rem .7rem; }
.stat .k { font-family:var(--font-mono); font-size:.66rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
.stat .v { font-family:var(--font-display); font-size:1.3rem; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem; color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.chip { font:inherit; font-size:.78rem; padding:.2rem .6rem; border-radius:999px; border:1px solid var(--border);
  background:var(--bg-raised); color:var(--color); cursor:pointer; }
.chip:hover { border-color:var(--accent); }
.strings-table { inline-size:100%; border-collapse:collapse; font-size:.9rem; margin-top:.6rem; }
.strings-table th, .strings-table td { text-align:start; padding:.45rem .5rem; border-block-end:1px solid var(--border); vertical-align:top; }
.strings-table th { font-family:var(--font-mono); font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
.drift-flag { font-family:var(--font-mono); font-size:.72rem; } .drift-ok { color:var(--good); } .drift-warn { color:var(--warn); }
`;
