// Front-end helpers for the paraphrase-generation pages. Keeps each page thin: it owns the worker
// handshake, streaming, N-paraphrase orchestration, and the render helpers. All inference lives in
// worker.js (off the main thread). Nothing leaves the device.

const WORKER_URL = "/web-ai-showcase/models/paraphrase-generation/worker.js";

export class ParaphraseEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.onProgress = null;
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
      this.onProgress?.(msg.p);
    } else if (msg.type === "ready") {
      this.ready = true;
      this.device = msg.device;
      for (const w of this._loadWaiters) w.resolve(msg.device);
      this._loadWaiters = [];
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

  load(onProgress) {
    if (onProgress) this.onProgress = onProgress;
    if (this.ready) return Promise.resolve(this.device);
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load" });
    });
  }

  /** Generate ONE paraphrase. opts: { maxNewTokens, temperature, topK, topP, diverse, numBeams }.
   *  onStream(partial) fires per token in sampling mode. Resolves with the full result message. */
  paraphrase(input, opts = {}, onStream) {
    const id = ++this._id;
    if (onStream) this._streams.set(id, onStream);
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, input, opts });
    });
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Lowercase word set for lexical comparison (letters/digits only). */
function wordSet(text) {
  return new Set(String(text).toLowerCase().match(/[a-z0-9]+/g) || []);
}

/**
 * LEXICAL overlap (Jaccard over word sets) between two strings, 0..1. This measures how much the WORDING
 * changed — NOT meaning. A good paraphrase keeps meaning while LOWERING lexical overlap (fresh wording).
 * Semantic similarity needs an embedding model (see the multi-model page).
 */
export function lexicalOverlap(a, b) {
  const A = wordSet(a), B = wordSet(b);
  if (!A.size && !B.size) return 1;
  let inter = 0;
  for (const w of A) if (B.has(w)) inter++;
  const union = A.size + B.size - inter;
  return union ? inter / union : 0;
}

/** A short label + colour hint for a lexical-overlap value (lower overlap = more novel wording). */
export function overlapLabel(v) {
  if (v >= 0.7) return { text: "very similar wording", varName: "--warn" };
  if (v >= 0.45) return { text: "moderately reworded", varName: "--color" };
  return { text: "freshly reworded", varName: "--good" };
}

/** Render the SentencePiece token chips (T5 uses ▁ for a leading space; shown as a middot). */
export function renderTokens(container, tokenStrings) {
  container.replaceChildren(...(tokenStrings || []).map((t) => {
    const chip = document.createElement("span");
    chip.className = "tok";
    chip.textContent = t.replace(/▁/g, "·");
    if (/^<.*>$/.test(t)) chip.classList.add("tok-special");
    return chip;
  }));
}

/** Draw the real per-token decode timeline as a bar chart of inter-token intervals (ms). */
export function drawTimeline(canvas, intervals) {
  const cs = getComputedStyle(document.body);
  const accent = cs.getPropertyValue("--accent").trim() || "#4b3aff";
  const muted = cs.getPropertyValue("--muted").trim() || "#888";
  const dpr = self.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600;
  const h = canvas.clientHeight || 90;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!intervals || intervals.length === 0) {
    ctx.fillStyle = muted;
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillText("No per-token timeline (diverse beam mode).", 8, h / 2);
    return;
  }
  const max = Math.max(...intervals, 1);
  const n = intervals.length;
  const bw = Math.max(1, w / n);
  for (let i = 0; i < n; i++) {
    const bh = Math.max(1, (intervals[i] / max) * (h - 6));
    ctx.fillStyle = accent;
    ctx.globalAlpha = 0.85;
    ctx.fillRect(i * bw, h - bh, Math.max(1, bw - 1), bh);
  }
  ctx.globalAlpha = 1;
  ctx.fillStyle = muted;
  ctx.font = "11px system-ui, sans-serif";
  ctx.fillText(`${n} tokens · peak ${max} ms/token`, 6, 12);
}

export const PP_CSS = `
.pp-input { inline-size:100%; font-family:var(--font-body); font-size:1rem; }
.controls-grid { display:grid; gap:.9rem 1.2rem; grid-template-columns:repeat(auto-fit, minmax(160px, 1fr));
  align-items:end; margin:.6rem 0; }
.controls-grid label { display:flex; flex-direction:column; gap:.3rem; font-size:.82rem; }
.controls-grid input[type=range] { inline-size:100%; }
.controls-grid .val { font-family:var(--font-mono); color:var(--muted); font-size:.78rem; }
.pp-actions { display:flex; flex-wrap:wrap; gap:.6rem; align-items:center; }
.pp-list { display:grid; gap:.7rem; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); margin-top:.6rem; }
.pp-card { border:1px solid var(--border); border-radius:var(--radius); background:var(--bg-raised);
  padding:.7rem .8rem; display:flex; flex-direction:column; gap:.5rem; min-inline-size:0; }
.pp-card .n { font-family:var(--font-mono); font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
.pp-card .txt { white-space:pre-wrap; word-break:break-word; min-block-size:2.4rem; }
.pp-card .meta { display:flex; flex-wrap:wrap; gap:.5rem 1rem; align-items:center; font-family:var(--font-mono); font-size:.72rem; color:var(--muted); margin-top:auto; }
.pp-card .meta b { color:var(--color); font-weight:600; }
.pp-copy { font:inherit; font-size:.74rem; padding:.25rem .6rem; border-radius:6px; border:1px solid var(--border);
  background:var(--bg-secondary); color:var(--color); cursor:pointer; }
.pp-copy:hover { border-color:var(--accent); }
.pp-copy:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.orig-box { border:1px dashed var(--border-strong); border-radius:8px; background:var(--bg-secondary);
  padding:.5rem .7rem; font-size:.9rem; white-space:pre-wrap; word-break:break-word; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem; color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.tok-wrap { display:flex; flex-wrap:wrap; gap:3px; margin-top:.4rem; }
.tok { font-family:var(--font-mono); font-size:.75rem; padding:.1rem .35rem; border-radius:4px; border:1px solid var(--border); background:var(--bg-raised); }
.tok-special { color:var(--muted); border-style:dashed; }
.timeline { inline-size:100%; block-size:90px; display:block; }
.preset-grid { display:grid; gap:.5rem; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); margin:.5rem 0; }
.preset { text-align:start; border:1px solid var(--border); border-radius:8px; background:var(--bg-raised); padding:.5rem .65rem; cursor:pointer; font:inherit; font-size:.85rem; }
.preset:hover { border-color:var(--accent); }
.seg { display:inline-flex; border:1px solid var(--border); border-radius:999px; overflow:hidden; }
.seg button { font:inherit; font-size:.8rem; padding:.35rem .8rem; background:var(--bg-raised); border:0; color:var(--color); cursor:pointer; }
.seg button[aria-pressed=true] { background:var(--accent); color:var(--accent-ink); }
`;
