// Front-end helpers for the BART-large-CNN summarization pages. Keeps each page thin: it owns the
// worker handshake, streaming, and the render helpers. All inference lives in worker.js (off-thread).

const WORKER_URL = "/web-ai-showcase/models/bart-large-cnn-summarization/worker.js";

export class BartEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.onProgress = null;
    this._loadWaiters = [];
    this._pending = new Map();
    this._streams = new Map();
    this._recapSteps = new Map();
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
    } else if (msg.type === "recap-step") {
      this._recapSteps.get(msg.id)?.(msg.step);
    } else if (msg.type === "result" || msg.type === "recap-done") {
      const p = this._pending.get(msg.id);
      if (p) {
        this._pending.delete(msg.id);
        this._streams.delete(msg.id);
        this._recapSteps.delete(msg.id);
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

  /** Summarize one text. opts: { maxLength, minLength, numBeams, lengthPenalty }. onStream(partial). */
  summarize(text, opts = {}, onStream) {
    const id = ++this._id;
    if (onStream) this._streams.set(id, onStream);
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, text, opts });
    });
  }

  /** Recursively summarize toward one sentence. onStep({round,inTokens,outTokens,summary,ms}). */
  recap(text, opts = {}, onStep) {
    const id = ++this._id;
    if (onStep) this._recapSteps.set(id, onStep);
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "recap", id, text, opts });
    });
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Split a summary into sentence bullets (for bullet-point mode — post-hoc shaping, honestly labelled). */
export function toBullets(summary) {
  return summary
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 1);
}

/** Render the token/compression readout for a completed summary. */
export function renderStats(els, r) {
  const compTok = r.inTokens > 0 ? (r.inTokens / Math.max(1, r.outTokens)) : 0;
  const compChar = r.inChars > 0 ? (1 - r.outChars / r.inChars) * 100 : 0;
  if (els.inTok) els.inTok.textContent = r.inTokens;
  if (els.outTok) els.outTok.textContent = r.outTokens;
  if (els.ratio) els.ratio.textContent = compTok ? compTok.toFixed(1) + "×" : "–";
  if (els.reduction) els.reduction.textContent = compChar ? compChar.toFixed(0) + "%" : "–";
  if (els.backend) els.backend.textContent = r.device.toUpperCase();
  if (els.ms) els.ms.textContent = (r.ms / 1000).toFixed(2) + " s";
  if (els.toksec) {
    const tps = r.ms > 0 ? (r.outTokens / (r.ms / 1000)) : 0;
    els.toksec.textContent = tps ? tps.toFixed(1) + " tok/s" : "–";
  }
}

/**
 * Draw the real per-token decode timeline as a bar chart of inter-token intervals (ms).
 * Longer bars = tokens that took longer to generate. Empty for beam search (no clean per-token order).
 */
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
    ctx.fillText("No per-token timeline (beam search).", 8, h / 2);
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

export const BART_CSS = `
.sum-io { display:grid; gap:1rem; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); }
.sum-io textarea { inline-size:100%; min-block-size:200px; resize:vertical; font-family:var(--font-body); }
.sum-out { border:1px solid var(--border); border-radius:var(--radius); background:var(--bg-raised);
  padding:.8rem 1rem; min-block-size:200px; white-space:pre-wrap; }
.sum-out:empty::before { content:"The summary will stream in here."; color:var(--muted); }
.sum-out ul { margin:.2rem 0; padding-inline-start:1.2rem; }
.sum-out li { margin:.25rem 0; }
.controls-grid { display:grid; gap:.9rem 1.2rem; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));
  align-items:end; margin:.6rem 0; }
.controls-grid label { display:flex; flex-direction:column; gap:.3rem; font-size:.82rem; }
.controls-grid input[type=range] { inline-size:100%; }
.controls-grid .val { font-family:var(--font-mono); color:var(--muted); font-size:.78rem; }
.stat-row { display:grid; gap:.6rem; grid-template-columns:repeat(auto-fit, minmax(120px, 1fr)); margin-top:.6rem; }
.stat { border:1px solid var(--border); border-radius:8px; background:var(--bg-secondary); padding:.5rem .7rem; }
.stat .k { font-family:var(--font-mono); font-size:.68rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
.stat .v { font-family:var(--font-display); font-size:1.4rem; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem; color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.chip { font:inherit; font-size:.78rem; padding:.35rem .7rem; border-radius:999px; border:1px solid var(--border);
  background:var(--bg-raised); color:var(--color); cursor:pointer; }
.chip:hover { border-color:var(--accent); }
.timeline { inline-size:100%; block-size:90px; display:block; }
.seg-toggle { display:inline-flex; border:1px solid var(--border); border-radius:999px; overflow:hidden; }
.seg-toggle button { border:0; border-radius:0; background:var(--bg-raised); padding:.35rem .8rem; font-size:.8rem; }
.seg-toggle button[aria-pressed=true] { background:var(--accent); color:var(--accent-ink); }
.recap-chain { display:flex; flex-direction:column; gap:.5rem; margin-top:.6rem; }
.recap-step { border:1px solid var(--border); border-inline-start:4px solid var(--accent); border-radius:8px;
  background:var(--bg-raised); padding:.5rem .7rem; }
.recap-step .meta { font-family:var(--font-mono); font-size:.72rem; color:var(--muted); }
.recap-step .txt { margin-top:.2rem; }
.cmp-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:1rem; margin-top:.6rem; align-items:start; }
.cmp-card { border:1px solid var(--border); border-radius:10px; padding:.8rem; background:var(--bg-raised); min-inline-size:0; }
.cmp-card h4 { margin:0 0 .2rem; font-family:var(--font-display); }
.cmp-card .cmp-sub { font-family:var(--font-mono); font-size:.72rem; color:var(--muted); margin:0 0 .5rem; }
.cmp-card .cmp-body { white-space:pre-wrap; }
`;
