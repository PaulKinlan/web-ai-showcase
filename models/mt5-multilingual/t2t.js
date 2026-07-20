// Front-end helpers for the mT5-small multilingual span-corruption pages. Keeps each page thin: it owns
// the worker handshake, streaming, and the render helpers. All inference lives in worker.js (off thread).

const WORKER_URL = "/web-ai-showcase/models/mt5-multilingual/worker.js";

export class Mt5Engine {
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

  /** Run one masked input. opts: { maxNewTokens, numBeams, doSample, temperature, topK }. onStream(partial). */
  run(input, opts = {}, onStream) {
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

/** Render the input token strings as chips, exposing mT5's SentencePiece ▁ word-boundary marks and the
 *  sentinel <extra_id_n> mask tokens (highlighted — they are the whole point of span corruption). */
export function renderTokens(container, tokenStrings) {
  container.replaceChildren(...(tokenStrings || []).map((t) => {
    const chip = document.createElement("span");
    chip.className = "tok";
    chip.textContent = t.replace(/▁/g, "·");
    if (/^<extra_id_\d+>$/.test(t)) chip.classList.add("tok-sentinel");
    else if (/^(<|>)/.test(t) || /^<.*>$/.test(t)) chip.classList.add("tok-special");
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

/** Fill a readout row of {backend, ms, toksec, inTok, outTok} elements from a result. */
export function renderStats(els, r) {
  if (els.backend) els.backend.textContent = r.device.toUpperCase();
  if (els.ms) els.ms.textContent = (r.ms / 1000).toFixed(2) + " s";
  if (els.toksec) {
    const tps = r.ms > 0 ? (r.outTokens / (r.ms / 1000)) : 0;
    els.toksec.textContent = tps ? tps.toFixed(1) + " tok/s" : "–";
  }
  if (els.inTok) els.inTok.textContent = r.inTokens;
  if (els.outTok) els.outTok.textContent = r.outTokens;
}

export const MT5_CSS = `
.mt5-out { border:1px solid var(--border); border-radius:var(--radius); background:var(--bg-raised);
  padding:.8rem 1rem; min-block-size:52px; white-space:pre-wrap; word-break:break-word; font-family:var(--font-mono); font-size:.9rem; }
.mt5-out:empty::before { content:"The sentinel-delimited fills will stream in here."; color:var(--muted); font-family:var(--font-body); }
.mt5-recon { border:1px solid var(--border); border-radius:var(--radius); background:var(--bg-secondary);
  padding:.7rem 1rem; min-block-size:40px; white-space:pre-wrap; word-break:break-word; margin-top:.4rem; }
.mt5-recon mark { background:color-mix(in srgb, var(--accent) 22%, transparent); color:inherit;
  padding:.05rem .25rem; border-radius:4px; font-weight:600; }
.fed-input { border:1px dashed var(--border-strong); border-radius:8px; background:var(--bg-secondary);
  padding:.5rem .7rem; font-family:var(--font-mono); font-size:.82rem; white-space:pre-wrap; word-break:break-word; }
.controls-grid { display:grid; gap:.9rem 1.2rem; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr));
  align-items:end; margin:.6rem 0; }
.controls-grid label { display:flex; flex-direction:column; gap:.3rem; font-size:.82rem; }
.controls-grid input[type=range] { inline-size:100%; }
.controls-grid .val { font-family:var(--font-mono); color:var(--muted); font-size:.78rem; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem; color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.stat-row { display:grid; gap:.6rem; grid-template-columns:repeat(auto-fit, minmax(120px, 1fr)); margin-top:.6rem; }
.stat { border:1px solid var(--border); border-radius:8px; background:var(--bg-secondary); padding:.5rem .7rem; }
.stat .k { font-family:var(--font-mono); font-size:.68rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
.stat .v { font-family:var(--font-display); font-size:1.4rem; }
.chip { font:inherit; font-size:.78rem; padding:.2rem .6rem; border-radius:999px; border:1px solid var(--border);
  background:var(--bg-raised); color:var(--color); cursor:pointer; }
.chip:hover { border-color:var(--accent); }
.tok-wrap { display:flex; flex-wrap:wrap; gap:3px; margin-top:.4rem; }
.tok { font-family:var(--font-mono); font-size:.75rem; padding:.1rem .35rem; border-radius:4px;
  border:1px solid var(--border); background:var(--bg-raised); }
.tok-special { color:var(--muted); border-style:dashed; }
.tok-sentinel { background:color-mix(in srgb, var(--accent) 20%, transparent); border-color:var(--accent);
  color:var(--color); font-weight:600; }
.timeline { inline-size:100%; block-size:90px; display:block; }
.preset-grid { display:grid; gap:.6rem; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); margin:.6rem 0; }
.preset { text-align:start; border:1px solid var(--border); border-radius:8px; background:var(--bg-raised);
  padding:.55rem .7rem; cursor:pointer; font:inherit; }
.preset:hover { border-color:var(--accent); }
.preset .t { font-weight:600; font-size:.9rem; } .preset .d { color:var(--muted); font-size:.78rem; font-family:var(--font-mono); }
`;
