// Front-end helpers for the AI-text-detection pages. Keeps each page thin: it owns the worker handshake
// and the verdict/meter renderers. All inference lives in worker.js (off the main thread) so typing and
// batch triage never jank.

const WORKER_URL = "/web-ai-showcase/models/ai-text-detection/worker.js";

export class AIDetectEngine {
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
    } else if (msg.type === "result" || msg.type === "batch") {
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

  /** Score one text → { ai, real, label, scores, ms, device }. Softmax: ai + real ≈ 1. */
  classify(text) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, text });
    });
  }

  /** Score many texts in one pass → { results:[{ai,real,label}…], ms, device }. */
  classifyBatch(texts) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "batch", id, texts });
    });
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

export function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

// Honest wording, everywhere: the model's own labels are "Fake"/"Real", but what they MEAN is narrow —
// "matches / doesn't match the GPT-2-1.5B-vs-WebText distribution this detector was trained on".
export function labelText(label) {
  return label === "Fake" ? "reads machine-written" : "reads human-written";
}

/** Render the verdict + a single meter (AI-likeness fraction). `els` = {label, conf, fill}. */
export function renderVerdict(els, { ai, label }) {
  els.label.textContent = labelText(label);
  els.label.className = "verdict-label " + (label === "Fake" ? "ai" : "human");
  els.conf.textContent = ((label === "Fake" ? ai : 1 - ai) * 100).toFixed(1) + "%";
  if (els.fill) els.fill.style.inlineSize = (ai * 100).toFixed(1) + "%";
}

export const AIDETECT_CSS = `
.verdict-row { display:flex; align-items:baseline; gap:.8rem; flex-wrap:wrap; margin-top:.6rem; }
.verdict-label { font-family:var(--font-display); font-size:1.8rem; }
.verdict-label.ai { color:var(--warn); }
.verdict-label.human { color:var(--good); }
.verdict-conf { font-family:var(--font-mono); color:var(--muted); font-size:.9rem; }
.ai-meter { position:relative; block-size:.85rem; border:1px solid var(--border); border-radius:999px;
  overflow:hidden; margin-top:.5rem; background:var(--bg-raised); }
.ai-fill { display:block; block-size:100%; background:var(--warn); border-radius:999px; transition:inline-size .18s ease; }
.ai-mid { position:absolute; inset-block:0; inset-inline-start:50%; inline-size:1px; background:var(--border-strong); }
.meter-labels { display:flex; justify-content:space-between; font-family:var(--font-mono); font-size:.72rem;
  color:var(--muted); margin-top:.2rem; }
.prob-list { display:flex; flex-direction:column; gap:.32rem; margin-top:.6rem; max-inline-size:520px; }
.prob-row { display:grid; grid-template-columns:11rem 1fr 3.4rem; gap:.55rem; align-items:center; }
.prob-name { font-size:.85rem; color:var(--muted); }
.prob-bar { block-size:.72rem; background:var(--bg-secondary); border:1px solid var(--border); border-radius:999px; overflow:hidden; min-inline-size:0; }
.prob-fill { display:block; block-size:100%; border-radius:999px; transition:inline-size .18s ease; }
.prob-score { font-family:var(--font-mono); font-size:.76rem; color:var(--muted); text-align:end; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem; color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.chip { font:inherit; font-size:.78rem; padding:.3rem .7rem; border-radius:999px; border:1px solid var(--border);
  background:var(--bg-raised); color:var(--color); cursor:pointer; min-block-size:2.2rem; }
.chip:hover { border-color:var(--accent); }
.controls-grid { display:grid; gap:.9rem 1.2rem; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); align-items:end; margin:.6rem 0; }
.controls-grid label { display:flex; flex-direction:column; gap:.3rem; font-size:.82rem; }
.pair-grid { display:grid; gap:1rem; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); margin-top:.6rem; }
.pair { border:1px solid var(--border); border-radius:8px; background:var(--bg-secondary); padding:.7rem .85rem; }
.pair .t { font-size:.95rem; margin-bottom:.4rem; overflow-wrap:anywhere; }
.pair .v { font-family:var(--font-mono); font-size:.78rem; }
.pair .src { font-family:var(--font-mono); font-size:.68rem; color:var(--muted); display:block; margin-bottom:.35rem; }
.pair.ai { border-inline-start:4px solid var(--warn); }
.pair.human { border-inline-start:4px solid var(--good); }
.ticket { border:1px solid var(--border); border-radius:8px; background:var(--bg-raised); padding:.55rem .75rem;
  display:flex; justify-content:space-between; gap:.6rem; align-items:center; margin-top:.5rem; flex-wrap:wrap; }
.ticket .msg { flex:1 1 200px; min-inline-size:0; overflow-wrap:anywhere; }
.ticket.flag { border-inline-start:4px solid var(--warn); }
.ticket.clear { border-inline-start:4px solid var(--muted); }
.ticket .route { font-family:var(--font-mono); font-size:.72rem; white-space:nowrap; }
.badge { font-family:var(--font-mono); font-size:.68rem; padding:.1rem .45rem; border-radius:999px; border:1px solid var(--border); }
.badge.ai { color:var(--warn); border-color:var(--warn); }
.badge.human { color:var(--good); border-color:var(--good); }
.probe { border:1px solid var(--border); border-radius:8px; background:var(--bg-secondary); padding:.6rem .8rem; margin-top:.5rem; }
.probe .head { display:flex; justify-content:space-between; gap:.6rem; align-items:baseline; flex-wrap:wrap; }
.probe .name { font-weight:600; font-size:.9rem; }
.probe .expect { font-size:.78rem; color:var(--muted); margin-top:.25rem; }
.probe .t { font-family:var(--font-mono); font-size:.74rem; color:var(--muted); margin-top:.3rem; overflow-wrap:anywhere; }
.probe.waver { border-inline-start:4px solid var(--warn); }
.probe.holds { border-inline-start:4px solid var(--good); }
`;
