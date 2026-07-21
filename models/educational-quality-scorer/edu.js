// Front-end helpers for the Educational-quality scorer page. Owns the worker handshake and renders the
// 0-5 score gauge. All inference (BERT regression) lives in worker.js, off the main thread.

const WORKER_URL = "/web-ai-showcase/models/educational-quality-scorer/worker.js";

// Verified sample passages (headless): educational content scores ~3+, casual/promotional ~0.
export const SAMPLES = {
  science:
    "Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into glucose and oxygen. It takes place in the chloroplasts, where the pigment chlorophyll absorbs light energy. The light-dependent reactions produce ATP and NADPH, which the Calvin cycle then uses to fix carbon dioxide into sugars.",
  history:
    "The Industrial Revolution, beginning in Britain around 1760, marked a major turning point in history. Mechanised manufacturing, the steam engine, and the factory system transformed economies and societies. Urban populations grew rapidly as workers moved from rural areas to industrial cities, reshaping labour, class, and daily life.",
  tutorial:
    "To reverse a linked list, iterate through the nodes while keeping three pointers: previous, current, and next. At each step, save current.next, point current.next back to previous, then advance previous and current. When current becomes null, previous is the new head. This runs in O(n) time and O(1) extra space.",
  casual:
    "omg last night was soooo fun lmaooo i cant even, we stayed out till like 4am and then got mcdonalds on the way home haha. anyway hmu later if you wanna hang, im so tired today though ngl",
  promo:
    "🔥 CONGRATULATIONS! 🔥 You've been SELECTED to win a FREE iPhone 15 Pro!!! Just click the link below and enter your details to CLAIM YOUR PRIZE now — limited time only, don't miss out!!! Act fast before it's gone!!!",
};

export function debounce(fn, ms) {
  let t;
  return (...a) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...a), ms);
  };
}

export class EduEngine {
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
    if (msg.type === "progress") this.onProgress?.(msg.p);
    else if (msg.type === "ready") {
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
        for (const w of this._loadWaiters) w.reject(new Error(msg.message));
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
  score(text) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "score", id, text });
    });
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// hue from red (0) → amber (2.5) → green (5)
export const scoreHue = (s) => Math.max(0, Math.min(120, (s / 5) * 120));

/** Render the 0-5 gauge: a graduated bar, a marker at the score, the big number + band text. */
export function renderScore(container, res) {
  const hue = scoreHue(res.score);
  const pct = (res.score / 5) * 100;
  container.innerHTML = `
    <div class="edu-score">
      <span class="edu-num" style="color:hsl(${hue} 70% 45%)">${res.score.toFixed(1)}</span>
      <span class="edu-out">/ 5 · ${escapeHTML(bandLabel(res.band))}</span>
    </div>
    <div class="edu-gauge" role="img" aria-label="Educational quality ${
    res.score.toFixed(1)
  } out of 5">
      <div class="edu-gauge-fill" style="width:${pct}%;background:hsl(${hue} 70% 50%)"></div>
      <div class="edu-ticks"><span>0</span><span>1</span><span>2</span><span>3</span><span>4</span><span>5</span></div>
    </div>
    <p class="edu-band">${escapeHTML(res.bandText || "")}</p>`;
}

export function bandLabel(b) {
  return ["Not educational", "Minimal", "Some", "Good", "High", "Outstanding"][b] ?? "";
}

export const EDU_CSS = `
.edu-input { font: inherit; inline-size: 100%; padding: .7rem .8rem; border-radius: 8px; min-block-size: 9rem;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); resize: vertical; line-height: 1.5; }
.edu-input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.edu-chips { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0; }
.edu-chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.edu-chip:hover, .edu-chip:focus-visible { border-color: var(--accent); }
.edu-out-wrap { margin-top: .7rem; }
.edu-score { display: flex; align-items: baseline; gap: .5rem; }
.edu-num { font-size: 2.4rem; font-weight: 800; line-height: 1; font-variant-numeric: tabular-nums; }
.edu-out { color: var(--muted); font-size: .95rem; }
.edu-gauge { position: relative; height: 12px; border-radius: 6px; margin: .5rem 0 .2rem; max-width: 34rem;
  background: linear-gradient(90deg, hsl(0 70% 55%), hsl(45 75% 55%), hsl(120 60% 45%)); opacity: .35; }
.edu-gauge-fill { position: absolute; inset: 0 auto 0 0; border-radius: 6px; opacity: 1; }
.edu-ticks { position: absolute; inset: 14px 0 auto 0; display: flex; justify-content: space-between;
  font-family: var(--font-mono, monospace); font-size: .68rem; color: var(--muted); max-width: 34rem; }
.edu-band { font-size: .9rem; color: var(--color); margin: 1.1rem 0 .2rem; max-width: 40rem; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono, monospace);
  font-size: .78rem; color: var(--muted); margin-top: .8rem; }
.readout b { color: var(--color); font-weight: 600; }
`;
