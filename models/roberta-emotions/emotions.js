// Front-end helpers for the RoBERTa GoEmotions pages. Keeps each page thin: it owns the worker
// handshake and the emotion renderers. All inference lives in worker.js (off the main thread).

const WORKER_URL = "/web-ai-showcase/models/roberta-emotions/worker.js";

// The 28 GoEmotions classes, grouped by affective tone. The grouping only drives colour + summary —
// the model itself outputs a flat, independent score per class. (Grouping follows the GoEmotions
// paper's sentiment mapping.)
export const EMOTION_GROUP = {
  admiration: "pos",
  amusement: "pos",
  approval: "pos",
  caring: "pos",
  desire: "pos",
  excitement: "pos",
  gratitude: "pos",
  joy: "pos",
  love: "pos",
  optimism: "pos",
  pride: "pos",
  relief: "pos",
  anger: "neg",
  annoyance: "neg",
  disappointment: "neg",
  disapproval: "neg",
  disgust: "neg",
  embarrassment: "neg",
  fear: "neg",
  grief: "neg",
  nervousness: "neg",
  remorse: "neg",
  sadness: "neg",
  confusion: "amb",
  curiosity: "amb",
  realization: "amb",
  surprise: "amb",
  neutral: "neu",
};

export const GROUP_VAR = {
  pos: "var(--good)",
  neg: "var(--bad)",
  amb: "var(--warn)",
  neu: "var(--muted)",
};
export const GROUP_NAME = { pos: "positive", neg: "negative", amb: "ambiguous", neu: "neutral" };

export function groupOf(label) {
  return EMOTION_GROUP[label] || "neu";
}
export function colorOf(label) {
  return GROUP_VAR[groupOf(label)];
}

export class EmotionEngine {
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

  /** Score one text → { scores:[{label,score}×28 sorted], ms, device }. */
  classify(text) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, text });
    });
  }

  /** Score many texts in one pass → { results:[[{label,score}×28]…], ms, device }. */
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

/** Split text into sentences for the emotion arc. Keeps it simple + punctuation-aware. */
export function splitSentences(text) {
  return text
    .replace(/\s+/g, " ")
    .match(/[^.!?]+[.!?]*/g)?.map((s) => s.trim()).filter((s) => s.length > 1) ?? [];
}

/**
 * Render a full ranked bar chart of all 28 class scores into `container`. Rows above `threshold`
 * are marked `.pass` (bold + badge). Colour encodes the affective group. `limit` caps visible rows.
 */
export function renderScores(container, scores, threshold = 0.3, limit = 28) {
  const rows = scores.slice(0, limit).map(({ label, score }) => {
    const pass = score >= threshold;
    const pct = (score * 100).toFixed(1);
    return `
      <div class="emo-row${pass ? " pass" : ""}">
        <span class="emo-name">${escapeHTML(label)}</span>
        <span class="emo-bar"><span class="emo-fill" style="inline-size:${pct}%;background:${
      colorOf(label)
    }"></span></span>
        <span class="emo-score">${pct}%</span>
      </div>`;
  });
  container.innerHTML = rows.join("");
}

/** Render the top emotions (those above threshold, else the single top) as coloured pills. */
export function renderTop(container, scores, threshold = 0.3) {
  let top = scores.filter((s) => s.score >= threshold);
  if (top.length === 0) top = scores.slice(0, 1);
  container.replaceChildren(...top.map(({ label, score }) => {
    const el = document.createElement("span");
    el.className = "emo-pill " + groupOf(label);
    el.innerHTML = `${escapeHTML(label)} <b>${(score * 100).toFixed(0)}%</b>`;
    return el;
  }));
  return top;
}

export const EMOTION_CSS = `
.emo-list { display:flex; flex-direction:column; gap:.32rem; margin-top:.6rem; }
.emo-row { display:grid; grid-template-columns:8.5rem 1fr 3.4rem; gap:.55rem; align-items:center; }
.emo-name { font-size:.85rem; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.emo-bar { block-size:.72rem; background:var(--bg-secondary); border:1px solid var(--border);
  border-radius:999px; overflow:hidden; }
.emo-fill { display:block; block-size:100%; border-radius:999px; transition:inline-size .18s ease; }
.emo-score { font-family:var(--font-mono); font-size:.76rem; color:var(--muted); text-align:end; }
.emo-row.pass .emo-name { color:var(--color); font-weight:600; }
.emo-row.pass .emo-score { color:var(--color); }
.emo-pills { display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.5rem; }
.emo-pill { font-family:var(--font-mono); font-size:.8rem; padding:.2rem .6rem; border-radius:999px;
  border:1px solid var(--border); background:var(--bg-raised); }
.emo-pill b { font-weight:700; }
.emo-pill.pos { color:var(--good); border-color:var(--good); }
.emo-pill.neg { color:var(--bad); border-color:var(--bad); }
.emo-pill.amb { color:var(--warn); border-color:var(--warn); }
.emo-pill.neu { color:var(--muted); }
.verdict-row { display:flex; align-items:baseline; gap:.8rem; flex-wrap:wrap; margin-top:.6rem; }
.verdict-label { font-family:var(--font-display); font-size:1.8rem; text-transform:capitalize; }
.verdict-label.pos { color:var(--good); } .verdict-label.neg { color:var(--bad); }
.verdict-label.amb { color:var(--warn); } .verdict-label.neu { color:var(--muted); }
.verdict-conf { font-family:var(--font-mono); color:var(--muted); font-size:.9rem; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem;
  color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.chip { font:inherit; font-size:.78rem; padding:.2rem .6rem; border-radius:999px; border:1px solid var(--border);
  background:var(--bg-raised); color:var(--color); cursor:pointer; }
.chip:hover { border-color:var(--accent); }
.controls-grid { display:grid; gap:.9rem 1.2rem; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr));
  align-items:end; margin:.6rem 0; }
.controls-grid label { display:flex; flex-direction:column; gap:.3rem; font-size:.82rem; }
.stat-row { display:grid; gap:.6rem; grid-template-columns:repeat(auto-fit, minmax(120px, 1fr)); margin-top:.6rem; }
.stat { border:1px solid var(--border); border-radius:8px; background:var(--bg-secondary); padding:.5rem .7rem; }
.stat .k { font-family:var(--font-mono); font-size:.66rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
.stat .v { font-family:var(--font-display); font-size:1.3rem; text-transform:capitalize; }
.ticket { border:1px solid var(--border); border-radius:8px; background:var(--bg-raised); padding:.55rem .75rem;
  display:flex; justify-content:space-between; gap:.6rem; align-items:center; margin-top:.5rem; }
.ticket .msg { flex:1 1 auto; }
.ticket.route-urgent { border-inline-start:4px solid var(--bad); }
.ticket.route-praise { border-inline-start:4px solid var(--good); }
.ticket.route-help { border-inline-start:4px solid var(--warn); }
.ticket.route-general { border-inline-start:4px solid var(--muted); }
.ticket .route { font-family:var(--font-mono); font-size:.72rem; white-space:nowrap; }
.arc-step { border:1px solid var(--border); border-radius:8px; background:var(--bg-raised); padding:.5rem .7rem;
  margin-top:.5rem; border-inline-start:4px solid var(--muted); }
.arc-step .meta { font-family:var(--font-mono); font-size:.72rem; margin-bottom:.2rem; }
.arc-sentence { font-size:.95rem; }
`;
