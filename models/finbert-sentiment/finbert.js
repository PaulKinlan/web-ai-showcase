// Front-end helpers for the FinBERT financial-sentiment pages. Keeps each page thin: it owns the
// worker handshake and the renderers (three-class meter, class bars, occlusion highlights, batch rows).
// All inference (classification + occlusion) lives in worker.js, off the main thread.

const WORKER_URL = "/web-ai-showcase/models/finbert-sentiment/worker.js";

export const CLASSES = ["positive", "neutral", "negative"];
export const CLASS_COLOR = {
  positive: "var(--good)",
  neutral: "var(--warn)",
  negative: "var(--bad)",
};

/** Client for the FinBERT text-classification worker. */
export class FinbertEngine {
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
    } else if (msg.type === "result" || msg.type === "attr" || msg.type === "batch") {
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

  /** Classify one text → { text, scores:{positive,negative,neutral}, label, net, ms, device }. */
  classify(text) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, text });
    });
  }

  /** Classify a batch → { rows:[{text,scores,label,net}], ms, device }. */
  classifyBatch(texts) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "batch", id, texts });
    });
  }

  /** Occlusion attribution → { text, words, attributions, scores, label, ms, device }. */
  attribute(text) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "attribute", id, text });
    });
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Simple debounce for "as you type" surfaces. */
export function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

/** Render a top-line verdict (label + confidence) into `els`. */
export function renderVerdict(els, { scores, label }) {
  els.label.textContent = label.toUpperCase();
  els.label.className = "verdict-label " + label;
  els.conf.textContent = (scores[label] * 100).toFixed(1) + "%";
}

/** Render the three-class probability split as a segmented meter (labels sit outside for contrast). */
export function renderTriMeter(meterEl, scores) {
  meterEl.replaceChildren(...CLASSES.map((c) => {
    const seg = document.createElement("div");
    seg.className = "tri-seg " + c;
    seg.style.inlineSize = (scores[c] * 100).toFixed(2) + "%";
    seg.title = `${c}: ${(scores[c] * 100).toFixed(1)}%`;
    return seg;
  }));
}

/** Render the three class scores as labelled horizontal bars with numeric values. */
export function renderClassBars(container, scores) {
  container.replaceChildren(...CLASSES.map((c) => {
    const row = document.createElement("div");
    row.className = "class-row";
    const name = document.createElement("span");
    name.className = "class-name " + c;
    name.textContent = c;
    const track = document.createElement("div");
    track.className = "class-track";
    const fill = document.createElement("div");
    fill.className = "class-fill " + c;
    fill.style.inlineSize = (scores[c] * 100).toFixed(2) + "%";
    track.append(fill);
    const val = document.createElement("span");
    val.className = "class-val";
    val.textContent = (scores[c] * 100).toFixed(1) + "%";
    row.append(name, track, val);
    return row;
  }));
}

/**
 * Render occlusion attributions as coloured word chips. Green = pushed the net signal bullish
 * (positive), red = pushed it bearish (negative); brightness ∝ magnitude. Aligned arrays.
 */
export function renderAttribution(container, words, attributions) {
  const maxAbs = Math.max(1e-6, ...attributions.map(Math.abs));
  container.replaceChildren(...words.map((w, i) => {
    const a = attributions[i];
    const t = Math.abs(a) / maxAbs;
    const hue = a >= 0 ? "var(--good)" : "var(--bad)";
    const span = document.createElement("span");
    span.className = "attr-word";
    span.textContent = w;
    span.style.background = `color-mix(in srgb, ${hue} ${(t * 60).toFixed(0)}%, transparent)`;
    span.title = `${a >= 0 ? "+" : ""}${a.toFixed(2)} net log-odds toward POSITIVE`;
    return span;
  }));
}

/** Render a batch of classified rows (news triage). `rows` = [{text, scores, label}]. */
export function renderRows(container, rows) {
  container.replaceChildren(...rows.map((r) => {
    const row = document.createElement("div");
    row.className = "fin-row " + r.label;
    const text = document.createElement("span");
    text.className = "fin-text";
    text.textContent = r.text;
    const meta = document.createElement("span");
    meta.className = "fin-meta";
    const badge = document.createElement("span");
    badge.className = "badge " + r.label;
    badge.textContent = `${r.label} ${(r.scores[r.label] * 100).toFixed(0)}%`;
    meta.append(badge);
    row.append(text, meta);
    return row;
  }));
}

export const FINBERT_CSS = `
.verdict-row { display: flex; align-items: baseline; gap: .8rem; flex-wrap: wrap; margin-top: .6rem; }
.verdict-label { font-family: var(--font-display); font-size: 1.8rem; }
.verdict-label.positive { color: var(--good); }
.verdict-label.negative { color: var(--bad); }
.verdict-label.neutral { color: var(--warn); }
.verdict-conf { font-family: var(--font-mono); color: var(--muted); font-size: .9rem; }
.tri-meter { display: flex; block-size: .9rem; border: 1px solid var(--border); border-radius: 999px;
  overflow: hidden; margin-top: .5rem; max-inline-size: 560px; background: var(--bg-raised); }
.tri-seg { block-size: 100%; min-inline-size: 0; transition: inline-size .15s ease; }
.tri-seg.positive { background: var(--good); }
.tri-seg.neutral { background: var(--warn); }
.tri-seg.negative { background: var(--bad); }
.tri-legend { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .72rem;
  color: var(--muted); margin-top: .3rem; }
.tri-legend .dot { display: inline-block; inline-size: .7rem; block-size: .7rem; border-radius: 50%;
  margin-inline-end: .3rem; vertical-align: -1px; }
.class-bars { display: flex; flex-direction: column; gap: .4rem; margin-top: .5rem; max-inline-size: 560px; }
.class-row { display: grid; grid-template-columns: 5.5rem 1fr 3.2rem; align-items: center; gap: .5rem; }
.class-name { font-family: var(--font-mono); font-size: .78rem; text-transform: capitalize; }
.class-name.positive { color: var(--good); }
.class-name.negative { color: var(--bad); }
.class-name.neutral { color: var(--warn); }
.class-track { block-size: .7rem; background: var(--bg-raised); border: 1px solid var(--border);
  border-radius: 999px; overflow: hidden; }
.class-fill { block-size: 100%; }
.class-fill.positive { background: var(--good); }
.class-fill.negative { background: var(--bad); }
.class-fill.neutral { background: var(--warn); }
.class-val { font-family: var(--font-mono); font-size: .76rem; color: var(--muted); text-align: end; }
.attr-wrap { line-height: 2.1; margin-top: .5rem; }
.attr-word { padding: .12rem .28rem; border-radius: 5px; margin: 0 1px; white-space: pre-wrap; }
.attr-legend { display: flex; flex-wrap: wrap; gap: 1rem; font-size: .78rem; color: var(--muted);
  font-family: var(--font-mono); margin-top: .6rem; }
.attr-legend .swatch { display: inline-block; inline-size: .8rem; block-size: .8rem; border-radius: 3px;
  margin-inline-end: .3rem; vertical-align: -1px; }
.fin-list { display: flex; flex-direction: column; gap: .5rem; margin-top: .5rem; }
.fin-row { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised);
  padding: .5rem .7rem; display: flex; justify-content: space-between; gap: .6rem; align-items: center; }
.fin-row.positive { border-inline-start: 4px solid var(--good); }
.fin-row.negative { border-inline-start: 4px solid var(--bad); }
.fin-row.neutral { border-inline-start: 4px solid var(--warn); }
.fin-text { flex: 1 1 auto; min-inline-size: 0; }
.fin-meta { white-space: nowrap; }
.badge { font-family: var(--font-mono); font-size: .68rem; padding: .1rem .45rem; border-radius: 999px;
  border: 1px solid var(--border); text-transform: capitalize; }
.badge.positive { color: var(--good); border-color: var(--good); }
.badge.negative { color: var(--bad); border-color: var(--bad); }
.badge.neutral { color: var(--warn); border-color: var(--warn); }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.chip:hover { border-color: var(--accent); }
.cmp-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem;
  margin-top: .6rem; align-items: start; }
.cmp-card { border: 1px solid var(--border); border-radius: 10px; padding: .8rem; background: var(--bg-raised); }
.cmp-card h4 { margin: 0 0 .2rem; font-family: var(--font-display); }
.cmp-card .cmp-sub { font-family: var(--font-mono); font-size: .72rem; color: var(--muted); margin: 0 0 .5rem; }
.cmp-verdict { font-family: var(--font-display); font-size: 1.3rem; }
.cmp-verdict.positive { color: var(--good); }
.cmp-verdict.negative { color: var(--bad); }
.cmp-verdict.neutral { color: var(--warn); }
.disagree { border: 1px solid var(--warn); border-radius: 8px; padding: .5rem .7rem; margin-top: .6rem;
  background: color-mix(in srgb, var(--warn) 10%, transparent); font-size: .86rem; }
`;
