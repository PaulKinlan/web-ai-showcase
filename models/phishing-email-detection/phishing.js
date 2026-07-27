// Front-end helpers for the phishing email / URL detection pages. Keeps each page thin: it owns the worker
// handshake and the renderers (verdict meter, class bars, occlusion highlights, inbox triage rows).
// All inference (classify / batch / occlusion) lives in worker.js, off the main thread.

const WORKER_URL = "/web-ai-showcase/models/phishing-email-detection/worker.js";

/** Client for the phishing text-classification worker. */
export class PhishEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this._loadWaiters = [];
    this._pending = new Map();
    this._batchItems = new Map();
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
    } else if (msg.type === "batch-item") {
      this._batchItems.get(msg.id)?.(msg.item);
    } else if (msg.type === "result" || msg.type === "attr" || msg.type === "batch") {
      const p = this._pending.get(msg.id);
      if (p) {
        this._pending.delete(msg.id);
        this._batchItems.delete(msg.id);
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

  /** Classify one message → { text, threat, safe, label, ms, device }. */
  classify(text) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, text });
    });
  }

  /** Classify a batch → { results:[{index,text,threat,safe,label}], ms, device }. onItem fires per message. */
  classifyBatch(texts, onItem) {
    const id = ++this._id;
    if (onItem) this._batchItems.set(id, onItem);
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "batch", id, texts });
    });
  }

  /** Occlusion attribution → { text, words, attributions, threat, label, ms, device }. */
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

/** Render a PHISHING/LEGITIMATE verdict + confidence + dual meter into `els`. */
export function renderVerdict(els, { threat, safe, label }) {
  const conf = Math.max(threat, safe);
  els.label.textContent = label;
  els.label.className = "verdict-label " + (label === "PHISHING" ? "threat" : "safe");
  els.conf.textContent = (conf * 100).toFixed(1) + "%";
  if (els.fillThreat) els.fillThreat.style.inlineSize = (threat * 100).toFixed(1) + "%";
  if (els.fillSafe) els.fillSafe.style.inlineSize = (safe * 100).toFixed(1) + "%";
}

/**
 * Render class scores as labelled horizontal bars with numeric values. Pass `{ threat, safe }` for the
 * aggregated 2-way split, or `{ classes: [[name, p], …] }` for the model's four raw class scores
 * (legitimate email / phishing / legitimate url / phishing url).
 */
export function renderClassBars(container, { threat, safe, classes }) {
  const rows = classes
    ? classes.map(([name, val]) => [name.startsWith("phish") ? "threat" : "safe", name, val])
    : [
      ["safe", "legitimate", safe],
      ["threat", "phishing", threat],
    ];
  container.replaceChildren(...rows.map(([cls, name, val]) => {
    const row = document.createElement("div");
    row.className = "class-row";
    const label = document.createElement("span");
    label.className = "class-name " + cls;
    label.textContent = name;
    const track = document.createElement("div");
    track.className = "class-track";
    const fill = document.createElement("div");
    fill.className = "class-fill " + cls;
    fill.style.inlineSize = (val * 100).toFixed(2) + "%";
    track.append(fill);
    const v = document.createElement("span");
    v.className = "class-val";
    v.textContent = (val * 100).toFixed(1) + "%";
    row.append(label, track, v);
    return row;
  }));
}

/**
 * Render occlusion attributions as coloured word chips. Red = pushed PHISHING, green = pushed
 * LEGITIMATE; brightness ∝ magnitude. `words`/`attributions` are aligned arrays from engine.attribute().
 */
export function renderAttribution(container, words, attributions) {
  const maxAbs = Math.max(1e-6, ...attributions.map(Math.abs));
  container.replaceChildren(...words.map((w, i) => {
    const a = attributions[i];
    const t = Math.abs(a) / maxAbs;
    const hue = a >= 0 ? "var(--bad)" : "var(--good)";
    const span = document.createElement("span");
    span.className = "attr-word";
    span.textContent = w;
    span.style.background = `color-mix(in srgb, ${hue} ${(t * 60).toFixed(0)}%, transparent)`;
    span.title = `${a >= 0 ? "+" : ""}${a.toFixed(2)} log-odds toward PHISHING`;
    return span;
  }));
}

/** Build one triage row element for a classified message. */
export function makeTriageRow(item) {
  const row = document.createElement("div");
  row.className = "triage-row " + (item.label === "PHISHING" ? "threat" : "safe");
  const text = document.createElement("span");
  text.className = "triage-text";
  text.textContent = item.text;
  const meta = document.createElement("span");
  meta.className = "triage-meta";
  const badge = document.createElement("span");
  badge.className = "badge " + (item.label === "PHISHING" ? "threat" : "safe");
  const conf = item.label === "PHISHING" ? item.threat : item.safe;
  badge.textContent = `${item.label} ${(conf * 100).toFixed(0)}%`;
  meta.append(badge);
  row.append(text, meta);
  return row;
}

export const PHISH_CSS = `
.verdict-row { display: flex; align-items: baseline; gap: .8rem; flex-wrap: wrap; margin-top: .6rem; }
.verdict-label { font-family: var(--font-display); font-size: 1.8rem; }
.verdict-label.threat { color: var(--bad); }
.verdict-label.safe { color: var(--good); }
.verdict-conf { font-family: var(--font-mono); color: var(--muted); font-size: .9rem; }
.meter-dual { display: flex; block-size: .8rem; border: 1px solid var(--border); border-radius: 999px;
  overflow: hidden; margin-top: .5rem; max-inline-size: 520px; background: var(--bg-raised); }
.meter-safe { background: var(--good); block-size: 100%; }
.meter-threat { background: var(--bad); block-size: 100%; }
.meter-labels { display: flex; justify-content: space-between; max-inline-size: 520px;
  font-family: var(--font-mono); font-size: .72rem; color: var(--muted); margin-top: .2rem; }
.class-bars { display: flex; flex-direction: column; gap: .4rem; margin-top: .6rem; max-inline-size: 560px; }
.class-row { display: grid; grid-template-columns: 5.5rem 1fr 3.2rem; align-items: center; gap: .5rem; }
.class-name { font-family: var(--font-mono); font-size: .78rem; }
.class-name.threat { color: var(--bad); }
.class-name.safe { color: var(--good); }
.class-track { block-size: .7rem; background: var(--bg-raised); border: 1px solid var(--border);
  border-radius: 999px; overflow: hidden; }
.class-fill { block-size: 100%; }
.class-fill.threat { background: var(--bad); }
.class-fill.safe { background: var(--good); }
.class-val { font-family: var(--font-mono); font-size: .76rem; color: var(--muted); text-align: end; }
.attr-wrap { line-height: 2.1; margin-top: .5rem; }
.attr-word { padding: .12rem .28rem; border-radius: 5px; margin: 0 1px; white-space: pre-wrap; }
.attr-legend { display: flex; flex-wrap: wrap; gap: 1rem; font-size: .78rem; color: var(--muted);
  font-family: var(--font-mono); margin-top: .6rem; }
.attr-legend .swatch { display: inline-block; inline-size: .8rem; block-size: .8rem; border-radius: 3px;
  margin-inline-end: .3rem; vertical-align: -1px; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.chip { font: inherit; font-size: .78rem; padding: .35rem .7rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.chip:hover { border-color: var(--accent); }
.triage-board { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  margin-top: .8rem; }
.triage-col { border: 1px solid var(--border); border-radius: 10px; background: var(--bg-secondary);
  padding: .6rem; min-inline-size: 0; }
.triage-col h4 { margin: 0 0 .5rem; font-family: var(--font-display); display: flex;
  justify-content: space-between; align-items: baseline; gap: .5rem; }
.triage-col .count { font-family: var(--font-mono); font-size: .78rem; color: var(--muted); }
.triage-col.inbox { border-block-start: 4px solid var(--good); }
.triage-col.junk { border-block-start: 4px solid var(--bad); }
.triage-list { display: flex; flex-direction: column; gap: .5rem; }
.triage-row { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised);
  padding: .5rem .7rem; display: flex; justify-content: space-between; gap: .6rem; align-items: center; }
.triage-row.threat { border-inline-start: 4px solid var(--bad); }
.triage-row.safe { border-inline-start: 4px solid var(--good); }
.triage-text { flex: 1 1 auto; min-inline-size: 0; overflow-wrap: anywhere; }
.triage-meta { white-space: nowrap; }
.badge { font-family: var(--font-mono); font-size: .68rem; padding: .1rem .45rem; border-radius: 999px;
  border: 1px solid var(--border); }
.badge.threat { color: var(--bad); border-color: var(--bad); }
.badge.safe { color: var(--good); border-color: var(--good); }
.controls-grid { display: grid; gap: .9rem 1.2rem; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  align-items: end; margin: .6rem 0; }
.controls-grid label { display: flex; flex-direction: column; gap: .3rem; font-size: .82rem; }
.controls-grid input[type=range] { inline-size: 100%; }
.controls-grid .val { font-family: var(--font-mono); color: var(--muted); font-size: .78rem; }
.obf-toggles { display: flex; flex-wrap: wrap; gap: .5rem 1rem; margin: .6rem 0; }
.obf-toggles label { display: flex; align-items: center; gap: .4rem; font-size: .84rem; }
.obf-out { border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-raised);
  padding: .7rem .9rem; min-block-size: 3rem; white-space: pre-wrap; overflow-wrap: anywhere;
  font-family: var(--font-mono); font-size: .9rem; margin-top: .3rem; }
`;
