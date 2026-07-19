// Front-end helpers for the news-topic-classification pages. Keeps each page thin: it owns the worker
// handshake and the shared renderers (distribution bars, topic chips, the tagging board). All inference
// lives in worker.js (off the main thread).

const WORKER_URL = "/web-ai-showcase/models/news-topic-classification/worker.js";

// The model's fixed 10-class label set (order matches its id2label). Each gets a stable accent hue so
// the same topic reads the same colour across the board, the bars, and the feed.
export const TOPICS = [
  { label: "Society & Culture", hue: 275 },
  { label: "Science & Mathematics", hue: 200 },
  { label: "Health", hue: 150 },
  { label: "Education & Reference", hue: 45 },
  { label: "Computers & Internet", hue: 220 },
  { label: "Sports", hue: 15 },
  { label: "Business & Finance", hue: 95 },
  { label: "Entertainment & Music", hue: 325 },
  { label: "Family & Relationships", hue: 340 },
  { label: "Politics & Government", hue: 0 },
];

const HUE = new Map(TOPICS.map((t) => [t.label, t.hue]));
export function topicHue(label) {
  return HUE.has(label) ? HUE.get(label) : 210;
}
export function topicColor(label, alpha = 1) {
  return `hsl(${topicHue(label)} 70% 45% / ${alpha})`;
}

export class TopicEngine {
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

  /** Classify one text → { label, score, dist[10], entropy, entropyNorm, margin, ms, device }. */
  classify(text) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, text });
    });
  }

  /** Classify an array of texts in one batched pass → { rows:[{text,label,score,dist}], ms, device }. */
  batch(texts) {
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

/** A coloured pill naming a topic. */
export function topicChip(label) {
  const span = document.createElement("span");
  span.className = "topic-chip";
  span.textContent = label;
  span.style.setProperty("--h", topicHue(label));
  return span;
}

/**
 * Render a full class distribution as horizontal bars into `container`. `dist` is the sorted
 * [{label, score}] array (10 rows). The top row is emphasised. Accessible: each bar is a
 * role="meter" with aria-valuenow so a screen reader can read the probabilities.
 */
export function renderDistribution(container, dist) {
  container.replaceChildren(...dist.map((d, i) => {
    const row = document.createElement("div");
    row.className = "dist-row" + (i === 0 ? " top" : "");
    const name = document.createElement("span");
    name.className = "dist-name";
    name.textContent = d.label;
    const track = document.createElement("div");
    track.className = "dist-track";
    track.setAttribute("role", "meter");
    track.setAttribute("aria-label", `${d.label} probability`);
    track.setAttribute("aria-valuemin", "0");
    track.setAttribute("aria-valuemax", "100");
    track.setAttribute("aria-valuenow", (d.score * 100).toFixed(0));
    const fill = document.createElement("div");
    fill.className = "dist-fill";
    fill.style.inlineSize = (d.score * 100).toFixed(2) + "%";
    fill.style.background = topicColor(d.label, 0.9);
    track.append(fill);
    const pct = document.createElement("span");
    pct.className = "dist-pct";
    pct.textContent = (d.score * 100).toFixed(1) + "%";
    row.append(name, track, pct);
    return row;
  }));
}

export const TOPIC_CSS = `
.verdict-row { display: flex; align-items: baseline; gap: .8rem; flex-wrap: wrap; margin-top: .6rem; }
.verdict-topic { font-family: var(--font-display); font-size: 1.7rem; line-height: 1.1;
  color: hsl(var(--vh, 210) 70% 42%); }
@media (prefers-color-scheme: dark) { .verdict-topic { color: hsl(var(--vh, 210) 70% 68%); } }
.verdict-conf { font-family: var(--font-mono); color: var(--muted); font-size: .9rem; }
.topic-chip { display: inline-flex; align-items: center; font-family: var(--font-mono); font-size: .72rem;
  padding: .12rem .5rem; border-radius: 999px; white-space: nowrap;
  color: hsl(var(--h) 60% 30%); background: hsl(var(--h) 70% 45% / .16); border: 1px solid hsl(var(--h) 60% 45% / .5); }
@media (prefers-color-scheme: dark) { .topic-chip { color: hsl(var(--h) 70% 78%); } }
.dist-list { display: flex; flex-direction: column; gap: .3rem; margin-top: .5rem; }
.dist-row { display: grid; grid-template-columns: minmax(0, 11rem) 1fr auto; align-items: center; gap: .6rem; }
.dist-name { font-size: .8rem; color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.dist-row.top .dist-name { color: var(--color); font-weight: 600; }
.dist-track { block-size: .7rem; background: var(--bg-secondary); border: 1px solid var(--border);
  border-radius: 999px; overflow: hidden; min-inline-size: 0; }
.dist-fill { block-size: 100%; border-radius: 999px; transition: inline-size .25s ease; }
.dist-pct { font-family: var(--font-mono); font-size: .72rem; color: var(--muted); min-inline-size: 3.2rem; text-align: end; }
.dist-row.top .dist-pct { color: var(--color); }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .7rem; }
.readout b { color: var(--color); font-weight: 600; }
.chip { font: inherit; font-size: .78rem; padding: .3rem .7rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; min-block-size: 2.2rem; }
.chip:hover { border-color: var(--accent); }
.chip-row { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0; }
.desk-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 15rem), 1fr));
  gap: 1rem; margin-top: .8rem; }
.desk { border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-raised);
  padding: .6rem .7rem; border-block-start: 4px solid hsl(var(--h) 70% 45%); }
.desk h4 { margin: 0 0 .4rem; font-size: .85rem; display: flex; justify-content: space-between; gap: .5rem; align-items: baseline; }
.desk-count { font-family: var(--font-mono); font-size: .72rem; color: var(--muted); }
.desk-item { font-size: .82rem; padding: .3rem 0; border-block-start: 1px solid var(--border); }
.desk-item:first-of-type { border-block-start: 0; }
.desk-item .meta { font-family: var(--font-mono); font-size: .68rem; color: var(--muted); }
.feed-list { display: flex; flex-direction: column; gap: .5rem; margin-top: .6rem; }
.feed-row { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; justify-content: space-between;
  border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised); padding: .5rem .7rem;
  border-inline-start: 4px solid hsl(var(--h) 70% 45%); }
.feed-text { flex: 1 1 14rem; min-inline-size: 0; }
.amb-meter { block-size: .8rem; border: 1px solid var(--border); border-radius: 999px; overflow: hidden;
  background: linear-gradient(90deg, var(--good), var(--warn), var(--bad)); position: relative; max-inline-size: 520px; margin-top: .4rem; }
.amb-needle { position: absolute; inset-block: -3px; inline-size: 3px; background: var(--color); border-radius: 2px; transition: inset-inline-start .25s ease; }
.amb-labels { display: flex; justify-content: space-between; max-inline-size: 520px; font-family: var(--font-mono);
  font-size: .7rem; color: var(--muted); margin-top: .2rem; }
`;
