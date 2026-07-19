// Shared front-end helpers for the Food-101 classification pages. Keeps each page thin: it owns the
// worker handshake, turns files/samples into data URLs, renders the probability bars, and formats the
// "see inside" numbers. All inference happens off the main thread in worker.js.

const WORKER_URL = "/web-ai-showcase/models/food-classification/worker.js";

export class FoodEngine {
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

  /** Classify an image. Returns { top, entropy, margin, numClasses, ms, device }. */
  classify(imageURL, topK = 5) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, image: imageURL, topK });
    });
  }
}

/** Read a File (from upload or drop) into a data URL usable by the worker. */
export function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}

/**
 * Render probability bars into `container`. `items` = [{label, prob}], sorted high→low here.
 * The top result is highlighted. Percentages are the softmax probabilities.
 */
export function renderBars(container, items) {
  const sorted = [...items].sort((a, b) => b.prob - a.prob);
  const top = sorted[0]?.label;
  container.replaceChildren(
    ...sorted.map((it) => {
      const row = document.createElement("div");
      row.className = "bar-row" + (it.label === top ? " bar-top" : "");
      const pct = (it.prob * 100).toFixed(1);
      const name = prettyDish(it.label);
      row.innerHTML = `
        <div class="bar-head">
          <span class="bar-label">${escapeHTML(name)}</span>
          <span class="bar-val">${pct}%</span>
        </div>
        <div class="bar-track" role="meter" aria-valuemin="0" aria-valuemax="100"
             aria-valuenow="${pct}" aria-label="${escapeHTML(name)}: ${pct} percent">
          <div class="bar-fill" style="inline-size:${pct}%"></div>
        </div>`;
      return row;
    }),
  );
}

/** Food-101 labels are snake_case ("baby_back_ribs"); turn them into a readable dish name. */
export function prettyDish(label) {
  const s = String(label).replace(/_/g, " ").trim();
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Shared inline styles for the Food widgets (bars, dropzone, readout). Injected once per page. */
export const FOOD_CSS = `
.dropzone {
  border: 2px dashed var(--border-strong);
  border-radius: var(--radius);
  background: var(--bg-raised);
  padding: 1rem;
  text-align: center;
  cursor: pointer;
  transition: border-color .15s, background .15s;
}
.dropzone.drag { border-color: var(--accent); background: var(--bg-secondary); }
.dropzone:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.preview-wrap { position: relative; display: inline-block; max-inline-size: 100%; }
.preview-img { max-inline-size: 100%; max-block-size: 340px; border-radius: 8px; display: block; }
.sample-strip { display: flex; gap: .5rem; flex-wrap: wrap; margin: .5rem 0; }
.sample-thumb {
  inline-size: 76px; block-size: 56px; object-fit: cover; border-radius: 6px;
  border: 2px solid transparent; cursor: pointer; padding: 0;
}
.sample-thumb.active { border-color: var(--accent); }
.sample-thumb:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.bars { display: flex; flex-direction: column; gap: .55rem; margin-top: .5rem; }
.bar-row { min-inline-size: 0; }
.bar-head { display: flex; justify-content: space-between; gap: .5rem; font-size: .85rem; }
.bar-label { font-family: var(--font-body); min-inline-size: 0; overflow-wrap: anywhere; }
.bar-val { font-family: var(--font-mono); color: var(--muted); white-space: nowrap; }
.bar-track {
  block-size: .7rem; background: var(--bg-raised); border: 1px solid var(--border);
  border-radius: 999px; overflow: hidden; margin-top: .15rem;
}
.bar-fill {
  block-size: 100%; background: var(--muted); border-radius: 999px;
  transition: inline-size .35s ease;
}
.bar-top .bar-fill { background: var(--accent); }
.bar-top .bar-label { font-weight: 600; }
.field-row { display: flex; flex-wrap: wrap; gap: .6rem; align-items: end; margin: .6rem 0; }
.readout {
  display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem;
}
.readout b { color: var(--color); font-weight: 600; }
.chip {
  font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer;
}
.chip:hover { border-color: var(--accent); }
.inside-table { inline-size: 100%; border-collapse: collapse; font-size: .82rem; margin-top: .5rem; }
.inside-table th, .inside-table td {
  text-align: left; padding: .3rem .5rem; border-bottom: 1px solid var(--border);
  font-family: var(--font-mono);
}
.inside-table th { color: var(--muted); font-weight: 600; }
.conf-meter { block-size: .9rem; border-radius: 999px; overflow: hidden; border: 1px solid var(--border);
  background: linear-gradient(to right, var(--bad), var(--warn), var(--good)); position: relative; margin: .3rem 0; }
.conf-needle { position: absolute; top: -3px; bottom: -3px; inline-size: 3px; background: var(--color); border-radius: 2px; }
`;
