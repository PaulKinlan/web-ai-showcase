// Front-end helpers for the human action-recognition pages. Owns the worker handshake, turns files into
// data URLs, and renders the probability bars. All inference happens off the main thread in worker.js.

const WORKER_URL = "/web-ai-showcase/models/human-action-recognition/worker.js";

export class ActionEngine {
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

  /** Classify an image → { top, entropy, margin, numClasses, ms, device }. */
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

export function escapeHTML(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]),
  );
}

/** Render the top-k actions as labelled probability bars into a container. */
export function renderBars(container, top) {
  container.innerHTML = top.map((t) => {
    const pct = Math.round(t.prob * 100);
    return `<div class="har-bar"><span class="har-name">${escapeHTML(t.label)}</span>` +
      `<span class="har-track"><i style="width:${Math.max(2, pct)}%"></i></span>` +
      `<span class="har-pct">${pct}%</span></div>`;
  }).join("");
}

export const HAR_CSS = `
  .har-gallery { display: flex; flex-wrap: wrap; gap: 0.9rem; margin: 0.8rem 0; }
  .har-card { width: 200px; }
  .har-card img { width: 200px; height: 150px; object-fit: cover; border-radius: 10px; background: #7772; display: block; }
  .har-top { font-weight: 600; margin: 0.35rem 0 0.2rem; font-size: 0.9rem; }
  .har-bars { display: flex; flex-direction: column; gap: 0.2rem; }
  .har-bar { display: grid; grid-template-columns: 5.2rem 1fr 2.4rem; align-items: center; gap: 0.4rem; font-size: 0.74rem; }
  .har-name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .har-track { height: 7px; border-radius: 4px; background: #7772; overflow: hidden; }
  .har-track > i { display: block; height: 100%; background: linear-gradient(90deg, #2bb59a, #4ac6e0); }
  .har-pct { font-family: var(--font-mono, monospace); text-align: right; }
  .har-dropzone { border: 1.5px dashed #8884; border-radius: 10px; padding: 0.9rem; text-align: center; cursor: pointer; font-size: 0.9rem; }
  .har-dropzone:focus-visible { outline: 2px solid #2bb59a; }
  .har-status { font-family: var(--font-mono, monospace); font-size: 0.9rem; margin: 0.4rem 0; }
`;
