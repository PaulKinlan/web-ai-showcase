// Front-end helpers for the prompt-injection page: the worker handshake + verdict rendering.
// All inference lives in worker.js (off the main thread).

export class GuardEngine {
  constructor() {
    this.worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
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
  /** Classify a prompt → { label, score, scores, ms }. */
  classify(text) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, text });
    });
  }
}

export function escapeHTML(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]),
  );
}

export const isInjection = (label) => /injection/i.test(label);

export const GUARD_CSS = `
  .pg-field { margin: 0.5rem 0; }
  .pg-field label { display: block; font-size: 0.85rem; margin-bottom: 0.2rem; color: var(--muted, #888); }
  .pg-field textarea { width: 100%; box-sizing: border-box; padding: 0.6rem 0.7rem; border-radius: 8px; border: 1px solid #8886; font: inherit; font-size: 0.98rem; min-height: 4.5rem; resize: vertical; }
  .pg-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.5rem 0; }
  .pg-chip.inj { border-color: #e05b5b88; }
  .pg-verdict { display: flex; align-items: center; gap: 0.7rem; margin: 0.8rem 0 0.3rem; font-size: 1.25rem; font-weight: 700; }
  .pg-badge { display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.25rem 0.7rem; border-radius: 999px; color: #fff; }
  .pg-badge svg { width: 1.1rem; height: 1.1rem; }
  .pg-badge.safe { background: #2bb59a; }
  .pg-badge.inj { background: #e05b5b; }
  .pg-conf { font-family: var(--font-mono, monospace); font-size: 0.9rem; color: var(--muted, #888); font-weight: 400; }
  .pg-bar { height: 10px; border-radius: 5px; background: #7772; overflow: hidden; max-width: 34rem; margin: 0.3rem 0; }
  .pg-bar > i { display: block; height: 100%; }
  .pg-status { font-family: var(--font-mono, monospace); font-size: 0.9rem; margin: 0.4rem 0; }
`;
