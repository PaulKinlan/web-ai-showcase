// Front-end helpers for the Chinese spell-check pages. Thin: owns the worker handshake + renderers.

const WORKER_URL = "/web-ai-showcase/models/chinese-spell-check/worker.js";

export class CscEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
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
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load" });
    });
  }

  correct(text) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "correct", id, text });
    });
  }
}

export const CSC_CSS = `
textarea.prompt { inline-size: 100%; font: inherit; padding: .6rem .7rem; border-radius: 8px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); min-block-size: 4.2rem; resize: vertical; }
.gen-out { font-size: 1.15rem; line-height: 2; border: 1px solid var(--border); border-radius: 8px;
  background: var(--bg-raised); padding: .7rem .8rem; margin-top: .6rem; min-block-size: 3rem; }
.gen-out .fix { background: color-mix(in srgb, var(--good) 28%, transparent); border-radius: 4px; padding: 0 .12rem; }
.gen-out .fix-old { color: var(--muted); text-decoration: line-through; }
.controls { display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; margin: .6rem 0; }
.sample-row { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0; }
.chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.chip:hover, .chip:focus-visible { border-color: var(--accent); }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.changes { display: flex; flex-direction: column; gap: .3rem; margin-top: .6rem; font-size: .9rem; }
.changes .c { display: flex; gap: .5rem; align-items: baseline; }
.changes .c .from { color: var(--muted); text-decoration: line-through; }
.changes .c .to { color: var(--good); font-weight: 600; }
.changes .c .conf { font-family: var(--font-mono); font-size: .74rem; color: var(--muted); }
`;
