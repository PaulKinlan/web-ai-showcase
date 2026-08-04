// Front-end helpers for the MobileBERT-NLI pages.

const WORKER_URL = "/web-ai-showcase/models/mobilebert-nli/worker.js";

export class NliEngine {
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

  classify(premise, hypothesis) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "classify", id, premise, hypothesis });
    });
  }
}

export const NLI_CSS = `
textarea.prompt { inline-size: 100%; font: inherit; padding: .6rem .7rem; border-radius: 8px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); min-block-size: 3rem; resize: vertical; }
.results { display: flex; flex-direction: column; gap: .5rem; margin-top: .7rem; }
.nli-row { display: grid; grid-template-columns: 7rem 1fr 4rem; align-items: center; gap: .6rem; font-size: .9rem; }
.nli-row .label { font-weight: 600; }
.nli-row .bar { block-size: .8rem; border-radius: 999px; background: var(--bg-secondary); border: 1px solid var(--border); overflow: hidden; }
.nli-row .fill { display: block; block-size: 100%; background: var(--accent); }
.nli-row.top .fill { background: var(--good); }
.nli-row .score { font-family: var(--font-mono); font-size: .8rem; text-align: end; }
.controls { display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; margin: .6rem 0; }
.sample-row { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0; }
.chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.chip:hover, .chip:focus-visible { border-color: var(--accent); }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
@media (max-width: 560px){ .nli-row { grid-template-columns: 5.5rem 1fr 3.2rem; } }
`;
