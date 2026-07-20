// Thin engine for the summarizer LLM (SmolDocling multi-model demo). Owns the worker handshake; all
// generation runs in summarize-worker.js (off-thread). Streams tokens for a live brief.

const WORKER_URL = "/web-ai-showcase/models/smoldocling-document/multi-model/summarize-worker.js";

export class Summarizer {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.onProgress = null;
    this._loadWaiters = [];
    this._active = null;
    this._id = 0;
    this.worker.addEventListener("message", (e) => this._onMessage(e.data));
    this.worker.addEventListener("error", (e) => {
      const err = new Error(e.message || "Worker failed to start");
      for (const w of this._loadWaiters) w.reject(err);
      this._loadWaiters = [];
      if (this._active) {
        this._active.reject(err);
        this._active = null;
      }
    });
  }

  _onMessage(msg) {
    switch (msg.type) {
      case "progress":
        this.onProgress?.(msg.p);
        break;
      case "ready":
        this.ready = true;
        for (const w of this._loadWaiters) w.resolve(msg.device);
        this._loadWaiters = [];
        break;
      case "token":
        if (this._active && this._active.id === msg.id) this._active.onToken?.(msg.token);
        break;
      case "done":
        if (this._active && this._active.id === msg.id) {
          this._active.resolve({ ms: msg.ms, tokens: msg.tokens });
          this._active = null;
        }
        break;
      case "error":
        if (this._active && this._active.id === msg.id) {
          this._active.reject(new Error(msg.message));
          this._active = null;
        } else {
          for (const w of this._loadWaiters) w.reject(new Error(msg.message));
          this._loadWaiters = [];
        }
        break;
    }
  }

  load(onProgress) {
    if (onProgress) this.onProgress = onProgress;
    if (this.ready) return Promise.resolve("webgpu");
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load" });
    });
  }

  /** Summarise a Markdown document. onToken(token) streams. Returns { ms, tokens }. */
  summarize(doc, { maxTokens = 180, onToken } = {}) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._active = { id, onToken, resolve, reject };
      this.worker.postMessage({ type: "summarize", id, doc, maxTokens });
    });
  }
}
