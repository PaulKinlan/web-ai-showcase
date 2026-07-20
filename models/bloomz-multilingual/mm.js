// Front-end engine for the BLOOMZ multi-model (RAG) page. MiniLM retrieval + BLOOMZ generation run
// off the main thread in mm-worker.js.
const WORKER_URL = "/web-ai-showcase/models/bloomz-multilingual/mm-worker.js";

export class RagEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.onProgress = null;
    this._loadWaiters = [];
    this._active = null;
    this._id = 0;
    this.worker.addEventListener("message", (e) => this._onMessage(e.data));
    this.worker.addEventListener(
      "error",
      (e) => this._rejectAll(new Error(e.message || "Worker failed to start")),
    );
  }
  _rejectAll(err) {
    for (const w of this._loadWaiters) w.reject(err);
    this._loadWaiters = [];
    if (this._active) {
      this._active.reject(err);
      this._active = null;
    }
  }
  _onMessage(msg) {
    switch (msg.type) {
      case "progress":
        this.onProgress?.(msg.p);
        break;
      case "ready":
        this.ready = true;
        this.device = msg.device;
        for (const w of this._loadWaiters) w.resolve(msg.device);
        this._loadWaiters = [];
        break;
      case "retrieved":
        if (this._active && this._active.id === msg.id) this._active.onRetrieved?.(msg);
        break;
      case "token":
        if (this._active && this._active.id === msg.id) this._active.onToken?.(msg.token);
        break;
      case "done":
        if (this._active && this._active.id === msg.id) {
          this._active.resolve(msg);
          this._active = null;
        }
        break;
      case "error":
        if (this._active && this._active.id === msg.id) {
          this._active.reject(new Error(msg.message));
          this._active = null;
        } else this._rejectAll(new Error(msg.message));
        break;
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
  run(query, notes, { onRetrieved, onToken } = {}) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._active = { id, onRetrieved, onToken, resolve, reject };
      this.worker.postMessage({ type: "run", id, query, notes });
    });
  }
}
export function escapeHTML(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]),
  );
}
