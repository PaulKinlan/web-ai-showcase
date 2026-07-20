// Front-end engine for the ALIKED+LightGlue multi-model (retrieve+verify) page. DINOv2 ranking +
// ALIKED/LightGlue verification run off the main thread in mm-worker.js.
const WORKER_URL = "/web-ai-showcase/models/aliked-lightglue-matching/mm-worker.js";
export class RetrieveVerifyEngine {
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
      case "ranked":
        if (this._active && this._active.id === msg.id) this._active.onRanked?.(msg.ranked);
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
    return new Promise((res, rej) => {
      this._loadWaiters.push({ resolve: res, reject: rej });
      this.worker.postMessage({ type: "load" });
    });
  }
  run(queryBitmap, galleryBitmaps, { onRanked } = {}) {
    const id = ++this._id;
    return new Promise((res, rej) => {
      this._active = { id, onRanked, resolve: res, reject: rej };
      this.worker.postMessage({ type: "run", id, query: queryBitmap, gallery: galleryBitmaps }, [
        queryBitmap,
        ...galleryBitmaps,
      ]);
    });
  }
}
export function escapeHTML(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]),
  );
}
