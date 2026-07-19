// Front-end helpers for the RT-DETRv2 pages. Keeps each page thin: it owns the worker handshake and
// re-uses the DETR page's drawing/colour/table helpers (the box format is identical), so every
// detector page stays visually consistent and there is one source of truth for the canvas overlay.
// All inference lives in this model's worker.js (off the main thread).
//
// Why RT-DETRv2 is its OWN entry, DISTINCT from our RT-DETR(v1) page: v2 keeps the exact same r18vd
// backbone, the same 80 COCO classes and the same NMS-free, real-time design, but redesigns the
// DECODER. v1's deformable-attention decoder uses a fixed sampling scheme and the `grid_sample`
// operator; v2 makes the number of sampling points **selectable per feature scale** ("flexible
// decoder") and replaces `grid_sample` with a **discrete sampling** operator that exports/deploys
// cleanly — then trains with a "bag of freebies" recipe. The result: cleaner, better-localised boxes
// at the SAME inference budget. This page runs v2 and lets you race it against v1 on the same image
// to see the difference honestly.

import {
  BOX_PALETTE,
  colorForLabel,
  countByLabel,
  DETR_CSS,
  drawDetections,
  escapeHTML,
  fileToDataURL,
} from "/web-ai-showcase/models/detr-object-detection/detr.js";

const WORKER_URL = "/web-ai-showcase/models/rtdetr-v2-r18vd/worker.js";

export class RtDetrV2Engine {
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

  detect(imageURL) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, image: imageURL });
    });
  }
}

// Re-export the shared detection helpers so RT-DETRv2 pages import everything from one module.
export {
  BOX_PALETTE,
  colorForLabel,
  countByLabel,
  DETR_CSS,
  drawDetections,
  escapeHTML,
  fileToDataURL,
};
