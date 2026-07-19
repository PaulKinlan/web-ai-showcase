// Front-end helpers for the D-FINE pages. Keeps each page thin: it owns the worker handshake and
// re-uses the DETR page's drawing/colour/table helpers (the box format is identical), so all three
// detectors (DETR, RT-DETR, D-FINE) stay visually consistent with one source of truth for the canvas
// overlay. All inference lives in this model's worker.js (off the main thread).
//
// Why D-FINE is its OWN family, not another DETR/RT-DETR skin: D-FINE keeps RT-DETR's efficient hybrid
// encoder and end-to-end, NMS-free set-prediction decoder, but replaces direct box-coordinate
// regression with Fine-grained Distribution Refinement (FDR) — each box edge is a probability
// distribution over discretized offsets, iteratively sharpened layer by layer — plus Global Optimal
// Localization Self-Distillation (GO-LSD), which distils the final refined distributions back into the
// earlier layers. That yields higher COCO accuracy than RT-DETR at the same real-time latency, with no
// extra inference cost, still small enough (dfine_s, ~10 MB q8) to run in a tab.

import {
  BOX_PALETTE,
  colorForLabel,
  countByLabel,
  DETR_CSS,
  drawDetections,
  escapeHTML,
  fileToDataURL,
} from "/web-ai-showcase/models/detr-object-detection/detr.js";

const WORKER_URL = "/web-ai-showcase/models/dfine-detection/worker.js";

export class DFineEngine {
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

// Re-export the shared detection helpers so D-FINE pages import everything from one module.
export {
  BOX_PALETTE,
  colorForLabel,
  countByLabel,
  DETR_CSS,
  drawDetections,
  escapeHTML,
  fileToDataURL,
};
