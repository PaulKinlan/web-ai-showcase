// Front-end helpers for the DepthPro pages. Keeps pages thin: owns the worker handshake and re-uses
// the Depth Anything page's colour-map / parallax / canvas helpers (the depth-map format is identical),
// so the two depth demos stay visually consistent with one source of truth. All inference lives in this
// model's worker.js (off the main thread).
//
// Why DepthPro is its OWN family, not a Depth Anything skin: Depth Anything emits a SMOOTH, RELATIVE
// depth map in arbitrary units. DepthPro (Apple) is trained for SHARP, boundary-accurate, METRIC-SCALE
// depth at high resolution (native 1536²) — its predicted_depth carries a consistent absolute scale and
// its object edges stay crisp. It is a large WebGPU-class model (~572 MB q4f16); the page gates on a
// real adapter and degrades honestly (needs-WebGPU) where there isn't one.

import {
  COLORMAPS,
  DEPTH_CSS,
  escapeHTML,
  fileToDataURL,
  fitCanvas,
  parallaxWarp,
  renderColorLegend,
  renderDepthColor,
  sampleMap,
  urlToDataURL,
} from "/web-ai-showcase/models/depth-anything/depth.js";

const WORKER_URL = "/web-ai-showcase/models/depth-pro-metric/worker.js";

export class DepthProEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "webgpu";
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
      this.ready = true;
      this.device = msg.device;
      this.dtype = msg.dtype;
      for (const w of this._loadWaiters) w.resolve(msg.device);
      this._loadWaiters = [];
    } else if (msg.type === "result") {
      const p = this._pending.get(msg.id);
      if (p) {
        this._pending.delete(msg.id);
        msg.depth = new Uint8Array(msg.depth); // rehydrate the transferred buffer
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

  estimate(imageURL) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, image: imageURL });
    });
  }
}

// Re-export the shared depth helpers so DepthPro pages import everything from one module.
export {
  COLORMAPS,
  DEPTH_CSS,
  escapeHTML,
  fileToDataURL,
  fitCanvas,
  parallaxWarp,
  renderColorLegend,
  renderDepthColor,
  sampleMap,
  urlToDataURL,
};
