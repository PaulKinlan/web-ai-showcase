// Front-end helpers for the Metric3D pages. Keeps pages thin: owns the worker handshake and re-uses the
// Depth Anything page's colour-map / canvas / parallax helpers (the depth-map format is identical), so
// the depth demos stay visually consistent with one source of truth. ALL inference AND the dense colour
// composite live in this model's worker.js (off the main thread, invariant 15) — the worker transfers a
// ready-to-blit ImageBitmap plus the raw metric-depth field.
//
// Why Metric3D is its OWN family, not a Depth Anything skin: Depth Anything / DPT emit a SMOOTH, RELATIVE
// depth map in arbitrary units re-normalized per image. Metric3D predicts ABSOLUTE depth in METRES
// against a canonical camera — its predicted_depth carries real distances, so you can read "how far" and
// compare distances across the scene, not just "nearer/farther". It runs on WebGPU (fp16) or WASM (fp32).

import {
  COLORMAPS,
  DEPTH_CSS,
  drawDepthBitmap,
  escapeHTML,
  fileToDataURL,
  fitCanvas,
  paintDepth,
  parallaxWarp,
  renderColorLegend,
  renderDepthColor,
  urlToDataURL,
} from "/web-ai-showcase/models/depth-anything/depth.js";

export {
  COLORMAPS,
  DEPTH_CSS,
  drawDepthBitmap,
  escapeHTML,
  fileToDataURL,
  fitCanvas,
  paintDepth,
  parallaxWarp,
  renderColorLegend,
  renderDepthColor,
  urlToDataURL,
};

const WORKER_URL = "/web-ai-showcase/models/metric3d-depth/worker.js";

export class Metric3DEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.dtype = null;
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
        msg.depth = new Uint8Array(msg.depth); // rehydrate transferred buffer
        if (msg.metricField) msg.metricField = new Float32Array(msg.metricField);
        p.resolve(msg);
      }
    } else if (msg.type === "recolored") {
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

  /** → { width,height, bitmap:ImageBitmap, depth:Uint8Array, metricField:Float32Array, metricMin,
   *       metricMax, metricMean, metricW, metricH, hist, rawDims, ms, device, dtype } */
  estimate(imageURL, cmap = "turbo") {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, image: imageURL, cmap });
    });
  }

  /** → { bitmap:ImageBitmap } — re-colourise the last depth map off the main thread. */
  recolor(cmap) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "recolor", id, cmap });
    });
  }
}
