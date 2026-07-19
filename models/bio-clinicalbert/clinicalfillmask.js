// Front-end helpers for the Bio_ClinicalBERT fill-mask pages. Thin: owns the worker handshake + a
// couple of small renderers. All inference (masked-LM logits, softmax, candidate scoring, the
// clinical-vs-general comparison) lives in worker.js, off the main thread.
//
// It RE-USES the general BERT fill-mask page's renderers + styles (probability bars, entropy,
// debounce) so every fill-mask page in the showcase looks and behaves consistently — there is one
// source of truth for the prediction UI.

import {
  debounce,
  escapeHTML,
  FILLMASK_CSS,
  MASK,
  renderPredictions,
  topkEntropy,
} from "/web-ai-showcase/models/bert-fill-mask/fillmask.js";

const WORKER_URL = "/web-ai-showcase/models/bio-clinicalbert/worker.js";

export class ClinicalFillMaskEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false; // clinical (primary) ready
    this.generalReady = false;
    this.device = "wasm";
    this.onProgress = null; // clinical progress
    this.onGeneralProgress = null; // general (comparator) progress
    this._loadWaiters = { clinical: [], general: [] };
    this._pending = new Map();
    this._id = 0;
    this.worker.addEventListener("message", (e) => this._onMessage(e.data));
    this.worker.addEventListener("error", (e) => {
      const err = new Error(e.message || "Worker failed to start");
      for (const k of ["clinical", "general"]) {
        for (const w of this._loadWaiters[k]) w.reject(err);
        this._loadWaiters[k] = [];
      }
      for (const [, p] of this._pending) p.reject(err);
      this._pending.clear();
    });
  }

  _onMessage(msg) {
    if (msg.type === "progress") {
      (msg.which === "general" ? this.onGeneralProgress : this.onProgress)?.(msg.p);
    } else if (msg.type === "ready") {
      if (msg.which === "general") this.generalReady = true;
      else {
        this.ready = true;
        this.device = msg.device;
      }
      const waiters = this._loadWaiters[msg.which] || [];
      for (const w of waiters) w.resolve(msg.device);
      this._loadWaiters[msg.which] = [];
    } else if (msg.type === "fill" || msg.type === "compare" || msg.type === "scores") {
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
        for (const k of ["clinical", "general"]) {
          for (const w of this._loadWaiters[k]) w.reject(new Error(msg.message));
          this._loadWaiters[k] = [];
        }
      }
    }
  }

  load(onProgress) {
    if (onProgress) this.onProgress = onProgress;
    if (this.ready) return Promise.resolve(this.device);
    return new Promise((resolve, reject) => {
      this._loadWaiters.clinical.push({ resolve, reject });
      this.worker.postMessage({ type: "load", which: "clinical" });
    });
  }

  loadGeneral(onProgress) {
    if (onProgress) this.onGeneralProgress = onProgress;
    if (this.generalReady) return Promise.resolve(this.device);
    return new Promise((resolve, reject) => {
      this._loadWaiters.general.push({ resolve, reject });
      this.worker.postMessage({ type: "load", which: "general" });
    });
  }

  _call(payload) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...payload, id });
    });
  }

  /** Fill one text on a chosen model → { text, masks:[{pos, predictions}], maskCount, which, ms, device } */
  fill(text, topk = 8, which = "clinical") {
    return this._call({ type: "fill", text, topk, which });
  }

  /** Run clinical + general on the same text → { clinical:{...}, general:{...}, ms, device } */
  compare(text, topk = 8) {
    return this._call({ type: "compare", text, topk });
  }

  /** Score a fixed candidate set at the first mask on the clinical model → { text, scores, ms, device } */
  scoreCandidates(text, candidates) {
    return this._call({ type: "scoreCandidates", text, candidates });
  }
}

export { debounce, escapeHTML, FILLMASK_CSS, MASK, renderPredictions, topkEntropy };
