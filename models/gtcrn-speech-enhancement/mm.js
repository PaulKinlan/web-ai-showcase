// Front-end engine for the GTCRN multi-model (denoise → Whisper) page. Both models run off the main
// thread in mm-worker.js.
const WORKER_URL = "/web-ai-showcase/models/gtcrn-speech-enhancement/mm-worker.js";
export class DenoiseAsrEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
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
    if (msg.type === "progress") this.onProgress?.(msg.p);
    else if (msg.type === "ready") {
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
  run(noisy) {
    const id = ++this._id;
    const copy = noisy.slice();
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, noisy: copy }, [copy.buffer]);
    });
  }
}
// Word Error Rate (Levenshtein on words) between a reference and hypothesis, normalised.
export function wer(ref, hyp) {
  const r = ref.toLowerCase().replace(/[^a-z0-9 ]/g, "").split(/\s+/).filter(Boolean);
  const h = hyp.toLowerCase().replace(/[^a-z0-9 ]/g, "").split(/\s+/).filter(Boolean);
  const d = Array.from({ length: r.length + 1 }, () => new Array(h.length + 1).fill(0));
  for (let i = 0; i <= r.length; i++) d[i][0] = i;
  for (let j = 0; j <= h.length; j++) d[0][j] = j;
  for (let i = 1; i <= r.length; i++) {
    for (let j = 1; j <= h.length; j++) {
      d[i][j] = r[i - 1] === h[j - 1]
        ? d[i - 1][j - 1]
        : 1 + Math.min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]);
    }
  }
  return r.length ? d[r.length][h.length] / r.length : (h.length ? 1 : 0);
}
export function escapeHTML(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]),
  );
}
