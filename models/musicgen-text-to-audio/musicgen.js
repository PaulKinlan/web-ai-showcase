// Front-end helpers for the MusicGen pages. Keeps each page thin: it owns the worker handshake,
// forwards real token-level generation progress, turns the worker's WAV into a playable/downloadable
// blob URL, and draws the generated waveform. All generation runs in worker.js (off the main thread).

const WORKER_URL = "/web-ai-showcase/models/musicgen-text-to-audio/worker.js";

export class MusicGenEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.onProgress = null; // model-download progress
    this._loadWaiters = [];
    this._pending = new Map(); // id -> { resolve, reject, onGen }
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
      for (const w of this._loadWaiters) w.resolve(msg.device);
      this._loadWaiters = [];
    } else if (msg.type === "gen") {
      this._pending.get(msg.id)?.onGen?.(msg.done, msg.total);
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

  /** Load the model. Resolves with the backend string ("wasm"). */
  load(onProgress) {
    if (onProgress) this.onProgress = onProgress;
    if (this.ready) return Promise.resolve(this.device);
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load" });
    });
  }

  /**
   * Generate music from a text prompt.
   * @param {string} prompt
   * @param {{seconds?:number, guidanceScale?:number, temperature?:number}} opts
   * @param {(done:number,total:number)=>void} [onGen] real token-level progress
   * @returns {Promise<{pcm:Float32Array, wav:ArrayBuffer, rate:number, durSec:number, ms:number,
   *   rtf:number, tokens:number, maxLength:number, guidanceScale:number, temperature:number, device:string}>}
   */
  generate(prompt, opts = {}, onGen) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject, onGen });
      this.worker.postMessage({ type: "run", id, prompt, opts });
    });
  }
}

/** Turn the worker's WAV ArrayBuffer into an object URL for an <audio> element / download link. */
export function wavUrl(wavBuffer) {
  return URL.createObjectURL(new Blob([wavBuffer], { type: "audio/wav" }));
}

/** Draw a mono waveform into a <canvas>, matching the light/dark design system. progress in 0..1. */
export function drawWaveform(canvas, pcm, progress = 0) {
  const cs = getComputedStyle(document.body);
  const accent = cs.getPropertyValue("--accent").trim() || "#4b3aff";
  const muted = cs.getPropertyValue("--muted").trim() || "#888";
  const dpr = self.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600;
  const h = canvas.clientHeight || 110;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!pcm || !pcm.length) return;
  const mid = h / 2;
  const step = Math.max(1, Math.floor(pcm.length / w));
  const cut = Math.floor(w * progress);
  for (let x = 0; x < w; x++) {
    let min = 1, max = -1;
    for (let i = 0; i < step; i++) {
      const v = pcm[x * step + i] ?? 0;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    ctx.strokeStyle = x <= cut ? accent : muted;
    ctx.globalAlpha = x <= cut ? 0.95 : 0.5;
    ctx.beginPath();
    ctx.moveTo(x + 0.5, mid + min * mid * 0.95);
    ctx.lineTo(x + 0.5, mid + max * mid * 0.95);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** MusicGen-specific widget styles, injected once per page (keeps us on the shared design system). */
export const MUSICGEN_CSS = `
.wave-wrap { margin:.6rem 0; }
.wave { inline-size:100%; block-size:120px; display:block; background:var(--bg-raised);
  border:1px solid var(--border); border-radius:var(--radius); }
.mg-grid { display:flex; flex-wrap:wrap; gap:1rem; align-items:flex-end; margin:.6rem 0; }
.mg-grid label { display:flex; flex-direction:column; gap:.25rem; font-size:.82rem; }
.mg-grid input[type=range] { inline-size:150px; }
#prompt, .mg-prompt { inline-size:100%; min-block-size:64px; }
.audio-row { display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; margin:.5rem 0; }
.audio-row audio { block-size:36px; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem;
  color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.chip { font:inherit; font-size:.78rem; padding:.2rem .6rem; border-radius:999px;
  border:1px solid var(--border); background:var(--bg-raised); color:var(--color); cursor:pointer; }
.chip:hover { border-color:var(--accent); }
.sample-row { display:flex; flex-wrap:wrap; gap:.4rem; margin:.4rem 0; }
.gen-progress { margin:.5rem 0; }
.gen-progress progress { inline-size:100%; block-size:.6rem; }
.heavy-note { border:1px solid var(--warn); border-radius:var(--radius); background:var(--bg-raised);
  padding:.7rem .9rem; font-size:.85rem; margin:.6rem 0; }
.heavy-note strong { color:var(--warn); }
.fallback { border:1px solid var(--warn); border-radius:var(--radius); background:var(--bg-raised); padding:1rem; }
`;
