// Front-end helpers for the Kokoro pages. Keeps each page thin: worker handshake, the real voice
// catalogue from the model, a WAV blob for the <audio> element, and a waveform draw. All synthesis
// runs in worker.js (off the main thread) via the kokoro-js library.

const WORKER_URL = "/web-ai-showcase/models/kokoro-text-to-speech/worker.js";

export class KokoroEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.voices = [];
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
      this.voices = msg.voices || [];
      for (const w of this._loadWaiters) w.resolve(msg);
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

  /** Load the model. Resolves with { device, voices }. */
  load(onProgress, device) {
    if (onProgress) this.onProgress = onProgress;
    if (this.ready) return Promise.resolve({ device: this.device, voices: this.voices });
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load", device });
    });
  }

  /** Synthesize speech. Returns { wav, pcm, rate, durSec, ms, rtf, device, voice }. */
  speak(text, voice, speed = 1) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, text, voice, speed });
    });
  }
}

/** Turn the worker's WAV ArrayBuffer into an object URL for an <audio> element. */
export function wavUrl(wavBuffer) {
  return URL.createObjectURL(new Blob([wavBuffer], { type: "audio/wav" }));
}

/** Group the flat voice list by gender+language for a readable <optgroup> picker. */
export function groupVoices(voices) {
  const groups = new Map();
  for (const v of voices) {
    const lang = /^b/.test(v.id)
      ? "British English"
      : /^a/.test(v.id)
      ? "American English"
      : (v.language || "Other");
    const g = v.gender ? ` · ${v.gender}` : "";
    const key = `${lang}${g}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(v);
  }
  return groups;
}

/** Draw a mono waveform into a <canvas>, matching the light/dark design system. progress in 0..1. */
export function drawWaveform(canvas, pcm, progress = 0) {
  const cs = getComputedStyle(document.body);
  const accent = cs.getPropertyValue("--accent").trim() || "#4b3aff";
  const muted = cs.getPropertyValue("--muted").trim() || "#888";
  const dpr = self.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600;
  const h = canvas.clientHeight || 96;
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

/** Kokoro-specific widget styles, injected once per page (keeps us on the shared design system). */
export const KOKORO_CSS = `
.wave-wrap { margin:.6rem 0; }
.wave { inline-size:100%; block-size:110px; display:block; background:var(--bg-raised);
  border:1px solid var(--border); border-radius:var(--radius); }
.tts-grid { display:flex; flex-wrap:wrap; gap:.8rem; align-items:flex-end; margin:.6rem 0; }
.tts-grid label { display:flex; flex-direction:column; gap:.25rem; font-size:.82rem; }
#voice { min-inline-size:220px; }
.speed-row { display:flex; align-items:center; gap:.5rem; }
.speed-row input[type=range] { inline-size:150px; }
.audio-row { display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; margin:.5rem 0; }
.audio-row audio { block-size:36px; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem;
  color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
#text { inline-size:100%; min-block-size:96px; }
.fallback { border:1px solid var(--warn); border-radius:var(--radius); background:var(--bg-raised); padding:1rem; }
.fallback code { background:var(--bg-secondary); padding:.05rem .3rem; border-radius:4px; }
.chip { font:inherit; font-size:.78rem; padding:.2rem .6rem; border-radius:999px;
  border:1px solid var(--border); background:var(--bg-raised); color:var(--color); cursor:pointer; }
.chip:hover { border-color:var(--accent); }
.sample-row { display:flex; flex-wrap:wrap; gap:.4rem; margin:.4rem 0; }
`;
