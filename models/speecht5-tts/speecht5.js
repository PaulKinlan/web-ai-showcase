// Front-end helpers for the SpeechT5 pages. Keeps each page thin: worker handshake, the speaker
// catalogue (real CMU-Arctic x-vectors bundled next to this file), a WAV blob for the <audio> element,
// and the "see inside" draws (waveform, spectrogram of the real output, and the speaker x-vector).
// All synthesis runs in worker.js (off the main thread) via the transformers.js text-to-speech pipeline.

const BASE = "/web-ai-showcase/models/speecht5-tts";
const WORKER_URL = `${BASE}/worker.js`;

// Six distinct CMU-Arctic speakers. Each .bin is a 512-dim x-vector the SpeechT5 model was trained
// against — swapping it changes the voice. Bundled locally so the demo works offline once cached.
export const SPEAKERS = [
  { id: "slt", name: "Sarah (US, female)", accent: "US English", gender: "female" },
  { id: "clb", name: "Claire (US, female)", accent: "US English", gender: "female" },
  { id: "bdl", name: "Brian (US, male)", accent: "US English", gender: "male" },
  { id: "rms", name: "Ray (US, male)", accent: "US English", gender: "male" },
  { id: "awb", name: "Alan (Scottish, male)", accent: "Scottish English", gender: "male" },
  { id: "ksp", name: "Kabir (Indian, male)", accent: "Indian English", gender: "male" },
].map((s) => ({ ...s, url: `${BASE}/speakers/${s.id}.bin` }));

export function speakerById(id) {
  return SPEAKERS.find((s) => s.id === id) || SPEAKERS[0];
}

export class SpeechT5Engine {
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
    if (msg.type === "progress") {
      this.onProgress?.(msg.p);
    } else if (msg.type === "ready") {
      this.ready = true;
      this.device = msg.device;
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

  /** Load the model. Resolves with { device }. */
  load(onProgress, device) {
    if (onProgress) this.onProgress = onProgress;
    if (this.ready) return Promise.resolve({ device: this.device });
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load", device });
    });
  }

  /** Synthesize. `spk` is a SPEAKERS entry. Returns { wav, pcm, rate, durSec, ms, rtf, device,
   *  voice, emb, spec, specFrames, specBins }. */
  speak(text, spk) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, text, spk });
    });
  }
}

/** Turn the worker's WAV ArrayBuffer into an object URL for an <audio> element. */
export function wavUrl(wavBuffer) {
  return URL.createObjectURL(new Blob([wavBuffer], { type: "audio/wav" }));
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function tokens() {
  const cs = getComputedStyle(document.body);
  return {
    accent: cs.getPropertyValue("--accent").trim() || "#4b3aff",
    muted: cs.getPropertyValue("--muted").trim() || "#888",
    raised: cs.getPropertyValue("--bg-raised").trim() || "#fff",
  };
}

/** Draw a mono waveform into a <canvas>, matching the design system. progress in 0..1. */
export function drawWaveform(canvas, pcm, progress = 0) {
  const { accent, muted } = tokens();
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

/** Heatmap of the REAL spectrogram of the generated audio (frames × freq bins). */
export function drawSpectrogram(canvas, spec, frames, bins) {
  const { accent, muted, raised } = tokens();
  const dpr = self.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600;
  const h = canvas.clientHeight || 120;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = raised;
  ctx.fillRect(0, 0, w, h);
  if (!spec || !frames || !bins) return;
  let max = 1e-6;
  for (let i = 0; i < spec.length; i++) if (spec[i] > max) max = spec[i];
  const [ar, ag, ab] = hexToRgb(accent);
  const cellW = w / frames;
  const cellH = h / bins;
  for (let f = 0; f < frames; f++) {
    for (let b = 0; b < bins; b++) {
      const v = spec[f * bins + b] / max; // 0..1
      ctx.fillStyle = `rgba(${ar},${ag},${ab},${(v * v).toFixed(3)})`;
      // low freq at the bottom
      ctx.fillRect(f * cellW, h - (b + 1) * cellH, Math.ceil(cellW), Math.ceil(cellH));
    }
  }
  ctx.fillStyle = muted;
  ctx.font = "11px system-ui, sans-serif";
  ctx.fillText(`${frames} frames × ${bins} bins · spectrogram of the generated audio`, 6, 13);
}

/** Draw the 512-dim speaker x-vector as a diverging bar strip: this is the "voice fingerprint". */
export function drawEmbedding(canvas, emb) {
  const { accent, muted } = tokens();
  const dpr = self.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600;
  const h = canvas.clientHeight || 70;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!emb || !emb.length) return;
  let max = 1e-6;
  for (let i = 0; i < emb.length; i++) max = Math.max(max, Math.abs(emb[i]));
  const mid = h / 2;
  const bw = w / emb.length;
  for (let i = 0; i < emb.length; i++) {
    const v = emb[i] / max; // -1..1
    const bh = Math.abs(v) * (mid - 2);
    ctx.fillStyle = v >= 0 ? accent : muted;
    ctx.globalAlpha = 0.85;
    if (v >= 0) ctx.fillRect(i * bw, mid - bh, Math.max(0.6, bw), bh);
    else ctx.fillRect(i * bw, mid, Math.max(0.6, bw), bh);
  }
  ctx.globalAlpha = 1;
  ctx.strokeStyle = muted;
  ctx.globalAlpha = 0.4;
  ctx.beginPath();
  ctx.moveTo(0, mid);
  ctx.lineTo(w, mid);
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function hexToRgb(hex) {
  const m = hex.replace("#", "");
  const n = m.length === 3 ? m.split("").map((c) => c + c).join("") : m;
  const int = parseInt(n, 16);
  return [(int >> 16) & 255, (int >> 8) & 255, int & 255];
}

/** SpeechT5-specific widget styles, injected once per page (keeps us on the shared design system). */
export const SPEECHT5_CSS = `
.wave-wrap { margin:.6rem 0; }
.wave { inline-size:100%; block-size:110px; display:block; background:var(--bg-raised);
  border:1px solid var(--border); border-radius:var(--radius); }
.spec { inline-size:100%; block-size:130px; display:block; border:1px solid var(--border);
  border-radius:var(--radius); }
.emb { inline-size:100%; block-size:72px; display:block; background:var(--bg-raised);
  border:1px solid var(--border); border-radius:var(--radius); }
.tts-grid { display:flex; flex-wrap:wrap; gap:.8rem; align-items:flex-end; margin:.6rem 0; }
.tts-grid label { display:flex; flex-direction:column; gap:.25rem; font-size:.82rem; }
#voice, #voiceA, #voiceB { min-inline-size:200px; }
.audio-row { display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; margin:.5rem 0; }
.audio-row audio { block-size:36px; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem;
  color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
#text { inline-size:100%; min-block-size:96px; }
.chip { font:inherit; font-size:.78rem; padding:.2rem .6rem; border-radius:999px;
  border:1px solid var(--border); background:var(--bg-raised); color:var(--color); cursor:pointer; }
.chip:hover { border-color:var(--accent); }
.sample-row { display:flex; flex-wrap:wrap; gap:.4rem; margin:.4rem 0; }
.viz-note { font-family:var(--font-mono); font-size:.78rem; color:var(--muted); margin:.3rem 0 .1rem; }
`;
