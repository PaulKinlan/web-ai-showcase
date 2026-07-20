// Front-end helpers for the GTCRN speech-enhancement pages. Thin: owns the worker handshake
// (transferring the enhanced audio buffer, zero-copy), decodes/resamples audio to 16 kHz mono, mixes in
// noise for the demos, encodes WAV, draws waveforms, and computes the honest before/after metrics. ALL
// inference + STFT/iSTFT DSP live in worker.js (off the main thread, raw ONNX Runtime Web). Privacy by
// construction: the audio and every enhanced sample never leave the device.
//
// Model: bitsydarel/gtcrn-onnx (gtcrn_simple.onnx, fp32, ~0.54 MB, 16 kHz). GTCRN is a tiny real-time
// speech denoiser — a DISTINCT audio capability (speech enhancement / audio-to-audio) from the built
// ASR / TTS / classification / diarization / VAD demos.

const WORKER_URL = "/web-ai-showcase/models/gtcrn-speech-enhancement/worker.js";
export const SR = 16000;

export class GtcrnEngine {
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
  /** Enhance a Float32 mono 16 kHz signal (transferred). Returns { enhanced, rate, frames, inRms, outRms, ms, device }. */
  enhance(noisy) {
    const id = ++this._id;
    const copy = noisy.slice(); // keep the caller's copy intact; transfer ours
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, noisy: copy }, [copy.buffer]);
    });
  }
}

// ── Audio I/O ──────────────────────────────────────────────────────────────────────────────────
let _ac = null;
function ctx() {
  return _ac || (_ac = new (self.AudioContext || self.webkitAudioContext)({ sampleRate: SR }));
}
/** Decode any browser-supported audio (ArrayBuffer) → Float32 mono at 16 kHz. */
export async function decodeMono16k(arrayBuffer) {
  const ab = await ctx().decodeAudioData(arrayBuffer.slice(0));
  if (ab.sampleRate === SR) return ab.getChannelData(0).slice();
  // linear resample to 16 kHz (the ctx is 16 kHz, so decodeAudioData usually already resamples; this is a fallback)
  const src = ab.getChannelData(0),
    ratio = ab.sampleRate / SR,
    out = new Float32Array(Math.floor(src.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const t = i * ratio, i0 = t | 0, f = t - i0;
    out[i] = (src[i0] || 0) * (1 - f) + (src[i0 + 1] || 0) * f;
  }
  return out;
}
export async function fileToMono16k(file) {
  return decodeMono16k(await file.arrayBuffer());
}
export async function urlToMono16k(url) {
  return decodeMono16k(await (await fetch(url)).arrayBuffer());
}

export function rms(x) {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return Math.sqrt(s / (x.length || 1));
}

/** Deterministic white noise (seedable), unit-ish amplitude. */
export function whiteNoise(len, seed = 1234) {
  const out = new Float32Array(len);
  let s = seed >>> 0;
  for (let i = 0; i < len; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    out[i] = s / 0x7fffffff * 2 - 1;
  }
  return out;
}
/** Mix `noise` into `clean` at a target SNR (dB). Returns a new Float32Array (same length as clean). */
export function mixAtSNR(clean, noise, snrDb) {
  const cr = rms(clean) || 1e-6;
  const nr = rms(noise) || 1e-6;
  const gain = (cr / nr) / Math.pow(10, snrDb / 20);
  const out = new Float32Array(clean.length);
  let peak = 0;
  for (let i = 0; i < clean.length; i++) {
    out[i] = clean[i] + gain * (noise[i % noise.length]);
    if (Math.abs(out[i]) > peak) peak = Math.abs(out[i]);
  }
  if (peak > 1) { for (let i = 0; i < out.length; i++) out[i] /= peak; // avoid clipping
   }
  return out;
}

/** SNR of `x` against a clean reference (dB): 10log10(||clean||² / ||x−clean||²). */
export function snrVsClean(x, clean) {
  let sig = 0, err = 0, n = Math.min(x.length, clean.length);
  for (let i = 0; i < n; i++) {
    sig += clean[i] * clean[i];
    const d = x[i] - clean[i];
    err += d * d;
  }
  return 10 * Math.log10((sig || 1e-12) / (err || 1e-12));
}
/** Estimate the noise floor: RMS of the quietest 10% of 20 ms frames. Works with no clean reference. */
export function noiseFloor(x) {
  const F = Math.floor(SR * 0.02), nf = Math.floor(x.length / F);
  if (nf < 4) return rms(x);
  const es = [];
  for (let f = 0; f < nf; f++) {
    let s = 0;
    for (let i = 0; i < F; i++) {
      const v = x[f * F + i];
      s += v * v;
    }
    es.push(Math.sqrt(s / F));
  }
  es.sort((a, b) => a - b);
  const k = Math.max(1, Math.floor(nf * 0.1));
  let s = 0;
  for (let i = 0; i < k; i++) s += es[i];
  return s / k;
}

// ── WAV + waveform (patterns shared with the TTS demos) ──────────────────────────────────────────
export function wavBlob(samples, rate = SR) {
  const n = samples.length, buf = new ArrayBuffer(44 + n * 2), dv = new DataView(buf);
  const wr = (o, s) => {
    for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i));
  };
  wr(0, "RIFF");
  dv.setUint32(4, 36 + n * 2, true);
  wr(8, "WAVE");
  wr(12, "fmt ");
  dv.setUint32(16, 16, true);
  dv.setUint16(20, 1, true);
  dv.setUint16(22, 1, true);
  dv.setUint32(24, rate, true);
  dv.setUint32(28, rate * 2, true);
  dv.setUint16(32, 2, true);
  dv.setUint16(34, 16, true);
  wr(36, "data");
  dv.setUint32(40, n * 2, true);
  let o = 44;
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    dv.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    o += 2;
  }
  return new Blob([buf], { type: "audio/wav" });
}
export function wavUrl(samples, rate = SR) {
  return URL.createObjectURL(wavBlob(samples, rate));
}

export function drawWaveform(canvas, pcm, color) {
  const dpr = Math.min(2, self.devicePixelRatio || 1);
  const w = canvas.clientWidth || 600, h = canvas.clientHeight || 80;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx2 = canvas.getContext("2d");
  ctx2.scale(dpr, dpr);
  ctx2.clearRect(0, 0, w, h);
  const cs = getComputedStyle(document.documentElement);
  const col = color || cs.getPropertyValue("--accent").trim() || "#4b3aff";
  const mid = h / 2, step = Math.max(1, Math.floor(pcm.length / w));
  ctx2.strokeStyle = col;
  for (let x = 0; x < w; x++) {
    let min = 1, max = -1;
    for (let j = 0; j < step; j++) {
      const v = pcm[x * step + j] || 0;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    ctx2.beginPath();
    ctx2.moveTo(x + 0.5, mid + min * mid * 0.95);
    ctx2.lineTo(x + 0.5, mid + max * mid * 0.95);
    ctx2.stroke();
  }
}

export function escapeHTML(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]),
  );
}

export const GTCRN_CSS = `
.dropzone { border:2px dashed var(--border-strong); border-radius:var(--radius); background:var(--bg-raised);
  padding:.9rem; text-align:center; cursor:pointer; transition:border-color .15s, background .15s; font-size:.85rem; }
.dropzone.drag { border-color:var(--accent); background:var(--bg-secondary); }
.dropzone:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.sample-row { display:flex; flex-wrap:wrap; gap:.4rem; margin:.5rem 0; }
.chip { font:inherit; font-size:.8rem; padding:.45rem .8rem; border-radius:999px; border:1px solid var(--border);
  background:var(--bg-raised); color:var(--color); cursor:pointer; min-block-size:2.4rem; }
.chip:hover, .chip:focus-visible { border-color:var(--accent); }
.chip.active { border-color:var(--accent); background:var(--bg-secondary); }
.ab-grid { display:grid; grid-template-columns:1fr 1fr; gap:.8rem; margin:.6rem 0; }
@media (max-width:620px){ .ab-grid { grid-template-columns:1fr; } }
.ab-card { border:1px solid var(--border); border-radius:8px; background:var(--bg-raised); padding:.6rem; min-inline-size:0; }
.ab-card.enh { border-color:var(--accent); }
.ab-card h4 { margin:0 0 .4rem; font-size:.85rem; display:flex; justify-content:space-between; align-items:baseline; }
.ab-card h4 .tag { font-family:var(--font-mono); font-size:.72rem; color:var(--muted); }
canvas.wave { inline-size:100%; block-size:80px; display:block; border-radius:6px; background:var(--bg-secondary); }
.ab-card audio { inline-size:100%; margin-top:.5rem; }
.slider-row { display:flex; align-items:center; gap:.6rem; margin:.5rem 0; flex-wrap:wrap; }
.slider-row input[type=range] { flex:1 1 160px; accent-color:var(--accent); min-inline-size:0; }
.slider-row output { font-family:var(--font-mono); font-size:.82rem; min-inline-size:4ch; }
.field-row { display:flex; flex-wrap:wrap; gap:.6rem; align-items:center; margin:.6rem 0; }
.field-row label { display:flex; flex-direction:column; gap:.25rem; font-size:.82rem; }
.metric-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.6rem; margin:.6rem 0; }
.metric { border:1px solid var(--border); border-radius:8px; padding:.6rem .7rem; background:var(--bg-raised); }
.metric .k { font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.03em; }
.metric .v { font-family:var(--font-mono); font-size:1.2rem; font-weight:600; }
.metric .v.good { color:var(--ok,#1a7a3a); }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem; color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.inside-table { inline-size:100%; border-collapse:collapse; font-size:.82rem; margin-top:.5rem; }
.inside-table th,.inside-table td { text-align:left; padding:.3rem .5rem; border-bottom:1px solid var(--border); font-family:var(--font-mono); }
.inside-table th { color:var(--muted); font-weight:600; }
.rec-dot { inline-size:.7rem; block-size:.7rem; border-radius:50%; background:var(--bad,#c0392b); display:inline-block; margin-inline-end:.4rem; animation:recpulse 1s ease-in-out infinite; }
@keyframes recpulse { 50% { opacity:.25; } }
@media (prefers-reduced-motion: reduce){ .rec-dot { animation:none; } }
`;
