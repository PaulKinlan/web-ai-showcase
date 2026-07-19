// Front-end helpers for the Speech Commands (keyword-spotting) pages. Keeps each page thin: it owns the
// worker handshake, decodes/records audio into the 16 kHz mono Float32Array the model wants, draws the
// waveform + the log-mel spectrogram heatmap, and renders the top-command score bars. All inference
// lives in worker.js (off the main thread). This is an AST fine-tuned on the Google Speech Commands v2
// set — same Audio-Spectrogram-Transformer architecture as the AudioSet demo, retrained to recognise 35
// short spoken keywords (yes/no/up/down/stop/go…) instead of 527 general sound classes.

const WORKER_URL = "/web-ai-showcase/models/speech-commands/worker.js";
const TARGET_RATE = 16000;

export class SpeechEngine {
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

  load(onProgress, device) {
    if (onProgress) this.onProgress = onProgress;
    if (this.ready) return Promise.resolve(this.device);
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load", device });
    });
  }

  /** Classify a 16 kHz mono Float32Array. Returns { labels, spectrogram, ms, device, durationS }. */
  classify(audio, opts) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, audio, opts });
    });
  }
}

let _audioCtx = null;
function audioCtx() {
  if (!_audioCtx) {
    const AC = self.AudioContext || self.webkitAudioContext;
    _audioCtx = new AC();
  }
  return _audioCtx;
}

/** Decode any browser-supported audio ArrayBuffer to a 16 kHz mono Float32Array. */
export async function decodeToMono16k(arrayBuffer) {
  const decoded = await audioCtx().decodeAudioData(arrayBuffer.slice(0));
  const frames = Math.ceil(decoded.duration * TARGET_RATE);
  const off = new OfflineAudioContext(1, frames, TARGET_RATE);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return { pcm: rendered.getChannelData(0), duration: decoded.duration };
}

export async function urlToMono16k(url) {
  const buf = await (await fetch(url)).arrayBuffer();
  return decodeToMono16k(buf);
}

export async function blobToMono16k(blob) {
  return decodeToMono16k(await blob.arrayBuffer());
}

/** A tiny mic recorder: start() then stop() → { blob, url }. Honest about missing mic support. */
export class MicRecorder {
  constructor() {
    this.rec = null;
    this.chunks = [];
    this.stream = null;
  }
  static supported() {
    return !!(navigator.mediaDevices?.getUserMedia && self.MediaRecorder);
  }
  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.chunks = [];
    this.rec = new MediaRecorder(this.stream);
    this.rec.addEventListener("dataavailable", (e) => {
      if (e.data.size) this.chunks.push(e.data);
    });
    this.rec.start();
  }
  stop() {
    return new Promise((resolve) => {
      this.rec.addEventListener("stop", () => {
        const type = this.rec.mimeType || "audio/webm";
        const blob = new Blob(this.chunks, { type });
        this.stream.getTracks().forEach((t) => t.stop());
        this.stream = null;
        resolve({ blob, url: URL.createObjectURL(blob) });
      }, { once: true });
      this.rec.stop();
    });
  }
}

/**
 * A live microphone listener. Keeps a rolling window of the last `windowS` seconds at 16 kHz and calls
 * `onWindow(Float32Array)` every `hopS` seconds, so a page can classify continuously. Honest about a
 * missing mic. Uses a ScriptProcessor tap (universally supported) and resamples to 16 kHz if needed.
 */
export class LiveListener {
  constructor({ windowS = 2, hopS = 1, onWindow } = {}) {
    this.windowS = windowS;
    this.hopS = hopS;
    this.onWindow = onWindow;
    this.running = false;
    this._ctx = null;
    this._stream = null;
    this._node = null;
    this._src = null;
    this._ring = null;
    this._write = 0;
    this._sinceHop = 0;
  }
  static supported() {
    return !!navigator.mediaDevices?.getUserMedia;
  }
  async start() {
    this._stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const AC = self.AudioContext || self.webkitAudioContext;
    this._ctx = new AC();
    const sr = this._ctx.sampleRate;
    this._srcRate = sr;
    this._ringLen = Math.ceil(this.windowS * sr);
    this._ring = new Float32Array(this._ringLen);
    this._write = 0;
    this._filled = 0;
    this._hopSamples = Math.ceil(this.hopS * sr);
    this._src = this._ctx.createMediaStreamSource(this._stream);
    this._node = this._ctx.createScriptProcessor(4096, 1, 1);
    this._node.onaudioprocess = (e) => {
      const inBuf = e.inputBuffer.getChannelData(0);
      for (let i = 0; i < inBuf.length; i++) {
        this._ring[this._write] = inBuf[i];
        this._write = (this._write + 1) % this._ringLen;
      }
      this._filled = Math.min(this._filled + inBuf.length, this._ringLen);
      this._sinceHop += inBuf.length;
      if (this._sinceHop >= this._hopSamples && this._filled >= this._ringLen) {
        this._sinceHop = 0;
        this._emit();
      }
    };
    this._src.connect(this._node);
    // Route through a zero-gain node so the processor runs without echoing audio to the speakers.
    const mute = this._ctx.createGain();
    mute.gain.value = 0;
    this._node.connect(mute);
    mute.connect(this._ctx.destination);
    this.running = true;
  }
  _emit() {
    // Unwrap the ring into chronological order, then resample to 16 kHz if needed.
    const win = new Float32Array(this._ringLen);
    for (let i = 0; i < this._ringLen; i++) win[i] = this._ring[(this._write + i) % this._ringLen];
    let out = win;
    if (this._srcRate !== TARGET_RATE) {
      const n = Math.round((win.length * TARGET_RATE) / this._srcRate);
      out = new Float32Array(n);
      const ratio = win.length / n;
      for (let i = 0; i < n; i++) {
        const t = i * ratio, i0 = Math.floor(t), f = t - i0;
        out[i] = (win[i0] ?? 0) * (1 - f) + (win[i0 + 1] ?? win[i0] ?? 0) * f;
      }
    }
    this.onWindow?.(out);
  }
  stop() {
    this.running = false;
    try {
      this._node && (this._node.onaudioprocess = null);
    } catch {}
    try {
      this._src?.disconnect();
      this._node?.disconnect();
    } catch {}
    try {
      this._stream?.getTracks().forEach((t) => t.stop());
    } catch {}
    try {
      this._ctx?.close();
    } catch {}
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Draw a mono waveform into a <canvas>, matching the design system's accent/muted colours. */
export function drawWaveform(canvas, pcm) {
  const cs = getComputedStyle(document.body);
  const accent = cs.getPropertyValue("--accent").trim() || "#4b3aff";
  const dpr = self.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600, h = canvas.clientHeight || 80;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!pcm || !pcm.length) return;
  const mid = h / 2, step = Math.max(1, Math.floor(pcm.length / w));
  ctx.strokeStyle = accent;
  ctx.globalAlpha = 0.9;
  for (let x = 0; x < w; x++) {
    let min = 1, max = -1;
    for (let i = 0; i < step; i++) {
      const v = pcm[x * step + i] ?? 0;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    ctx.beginPath();
    ctx.moveTo(x + 0.5, mid + min * mid * 0.95);
    ctx.lineTo(x + 0.5, mid + max * mid * 0.95);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

// Perceptual magma-ish ramp (dark → purple → orange → pale), readable in light and dark.
const MAGMA = [
  [0, 0, 4],
  [40, 11, 84],
  [101, 21, 110],
  [159, 42, 99],
  [212, 72, 66],
  [245, 125, 21],
  [250, 193, 39],
  [252, 253, 191],
];
function ramp(t) {
  t = Math.max(0, Math.min(1, t));
  const x = t * (MAGMA.length - 1);
  const i = Math.floor(x), f = x - i;
  const a = MAGMA[i], b = MAGMA[Math.min(i + 1, MAGMA.length - 1)];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/**
 * Draw a log-mel spectrogram ({data,frames,mels,min,max}) as a heatmap: time on x, frequency on y
 * (low frequencies at the bottom). This is the real 2D image AST classifies.
 */
export function drawSpectrogram(canvas, spec) {
  const { data, frames, mels, min, max } = spec;
  canvas.width = frames;
  canvas.height = mels;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(frames, mels);
  const rng = max - min || 1;
  for (let f = 0; f < frames; f++) {
    for (let m = 0; m < mels; m++) {
      const v = (data[f * mels + m] - min) / rng;
      const [r, g, b] = ramp(v);
      // flip vertically so low mel bands sit at the bottom
      const y = mels - 1 - m;
      const idx = (y * frames + f) * 4;
      img.data[idx] = r;
      img.data[idx + 1] = g;
      img.data[idx + 2] = b;
      img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

/** Render top labels as accessible score bars into `container`. */
export function renderScoreBars(container, labels, max = 8) {
  container.replaceChildren(
    ...labels.slice(0, max).map(({ label, score }, i) => {
      const row = document.createElement("div");
      row.className = "score-row" + (i === 0 ? " top" : "");
      const pct = (score * 100).toFixed(1);
      row.innerHTML = `<span class="score-label">${escapeHTML(label)}</span>` +
        `<span class="score-track"><span class="score-fill" style="inline-size:${
          Math.max(2, score * 100)
        }%"></span></span>` +
        `<span class="score-val">${pct}%</span>`;
      return row;
    }),
  );
}

export const SPEECH_CSS = `
.wave-wrap { margin:.6rem 0; }
.wave { inline-size:100%; block-size:80px; display:block; background:var(--bg-raised);
  border:1px solid var(--border); border-radius:var(--radius); }
.spec-canvas { inline-size:100%; block-size:auto; image-rendering:pixelated; display:block;
  border:1px solid var(--border); border-radius:var(--radius); background:var(--bg-raised);
  max-block-size:40vh; object-fit:fill; }
.spec-canvas:focus-visible { outline:3px solid var(--accent); outline-offset:2px; }
.audio-row { display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; margin:.5rem 0; }
.audio-row audio { block-size:34px; }
.rec-dot { inline-size:.7rem; block-size:.7rem; border-radius:50%; background:var(--bad,#c0392b);
  display:inline-block; margin-inline-end:.4rem; animation:recpulse 1s ease-in-out infinite; }
@keyframes recpulse { 50% { opacity:.25; } }
@media (prefers-reduced-motion: reduce) { .rec-dot { animation:none; } }
.dropzone { border:2px dashed var(--border-strong); border-radius:var(--radius); background:var(--bg-raised);
  padding:1rem; text-align:center; cursor:pointer; transition:border-color .15s, background .15s; }
.dropzone.drag { border-color:var(--accent); background:var(--bg-secondary); }
.dropzone:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.sample-row { display:flex; flex-wrap:wrap; gap:.4rem; margin:.5rem 0; }
.scores { display:flex; flex-direction:column; gap:.35rem; margin:.5rem 0; }
.score-row { display:grid; grid-template-columns:minmax(8ch,10rem) 1fr 4.5ch; align-items:center; gap:.5rem;
  font-size:.85rem; }
.score-row.top .score-label { font-weight:700; color:var(--accent); }
.score-label { text-align:right; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.score-track { block-size:.85rem; background:var(--bg-secondary); border-radius:999px; overflow:hidden;
  border:1px solid var(--border); }
.score-fill { display:block; block-size:100%; background:var(--accent); border-radius:999px; }
.score-val { font-family:var(--font-mono); font-size:.78rem; color:var(--muted); text-align:right; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem;
  color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.field-row { display:flex; flex-wrap:wrap; gap:.6rem; align-items:center; margin:.6rem 0; }
.fallback { border:1px solid var(--warn); border-radius:var(--radius); background:var(--bg-raised); padding:1rem; }
.fallback code { background:var(--bg-secondary); padding:.05rem .3rem; border-radius:4px; }
.inside-table { inline-size:100%; border-collapse:collapse; font-size:.82rem; margin-top:.5rem; }
.inside-table th, .inside-table td { text-align:left; padding:.3rem .5rem; border-bottom:1px solid var(--border);
  font-family:var(--font-mono); }
.inside-table th { color:var(--muted); font-weight:600; }
`;
