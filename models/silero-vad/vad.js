// Front-end helpers for the Silero VAD pages. Keeps each page thin: it owns the worker handshake,
// decodes/records audio into the 16 kHz mono Float32Array the model wants, streams live mic frames, and
// draws the probability curve + speech-segment timeline. ALL inference lives in worker.js (off the main
// thread, onnxruntime-web).

const WORKER_URL = "/web-ai-showcase/models/silero-vad/worker.js";
const TARGET_RATE = 16000;
const FRAME = 512;

export class VadEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm (onnxruntime-web)";
    this.onProgress = null;
    this.onStream = null;
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
    } else if (msg.type === "stream") {
      this.onStream?.(msg);
    } else if (msg.type === "stream-ready") {
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

  /** Analyse a whole 16 kHz mono clip. Returns { probs, frameSec, segments, ms, durationS, speechRatio, device }. */
  run(pcm, opts) {
    const id = ++this._id;
    // Copy so we can transfer without detaching the caller's buffer.
    const copy = pcm.slice();
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, pcm: copy, opts }, [copy.buffer]);
    });
  }

  /** Reset the streaming LSTM state before a new live session. */
  streamReset() {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "stream-reset", id });
    });
  }

  /** Feed a chunk of 16 kHz samples to the live path; results arrive via onStream. */
  streamChunk(pcm) {
    const copy = pcm.slice();
    this.worker.postMessage({ type: "stream-chunk", id: ++this._id, pcm: copy }, [copy.buffer]);
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
 * Live 16 kHz frame source for the talk-meter. Taps the mic, resamples to 16 kHz, and emits complete
 * 512-sample frames (a whole number of frames per callback) via onFrames(Float32Array). Honest about a
 * missing mic. Uses a ScriptProcessor tap (universally supported).
 */
export class LiveMic {
  constructor({ onFrames } = {}) {
    this.onFrames = onFrames;
    this.running = false;
    this._ctx = null;
    this._stream = null;
    this._node = null;
    this._src = null;
    this._acc = []; // pending resampled samples not yet forming a full 512-frame batch
    this._accLen = 0;
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
    this._src = this._ctx.createMediaStreamSource(this._stream);
    this._node = this._ctx.createScriptProcessor(4096, 1, 1);
    this._node.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);
      // Resample to 16 kHz (linear).
      let res = input;
      if (sr !== TARGET_RATE) {
        const n = Math.round((input.length * TARGET_RATE) / sr);
        res = new Float32Array(n);
        const ratio = input.length / n;
        for (let i = 0; i < n; i++) {
          const t = i * ratio, i0 = Math.floor(t), f = t - i0;
          res[i] = (input[i0] ?? 0) * (1 - f) + (input[i0 + 1] ?? input[i0] ?? 0) * f;
        }
      }
      this._acc.push(res);
      this._accLen += res.length;
      // Emit whole 512-sample frames.
      if (this._accLen >= FRAME) {
        const merged = new Float32Array(this._accLen);
        let off = 0;
        for (const c of this._acc) {
          merged.set(c, off);
          off += c.length;
        }
        const usable = Math.floor(merged.length / FRAME) * FRAME;
        const batch = merged.subarray(0, usable);
        const rest = merged.subarray(usable);
        this._acc = rest.length ? [rest.slice()] : [];
        this._accLen = rest.length;
        this.onFrames?.(batch.slice());
      }
    };
    this._src.connect(this._node);
    const mute = this._ctx.createGain();
    mute.gain.value = 0;
    this._node.connect(mute);
    mute.connect(this._ctx.destination);
    this.running = true;
  }
  stop() {
    this.running = false;
    try {
      this._node && (this._node.onaudioprocess = null);
    } catch { /* ignore */ }
    try {
      this._src?.disconnect();
      this._node?.disconnect();
    } catch { /* ignore */ }
    try {
      this._stream?.getTracks().forEach((t) => t.stop());
    } catch { /* ignore */ }
    try {
      this._ctx?.close();
    } catch { /* ignore */ }
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function cssVar(name, fallback) {
  return getComputedStyle(document.body).getPropertyValue(name).trim() || fallback;
}

/** Draw a mono waveform into a <canvas>, matching the design system's accent colour. */
export function drawWaveform(canvas, pcm) {
  const accent = cssVar("--accent", "#4b3aff");
  const dpr = self.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600, h = canvas.clientHeight || 70;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!pcm || !pcm.length) return;
  const mid = h / 2, step = Math.max(1, Math.floor(pcm.length / w));
  ctx.strokeStyle = accent;
  ctx.globalAlpha = 0.85;
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

/**
 * Draw the per-frame speech-probability curve — the real signal the VAD outputs. Time on x, probability
 * 0→1 on y, with the decision threshold as a dashed line and speech regions (prob≥threshold) shaded.
 */
export function drawProbCurve(canvas, probs, frameSec, opts = {}) {
  const threshold = opts.threshold ?? 0.5;
  const accent = cssVar("--accent", "#4b3aff");
  const good = cssVar("--good", "#0a7d33");
  const muted = cssVar("--muted", "#777");
  const border = cssVar("--border", "#ccc");
  const dpr = self.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600, h = canvas.clientHeight || 130;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!probs || !probs.length) return;
  const padB = 16;
  const plotH = h - padB;
  const y = (p) => plotH - p * plotH * 0.94 - 2;
  const x = (i) => (i / (probs.length - 1 || 1)) * w;

  // Shade speech regions.
  ctx.fillStyle = good;
  ctx.globalAlpha = 0.14;
  let segStart = -1;
  for (let i = 0; i <= probs.length; i++) {
    const on = i < probs.length && probs[i] >= threshold;
    if (on && segStart < 0) segStart = i;
    if (!on && segStart >= 0) {
      ctx.fillRect(x(segStart), 0, Math.max(1, x(i) - x(segStart)), plotH);
      segStart = -1;
    }
  }
  ctx.globalAlpha = 1;

  // Threshold line.
  ctx.strokeStyle = muted;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(0, y(threshold));
  ctx.lineTo(w, y(threshold));
  ctx.stroke();
  ctx.setLineDash([]);

  // Probability curve.
  ctx.strokeStyle = accent;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  for (let i = 0; i < probs.length; i++) {
    const px = x(i), py = y(probs[i]);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  }
  ctx.stroke();

  // Baseline + axis labels.
  ctx.strokeStyle = border;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, plotH);
  ctx.lineTo(w, plotH);
  ctx.stroke();
  ctx.fillStyle = muted;
  ctx.font = "10px system-ui, sans-serif";
  ctx.fillText("prob 1.0", 2, 10);
  ctx.fillText(`threshold ${threshold.toFixed(2)}`, 2, y(threshold) - 3);
  const dur = probs.length * frameSec;
  ctx.fillText(`0s`, 0, h - 3);
  ctx.fillText(`${dur.toFixed(1)}s`, w - 26, h - 3);
}

/** Draw a segment timeline: a strip where detected speech segments are filled accent, silence is bg. */
export function drawTimeline(canvas, segments, duration) {
  const good = cssVar("--good", "#0a7d33");
  const raised = cssVar("--bg-raised", "#eee");
  const dpr = self.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600, h = canvas.clientHeight || 26;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = raised;
  ctx.fillRect(0, 0, w, h);
  if (!duration) return;
  ctx.fillStyle = good;
  for (const s of segments) {
    const x0 = (s.start / duration) * w;
    const x1 = (s.end / duration) * w;
    ctx.fillRect(x0, 2, Math.max(2, x1 - x0), h - 4);
  }
}

/** Render detected speech segments as an accessible list of start→end times. */
export function renderSegments(container, segments) {
  if (!segments.length) {
    container.innerHTML = `<p class="muted">No speech segments detected in this clip.</p>`;
    return;
  }
  const rows = segments.map((s, i) =>
    `<tr><td>${i + 1}</td><td>${s.start.toFixed(2)}s</td><td>${s.end.toFixed(2)}s</td><td>${
      (s.end - s.start).toFixed(2)
    }s</td></tr>`
  ).join("");
  container.innerHTML =
    `<table class="inside-table"><thead><tr><th>#</th><th>start</th><th>end</th><th>length</th></tr></thead><tbody>${rows}</tbody></table>`;
}

export const VAD_CSS = `
.wave-wrap { margin:.5rem 0; }
.wave { inline-size:100%; block-size:70px; display:block; background:var(--bg-raised);
  border:1px solid var(--border); border-radius:var(--radius); }
.curve { inline-size:100%; block-size:130px; display:block; background:var(--bg-raised);
  border:1px solid var(--border); border-radius:var(--radius); }
.curve:focus-visible { outline:3px solid var(--accent); outline-offset:2px; }
.timeline { inline-size:100%; block-size:26px; display:block; border:1px solid var(--border);
  border-radius:999px; overflow:hidden; margin:.3rem 0; }
.audio-row { display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; margin:.5rem 0; }
.audio-row audio { block-size:34px; max-inline-size:100%; }
.rec-dot { inline-size:.7rem; block-size:.7rem; border-radius:50%; background:var(--bad,#c0392b);
  display:inline-block; margin-inline-end:.4rem; animation:recpulse 1s ease-in-out infinite; }
@keyframes recpulse { 50% { opacity:.25; } }
@media (prefers-reduced-motion: reduce) { .rec-dot { animation:none; } }
.dropzone { border:2px dashed var(--border-strong); border-radius:var(--radius); background:var(--bg-raised);
  padding:1rem; text-align:center; cursor:pointer; transition:border-color .15s, background .15s; }
.dropzone.drag { border-color:var(--accent); background:var(--bg-secondary); }
.dropzone:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.sample-row { display:flex; flex-wrap:wrap; gap:.4rem; margin:.5rem 0; }
.chip[aria-pressed="true"] { outline:2px solid var(--accent); outline-offset:1px; }
.field-row { display:flex; flex-wrap:wrap; gap:.6rem; align-items:center; margin:.6rem 0; }
.param-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,220px),1fr)); gap:.8rem 1.2rem; margin:.6rem 0; }
.param { display:flex; flex-direction:column; gap:.25rem; min-inline-size:0; }
.param label { font-size:.8rem; font-weight:600; }
.param .row { display:flex; align-items:center; gap:.5rem; }
.param input[type=range] { inline-size:100%; min-inline-size:0; }
.param output { font-family:var(--font-mono); font-size:.78rem; color:var(--muted); min-inline-size:3.5ch; text-align:right; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem;
  color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.verdict { font-size:1.1rem; font-weight:700; margin:.4rem 0; }
.verdict.speech { color:var(--good); }
.verdict.silent { color:var(--muted); }
.meter-wrap { margin:.6rem 0; }
.meter { inline-size:100%; block-size:34px; background:var(--bg-raised); border:1px solid var(--border);
  border-radius:999px; overflow:hidden; }
.meter-fill { display:block; block-size:100%; inline-size:0%; background:var(--good);
  transition:inline-size .08s linear, background .12s; border-radius:999px; }
@media (prefers-reduced-motion: reduce) { .meter-fill { transition:none; } }
.inside-table { inline-size:100%; border-collapse:collapse; font-size:.82rem; margin-top:.5rem; }
.inside-table th, .inside-table td { text-align:left; padding:.3rem .5rem; border-bottom:1px solid var(--border);
  font-family:var(--font-mono); }
.inside-table th { color:var(--muted); font-weight:600; }
.fallback { border:1px solid var(--warn); border-radius:var(--radius); background:var(--bg-raised); padding:1rem; }
`;
