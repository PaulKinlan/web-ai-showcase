// Front-end helpers for the CREPE pitch-detection pages. Keeps each page thin: it owns the worker
// handshake, decodes/records audio into the 16 kHz mono Float32Array CREPE wants, streams live mic
// windows, converts f0→musical notes, and draws the pitch curve + the raw 360-bin salience pitchgram.
// ALL inference lives in worker.js (off the main thread, onnxruntime-web).

const WORKER_URL = "/web-ai-showcase/models/pitch-detection/worker.js";
const TARGET_RATE = 16000;

export class PitchEngine {
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

  /** Analyse a whole 16 kHz mono clip. Resolves { f0, conf, times, activations, nFrames, nBins, hop, frameSec, durationS, ms, device }. */
  run(pcm, opts) {
    const id = ++this._id;
    const copy = pcm.slice();
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, pcm: copy, opts }, [copy.buffer]);
    });
  }

  /** Feed the most-recent window of live mic samples; the latest f0/conf arrives via onStream. */
  streamWindow(pcm) {
    const copy = pcm.slice();
    this.worker.postMessage({ type: "stream-window", id: ++this._id, pcm: copy }, [copy.buffer]);
  }
}

// ── Audio decode / capture (16 kHz mono, exactly what CREPE consumes) ─────────
let _audioCtx = null;
function audioCtx() {
  if (!_audioCtx) {
    const AC = self.AudioContext || self.webkitAudioContext;
    _audioCtx = new AC();
  }
  return _audioCtx;
}

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
  return decodeToMono16k(await (await fetch(url)).arrayBuffer());
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
 * Live 16 kHz window source for the tuner. Taps the mic, resamples to 16 kHz, keeps a rolling buffer,
 * and calls onWindow(Float32Array) with the most-recent `windowSize` samples at each audio callback.
 * Honest about a missing mic. Deterministic cleanup of tracks/context on stop().
 */
export class LiveMic {
  constructor({ onWindow, windowSize = 2048 } = {}) {
    this.onWindow = onWindow;
    this.windowSize = windowSize;
    this.running = false;
    this._ring = new Float32Array(windowSize);
    this._filled = 0;
    this._ctx = null;
    this._stream = null;
    this._node = null;
    this._src = null;
  }
  static supported() {
    return !!navigator.mediaDevices?.getUserMedia;
  }
  async start() {
    this._stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const AC = self.AudioContext || self.webkitAudioContext;
    this._ctx = new AC();
    const sr = this._ctx.sampleRate;
    this._src = this._ctx.createMediaStreamSource(this._stream);
    this._node = this._ctx.createScriptProcessor(2048, 1, 1);
    this._node.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);
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
      // Push into the rolling ring (shift-left then append).
      const w = this.windowSize;
      if (res.length >= w) {
        this._ring.set(res.subarray(res.length - w));
      } else {
        this._ring.copyWithin(0, res.length);
        this._ring.set(res, w - res.length);
      }
      this._filled = Math.min(w, this._filled + res.length);
      if (this._filled >= w) this.onWindow?.(this._ring.slice());
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

// ── Musical-note maths ───────────────────────────────────────────────────────
const NOTE_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];

/** f0 (Hz) → { name, octave, label, midi, nearestMidi, cents } (cents = deviation from equal temperament). */
export function freqToNote(f0) {
  if (!f0 || f0 <= 0) {
    return { name: "–", octave: 0, label: "–", midi: 0, nearestMidi: 0, cents: 0 };
  }
  const midi = 69 + 12 * Math.log2(f0 / 440);
  const nearest = Math.round(midi);
  const cents = Math.round((midi - nearest) * 100);
  const name = NOTE_NAMES[((nearest % 12) + 12) % 12];
  const octave = Math.floor(nearest / 12) - 1;
  return { name, octave, label: `${name}${octave}`, midi, nearestMidi: nearest, cents };
}

/** Hz for a MIDI note number (for gridlines). */
export function midiToFreq(m) {
  return 440 * Math.pow(2, (m - 69) / 12);
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function cssVar(name, fallback) {
  return getComputedStyle(document.body).getPropertyValue(name).trim() || fallback;
}

// ── Visualizations ───────────────────────────────────────────────────────────
export function drawWaveform(canvas, pcm) {
  const accent = cssVar("--accent", "#4b3aff");
  const dpr = self.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600, h = canvas.clientHeight || 60;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!pcm || !pcm.length) return;
  const mid = h / 2, step = Math.max(1, Math.floor(pcm.length / w));
  ctx.strokeStyle = accent;
  ctx.globalAlpha = 0.8;
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
 * Draw the extracted f0 curve — log-frequency y-axis with note gridlines, time on x. Voiced frames
 * (conf ≥ threshold) draw as an accent line whose opacity tracks confidence; unvoiced frames are
 * left as gaps. This is the model's real per-frame output.
 */
export function drawPitchCurve(canvas, res, opts = {}) {
  const threshold = opts.confThreshold ?? 0.5;
  const fMin = opts.fMin ?? 65; // ~C2
  const fMax = opts.fMax ?? 1050; // ~C6
  const accent = cssVar("--accent", "#4b3aff");
  const good = cssVar("--good", "#0a7d33");
  const muted = cssVar("--muted", "#777");
  const border = cssVar("--border", "#ccc");
  const dpr = self.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600, h = canvas.clientHeight || 200;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const padL = 34, padB = 15;
  const plotW = w - padL, plotH = h - padB;
  const lminf = Math.log2(fMin), lmaxf = Math.log2(fMax);
  const y = (f) => plotH - ((Math.log2(f) - lminf) / (lmaxf - lminf)) * (plotH - 4) - 2;
  const x = (i, n) => padL + (i / (n - 1 || 1)) * plotW;

  // Note gridlines: every C, plus the A above it, labelled.
  ctx.font = "10px system-ui, sans-serif";
  for (let m = 24; m <= 96; m++) {
    const f = midiToFreq(m);
    if (f < fMin || f > fMax) continue;
    const isC = m % 12 === 0, isA = m % 12 === 9;
    if (!isC && !isA) continue;
    const yy = y(f);
    ctx.strokeStyle = border;
    ctx.globalAlpha = isC ? 0.55 : 0.25;
    ctx.beginPath();
    ctx.moveTo(padL, yy);
    ctx.lineTo(w, yy);
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = muted;
    ctx.fillText(`${isC ? "C" : "A"}${Math.floor(m / 12) - 1}`, 2, yy + 3);
  }

  if (!res || !res.f0 || !res.f0.length) {
    ctx.fillStyle = muted;
    ctx.fillText("Analyse a clip to see its pitch curve.", padL + 8, plotH / 2);
    return;
  }
  const n = res.f0.length;
  // Voiced segments as connected accent strokes; opacity ~ confidence.
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  let drawing = false;
  for (let i = 0; i < n; i++) {
    const voiced = res.conf[i] >= threshold && res.f0[i] >= fMin && res.f0[i] <= fMax;
    if (voiced) {
      const px = x(i, n), py = y(res.f0[i]);
      ctx.strokeStyle = accent;
      ctx.globalAlpha = 0.35 + 0.65 * Math.min(1, res.conf[i]);
      if (!drawing) {
        ctx.beginPath();
        ctx.moveTo(px, py);
        drawing = true;
      } else {
        ctx.lineTo(px, py);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(px, py);
      }
    } else {
      drawing = false;
    }
  }
  ctx.globalAlpha = 1;

  // Axis baseline + time labels.
  ctx.strokeStyle = border;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, plotH);
  ctx.lineTo(w, plotH);
  ctx.stroke();
  ctx.fillStyle = muted;
  ctx.fillText("0s", padL, h - 3);
  ctx.fillText(`${res.durationS.toFixed(1)}s`, w - 26, h - 3);
  void good;
}

/**
 * Draw the raw 360-bin CREPE salience "pitchgram" — a heatmap of the model's own activation
 * (x = time frame, y = pitch bin, brightness = activation) with the extracted f0 curve overlaid.
 * This is the see-inside surface: it shows the activation the f0 is read from, and the confidence
 * (peak brightness) at every frame.
 */
export function drawPitchgram(canvas, res, opts = {}) {
  const accent = cssVar("--accent", "#4b3aff");
  const muted = cssVar("--muted", "#777");
  const dpr = self.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600, h = canvas.clientHeight || 220;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!res || !res.activations || !res.nFrames) {
    ctx.fillStyle = muted;
    ctx.font = "10px system-ui, sans-serif";
    ctx.fillText("Analyse a clip to see the CREPE salience map.", 8, h / 2);
    return;
  }
  const { activations, nFrames, nBins } = res;
  // Build a native-res image (nFrames wide × nBins tall) then scale it into the canvas.
  const img = new ImageData(nFrames, nBins);
  // accent rgb for tinting the heatmap
  const [ar, ag, ab] = accentRGB(accent);
  for (let f = 0; f < nFrames; f++) {
    for (let b = 0; b < nBins; b++) {
      const raw = Math.max(0, Math.min(1, activations[f * nBins + b]));
      const v = Math.pow(raw, 0.6); // display gamma — lift low activations so the salience is legible
      // y inverted: high pitch (high bin) at the top
      const yb = nBins - 1 - b;
      const o = (yb * nFrames + f) * 4;
      // dark→accent ramp
      img.data[o] = ar * v;
      img.data[o + 1] = ag * v;
      img.data[o + 2] = ab * v;
      img.data[o + 3] = 255;
    }
  }
  const tmp = typeof OffscreenCanvas !== "undefined"
    ? new OffscreenCanvas(nFrames, nBins)
    : document.createElement("canvas");
  tmp.width = nFrames;
  tmp.height = nBins;
  tmp.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(tmp, 0, 0, w, h);

  // Overlay extracted f0 as a bright line (bin index of the argmax band ≈ (f0 cents)).
  ctx.strokeStyle = "#fff";
  ctx.globalAlpha = 0.7;
  ctx.lineWidth = 1.2;
  let drawing = false;
  for (let f = 0; f < nFrames; f++) {
    const conf = res.conf[f];
    if (conf < (opts.confThreshold ?? 0.5)) {
      drawing = false;
      continue;
    }
    const bin = f0ToBin(res.f0[f]);
    if (bin < 0) {
      drawing = false;
      continue;
    }
    const px = (f / (nFrames - 1 || 1)) * w;
    const py = ((nBins - 1 - bin) / (nBins - 1)) * h;
    if (!drawing) {
      ctx.beginPath();
      ctx.moveTo(px, py);
      drawing = true;
    } else ctx.lineTo(px, py);
  }
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function accentRGB(css) {
  // Accept #rrggbb or fall back to indigo.
  const m = /^#?([0-9a-f]{6})$/i.exec(css.trim());
  if (m) {
    const n = parseInt(m[1], 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  return [75, 58, 255];
}
// f0 (Hz) → CREPE bin index (inverse of cents mapping), or -1 if out of range.
function f0ToBin(f0) {
  if (!f0 || f0 <= 0) return -1;
  const cents = 1200 * Math.log2(f0 / 10);
  const bin = Math.round(((cents - 1997.3794084376191) * 359) / 7180);
  return bin >= 0 && bin < 360 ? bin : -1;
}

/** A concise readable summary of a voiced-note histogram: which notes dominated the clip. */
export function summariseNotes(res, confThreshold = 0.5) {
  const counts = new Map();
  let voiced = 0;
  for (let i = 0; i < res.f0.length; i++) {
    if (res.conf[i] < confThreshold) continue;
    voiced++;
    const { label } = freqToNote(res.f0[i]);
    counts.set(label, (counts.get(label) || 0) + 1);
  }
  const rows = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  return { rows, voiced, total: res.f0.length };
}

/** Render detected notes as an accessible table (the text alternative to the pitch curve). */
export function renderNoteTable(container, res, confThreshold = 0.5) {
  const { rows, voiced, total } = summariseNotes(res, confThreshold);
  if (!voiced) {
    container.innerHTML =
      `<p class="muted">No voiced pitch detected above the confidence threshold in this clip.</p>`;
    return;
  }
  const body = rows.slice(0, 12).map(([label, c]) =>
    `<tr><td>${label}</td><td>${((c / voiced) * 100).toFixed(0)}%</td><td>${c}</td></tr>`
  ).join("");
  container.innerHTML =
    `<table class="inside-table"><caption class="muted" style="text-align:left;padding:.2rem .5rem">Notes in the clip (${voiced}/${total} frames voiced)</caption>` +
    `<thead><tr><th scope="col">note</th><th scope="col">share</th><th scope="col">frames</th></tr></thead><tbody>${body}</tbody></table>`;
}

export const PITCH_CSS = `
.wave-wrap { margin:.5rem 0; }
.wave { inline-size:100%; block-size:60px; display:block; background:var(--bg-raised);
  border:1px solid var(--border); border-radius:var(--radius); }
.pcurve { inline-size:100%; block-size:210px; display:block; background:var(--bg-raised);
  border:1px solid var(--border); border-radius:var(--radius); }
.pcurve:focus-visible { outline:3px solid var(--accent); outline-offset:2px; }
.pgram { inline-size:100%; block-size:220px; display:block; background:#0b0b17;
  border:1px solid var(--border); border-radius:var(--radius); }
.pgram:focus-visible { outline:3px solid var(--accent); outline-offset:2px; }
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
.notebig { font-size:2.6rem; font-weight:700; line-height:1.1; font-family:var(--font-mono); }
.notebig .oct { font-size:1.4rem; color:var(--muted); }
.cents-meter { inline-size:100%; block-size:34px; position:relative; background:var(--bg-raised);
  border:1px solid var(--border); border-radius:999px; overflow:hidden; margin:.4rem 0; }
.cents-center { position:absolute; inset-block:0; inset-inline-start:50%; inline-size:2px; background:var(--muted); }
.cents-needle { position:absolute; inset-block:2px; inline-size:6px; border-radius:3px; background:var(--accent);
  inset-inline-start:50%; transform:translateX(-50%); transition:inset-inline-start .08s linear, background .12s; }
@media (prefers-reduced-motion: reduce) { .cents-needle { transition:none; } }
.cents-needle.intune { background:var(--good); }
.inside-table { inline-size:100%; border-collapse:collapse; font-size:.82rem; margin-top:.5rem; }
.inside-table th, .inside-table td { text-align:left; padding:.3rem .5rem; border-bottom:1px solid var(--border);
  font-family:var(--font-mono); }
.inside-table th { color:var(--muted); font-weight:600; }
.fallback { border:1px solid var(--warn); border-radius:var(--radius); background:var(--bg-raised); padding:1rem; }
`;
