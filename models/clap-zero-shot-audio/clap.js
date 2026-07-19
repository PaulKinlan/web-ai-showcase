// Front-end helpers for the CLAP (zero-shot-audio-classification) pages. Keeps each page thin: it owns
// the worker handshake, decodes/records audio into the 48 kHz mono Float32Array CLAP wants, draws the
// waveform + the 64-band log-mel spectrogram CLAP's audio branch reasons over, and renders the ranked
// label bars. All inference lives in worker.js (off the main thread).
//
// CLAP is CLIP for sound: it maps audio and free-text labels into ONE shared 512-d embedding space and
// scores each label by cosine similarity to the clip. The label set is an input — you invent it at
// runtime — so a single model classifies against ANY vocabulary without retraining.

const WORKER_URL = "/web-ai-showcase/models/clap-zero-shot-audio/worker.js";
const TARGET_RATE = 48000; // CLAP's feature extractor samples at 48 kHz (NOT 16 kHz like most ASR).

export class ClapEngine {
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

  /**
   * Score a 48 kHz mono Float32Array against free-text labels.
   * Returns { labels, probs, logits, cosines, audioDims, txtDims, spectrogram, ms, device, durationS }.
   */
  classify(audio, labels, opts = {}) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      // Copy so we can transfer the buffer without detaching the caller's array.
      const buf = audio.slice();
      this.worker.postMessage({ type: "run", id, audio: buf, labels, opts }, [buf.buffer]);
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

/** Decode any browser-supported audio ArrayBuffer to a 48 kHz mono Float32Array (CLAP's rate). */
export async function decodeToMono48k(arrayBuffer) {
  const decoded = await audioCtx().decodeAudioData(arrayBuffer.slice(0));
  const frames = Math.max(1, Math.ceil(decoded.duration * TARGET_RATE));
  const off = new OfflineAudioContext(1, frames, TARGET_RATE);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return { pcm: rendered.getChannelData(0), duration: decoded.duration };
}

export async function urlToMono48k(url) {
  const buf = await (await fetch(url)).arrayBuffer();
  return decodeToMono48k(buf);
}

export async function blobToMono48k(blob) {
  return decodeToMono48k(await blob.arrayBuffer());
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

/** Parse a comma/newline-separated label field into a clean, de-duped list. */
export function parseLabels(text) {
  return [
    ...new Set(
      text.split(/[,\n]/).map((s) => s.trim()).filter(Boolean),
    ),
  ];
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Draw a mono waveform into a <canvas>, matching the design system's accent colour. */
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
 * (low frequencies at the bottom). This is the real 2D image CLAP's audio encoder reasons over.
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
      const y = mels - 1 - m; // flip so low mel bands sit at the bottom
      const idx = (y * frames + f) * 4;
      img.data[idx] = r;
      img.data[idx + 1] = g;
      img.data[idx + 2] = b;
      img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

/**
 * Render ranked label bars into `container`. `items` = [{label, prob, cosine?}]. The top label is
 * highlighted; percentages are the softmax over CLAP's audio↔label cosine similarities.
 */
export function renderBars(container, items, { showCosine = false, max = 12 } = {}) {
  const sorted = [...items].sort((a, b) => b.prob - a.prob).slice(0, max);
  const top = sorted[0]?.label;
  container.replaceChildren(
    ...sorted.map((it) => {
      const row = document.createElement("div");
      row.className = "bar-row" + (it.label === top ? " bar-top" : "");
      const pct = (it.prob * 100).toFixed(1);
      const meta = showCosine && it.cosine != null
        ? `<span class="bar-cos">cos ${it.cosine.toFixed(3)}</span>`
        : "";
      row.innerHTML = `
        <div class="bar-head">
          <span class="bar-label">${escapeHTML(it.label)}</span>
          <span class="bar-val">${meta}${pct}%</span>
        </div>
        <div class="bar-track" role="meter" aria-valuemin="0" aria-valuemax="100"
             aria-valuenow="${pct}" aria-label="${escapeHTML(it.label)}: ${pct} percent">
          <div class="bar-fill" style="inline-size:${pct}%"></div>
        </div>`;
      return row;
    }),
  );
}

export const CLAP_CSS = `
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
.bars { display:flex; flex-direction:column; gap:.55rem; margin-top:.5rem; }
.bar-head { display:flex; justify-content:space-between; gap:.5rem; font-size:.85rem; }
.bar-label { font-family:var(--font-body); }
.bar-val { font-family:var(--font-mono); color:var(--muted); white-space:nowrap; }
.bar-cos { margin-inline-end:.5rem; opacity:.8; }
.bar-track { block-size:.7rem; background:var(--bg-raised); border:1px solid var(--border);
  border-radius:999px; overflow:hidden; margin-top:.15rem; }
.bar-fill { block-size:100%; background:var(--muted); border-radius:999px; transition:inline-size .35s ease; }
.bar-top .bar-fill { background:var(--accent); }
.bar-top .bar-label { font-weight:600; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem;
  color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.field-row { display:flex; flex-wrap:wrap; gap:.6rem; align-items:center; margin:.6rem 0; }
.chip { font:inherit; font-size:.8rem; padding:.25rem .7rem; border-radius:999px;
  border:1px solid var(--border); background:var(--bg-raised); color:var(--color); cursor:pointer; }
.chip[aria-pressed=true] { border-color:var(--accent); background:var(--bg-secondary); font-weight:600; }
.chip:hover { border-color:var(--accent); }
.inside-table { inline-size:100%; border-collapse:collapse; font-size:.82rem; margin-top:.5rem; }
.inside-table th, .inside-table td { text-align:left; padding:.3rem .5rem; border-bottom:1px solid var(--border);
  font-family:var(--font-mono); }
.inside-table th { color:var(--muted); font-weight:600; }
textarea { inline-size:100%; font:inherit; font-size:.9rem; padding:.5rem; border-radius:var(--radius);
  border:1px solid var(--border-strong); background:var(--bg-raised); color:var(--color); }
textarea:focus-visible { outline:2px solid var(--accent); outline-offset:1px; }
`;
