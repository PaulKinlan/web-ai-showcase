// Front-end helpers for the CLAP (zero-shot audio) pages. Keeps each page thin: worker handshake,
// decoding/recording audio into the 48 kHz mono Float32Array CLAP wants, an accessible label-chip
// editor, and the visualizations (waveform, 64-band log-mel heatmap, 512-d embedding strips, score
// bars). All inference lives in worker.js, off the main thread.

const WORKER_URL = "/web-ai-showcase/models/clap-zero-shot-audio/worker.js";
export const TARGET_RATE = 48000; // ClapFeatureExtractor sampling_rate

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

  _call(msg) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...msg, id });
    });
  }

  /** Zero-shot classify a 48 kHz mono Float32Array against free-text labels.
   *  → { results:[{label,sentence,score,logit,cosine}], audioEmbed, textEmbeds, spectrogram, ms, device, durationS } */
  classify(audio, labels, opts = {}) {
    return this._call({ type: "classify", audio, labels, template: opts.template, wantSpectrogram: opts.spectrogram !== false });
  }

  /** → { audioEmbed: Float32Array(512), ms } */
  embedAudio(audio) {
    return this._call({ type: "embed-audio", audio });
  }

  /** → { textEmbeds: Float32Array(512)[], ms } */
  embedTexts(texts) {
    return this._call({ type: "embed-texts", texts });
  }
}

export function cosine(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s; // CLAP embeddings arrive L2-normalised, so the dot product IS the cosine
}

let _audioCtx = null;
function audioCtx() {
  if (!_audioCtx) {
    const AC = self.AudioContext || self.webkitAudioContext;
    _audioCtx = new AC();
  }
  return _audioCtx;
}

/** Decode any browser-supported audio ArrayBuffer to a 48 kHz mono Float32Array. */
export async function decodeToMono48k(arrayBuffer) {
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

/**
 * Live microphone listener: keeps a rolling window of the last `windowS` seconds and calls
 * `onWindow(Float32Array @ 48 kHz)` every `hopS` seconds. Honest about a missing mic.
 */
export class LiveListener {
  constructor({ windowS = 3, hopS = 1.5, onWindow } = {}) {
    this.windowS = windowS;
    this.hopS = hopS;
    this.onWindow = onWindow;
    this.running = false;
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
    this._sinceHop = 0;
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
    const mute = this._ctx.createGain();
    mute.gain.value = 0;
    this._node.connect(mute);
    mute.connect(this._ctx.destination);
    this.running = true;
  }
  _emit() {
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
    try { this._node && (this._node.onaudioprocess = null); } catch { /* already gone */ }
    try {
      this._src?.disconnect();
      this._node?.disconnect();
    } catch { /* already gone */ }
    try { this._stream?.getTracks().forEach((t) => t.stop()); } catch { /* already gone */ }
    try { this._ctx?.close(); } catch { /* already gone */ }
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/**
 * Accessible free-text label editor: removable chips + an input to add more. Calls onChange(labels).
 * Everything is keyboard-operable (Enter adds; each chip's button removes; Backspace in the empty
 * input removes the last label).
 */
export function createLabelEditor(mount, initial, onChange) {
  let labels = [...initial];
  mount.classList.add("label-editor");
  mount.innerHTML = `
    <ul class="label-chips" role="list" aria-label="Candidate labels"></ul>
    <div class="label-add">
      <input type="text" aria-label="Add a label" placeholder="type a label — anything — and press Enter" />
      <button type="button" class="secondary">Add</button>
    </div>`;
  const list = mount.querySelector(".label-chips");
  const input = mount.querySelector("input");
  const addBtn = mount.querySelector("button");

  function render() {
    list.replaceChildren(...labels.map((label, i) => {
      const li = document.createElement("li");
      li.className = "label-chip";
      const span = document.createElement("span");
      span.textContent = label;
      const rm = document.createElement("button");
      rm.type = "button";
      rm.setAttribute("aria-label", `Remove label: ${label}`);
      rm.textContent = "×";
      rm.addEventListener("click", () => {
        labels.splice(i, 1);
        render();
        onChange?.(labels);
      });
      li.append(span, rm);
      return li;
    }));
  }
  function add() {
    const v = input.value.trim();
    if (!v || labels.includes(v)) return;
    labels.push(v);
    input.value = "";
    render();
    onChange?.(labels);
  }
  addBtn.addEventListener("click", add);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      add();
    } else if (e.key === "Backspace" && !input.value && labels.length) {
      labels.pop();
      render();
      onChange?.(labels);
    }
  });
  render();
  return {
    get labels() {
      return [...labels];
    },
    set(next) {
      labels = [...next];
      render();
      onChange?.(labels);
    },
  };
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
  [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
  [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 253, 191],
];
function ramp(t) {
  t = Math.max(0, Math.min(1, t));
  const x = t * (MAGMA.length - 1);
  const i = Math.floor(x), f = x - i;
  const a = MAGMA[i], b = MAGMA[Math.min(i + 1, MAGMA.length - 1)];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/** Draw a log-mel spectrogram ({data,frames,mels,min,max}): time on x, low frequencies at the bottom. */
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

/** Draw a 512-d embedding as a 64×8 diverging heat strip (blue negative, warm positive). */
export function drawEmbeddingStrip(canvas, embed) {
  const cols = 64, rows = Math.ceil(embed.length / cols);
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(cols, rows);
  let peak = 1e-6;
  for (const v of embed) peak = Math.max(peak, Math.abs(v));
  for (let i = 0; i < embed.length; i++) {
    const t = embed[i] / peak; // −1 … 1
    const x = i % cols, y = (i / cols) | 0;
    const idx = (y * cols + x) * 4;
    if (t >= 0) {
      img.data[idx] = 245;
      img.data[idx + 1] = 125 + 100 * (1 - t);
      img.data[idx + 2] = 21 + 170 * (1 - t);
    } else {
      img.data[idx] = 60 + 150 * (1 + t);
      img.data[idx + 1] = 90 + 130 * (1 + t);
      img.data[idx + 2] = 235;
    }
    img.data[idx + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

/** Render labels as accessible score bars into `container`. */
export function renderScoreBars(container, results, max = 8) {
  container.replaceChildren(
    ...results.slice(0, max).map(({ label, score }, i) => {
      const row = document.createElement("div");
      row.className = "score-row" + (i === 0 ? " top" : "");
      const pct = (score * 100).toFixed(1);
      row.innerHTML =
        `<span class="score-label">${escapeHTML(label)}</span>` +
        `<span class="score-track"><span class="score-fill" style="inline-size:${
          Math.max(2, score * 100)
        }%"></span></span>` +
        `<span class="score-val">${pct}%</span>`;
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
.scores { display:flex; flex-direction:column; gap:.35rem; margin:.5rem 0; }
.score-row { display:grid; grid-template-columns:minmax(8ch,14rem) 1fr 4.5ch; align-items:center; gap:.5rem;
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
.inside-table { inline-size:100%; border-collapse:collapse; font-size:.82rem; margin-top:.5rem; }
.inside-table th, .inside-table td { text-align:left; padding:.3rem .5rem; border-bottom:1px solid var(--border);
  font-family:var(--font-mono); }
.inside-table th { color:var(--muted); font-weight:600; }
.label-editor { margin:.5rem 0; }
.label-chips { list-style:none; display:flex; flex-wrap:wrap; gap:.4rem; padding:0; margin:0 0 .5rem; }
.label-chip { display:inline-flex; align-items:center; gap:.35rem; background:var(--bg-secondary);
  border:1px solid var(--border); border-radius:999px; padding:.2rem .35rem .2rem .7rem; font-size:.85rem; }
.label-chip button { background:none; border:none; color:var(--muted); cursor:pointer; font-size:1rem;
  line-height:1; padding:.15rem .35rem; border-radius:50%; }
.label-chip button:hover, .label-chip button:focus-visible { color:var(--bad,#c0392b);
  background:var(--bg-raised); outline:2px solid var(--accent); }
.label-add { display:flex; gap:.4rem; }
.label-add input { flex:1; min-inline-size:12ch; }
.embed-row { display:grid; grid-template-columns:minmax(8ch,14rem) 1fr 6ch; align-items:center; gap:.5rem;
  margin:.25rem 0; font-size:.85rem; }
.embed-row .embed-label { text-align:right; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.embed-row.top .embed-label { font-weight:700; color:var(--accent); }
.embed-strip { inline-size:100%; block-size:26px; image-rendering:pixelated; display:block;
  border:1px solid var(--border); border-radius:4px; }
.embed-cos { font-family:var(--font-mono); font-size:.78rem; color:var(--muted); text-align:right; }
.fallback { border:1px solid var(--warn); border-radius:var(--radius); background:var(--bg-raised); padding:1rem; }
`;
