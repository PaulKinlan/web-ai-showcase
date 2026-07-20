// Front-end helpers shared by every Speaker-Verification page. Keeps pages thin: owns the worker
// handshake (transferring the audio buffer so nothing is copied), decodes files / samples / mic clips
// into the 16 kHz mono Float32Array the model wants, computes cosine similarity between two
// L2-normalized speaker embeddings, formats the same/different SIGNAL, and renders the waveform +
// embedding + gauge visualizations. ALL inference lives in worker.js (off the main thread). Privacy by
// construction: audio and embeddings never leave the device. This is voice SIMILARITY, not identity,
// lookup, or surveillance.

const WORKER_URL = "/web-ai-showcase/models/speaker-verification/worker.js";
const TARGET_RATE = 16000;

// Cosine-similarity decision band for WavLM-base-plus-sv (L2-normalized x-vectors). These are SIGNAL
// thresholds for a demo, not a security policy: the WavLM-SV model card uses ~0.86 as the same-speaker
// cut. Same-speaker pairs typically land ≥0.86, clearly-different pairs well below — with a
// genuinely-uncertain band between, and some voices that fool it (see the Wild demo). A real system
// would calibrate its own threshold on its own data.
export const SAME_THRESHOLD = 0.86;
export const MAYBE_THRESHOLD = 0.80;

export class SpeakerEngine {
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
        msg.embedding = new Float32Array(msg.embedding); // rehydrate transferred buffer
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

  /** Embed a 16 kHz mono Float32Array → { embedding:Float32Array(512), dims, ms, device, durationS }. */
  embed(audio, opts) {
    const id = ++this._id;
    const copy = audio.slice();
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, audio: copy, opts }, [copy.buffer]);
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

/** Cosine similarity of two L2-normalized embeddings == their dot product. */
export function cosine(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Map a cosine similarity to a human, honest SIGNAL (never a hard identity verdict). */
export function verdict(cos, threshold = SAME_THRESHOLD, maybe = MAYBE_THRESHOLD) {
  if (cos >= threshold) return { key: "same", label: "Likely the same speaker", tone: "good" };
  if (cos >= maybe) return { key: "maybe", label: "Uncertain — borderline", tone: "warn" };
  return { key: "different", label: "Likely different speakers", tone: "bad" };
}

/** Draw a mono waveform into a <canvas>, matching the design system's accent colour. */
export function drawWaveform(canvas, pcm) {
  const accent = getComputedStyle(document.body).getPropertyValue("--accent").trim() || "#4b3aff";
  const dpr = self.devicePixelRatio || 1;
  const w = canvas.clientWidth || 300, h = canvas.clientHeight || 60;
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

/** Render a compact embedding as signed bars (up = positive dim, down = negative). */
export function renderEmbeddingBars(container, embedding, maxBars = 128) {
  const emb = embedding.length > maxBars ? embedding.subarray(0, maxBars) : embedding;
  const max = Math.max(0.001, ...Array.from(emb, Math.abs));
  container.replaceChildren(
    ...Array.from(emb, (v) => {
      const cell = document.createElement("span");
      cell.className = "emb-cell";
      const bar = document.createElement("span");
      bar.className = "emb-bar " + (v >= 0 ? "pos" : "neg");
      bar.style.blockSize = (Math.abs(v) / max * 100).toFixed(1) + "%";
      cell.append(bar);
      return cell;
    }),
  );
}

/** A similarity gauge: place a needle on a −0.2 … 1.0 scale with the decision band marked. */
export function renderGauge(el, cos, threshold = SAME_THRESHOLD) {
  const lo = -0.2, hi = 1.0;
  const pct = Math.max(0, Math.min(1, (cos - lo) / (hi - lo))) * 100;
  const tpct = Math.max(0, Math.min(1, (threshold - lo) / (hi - lo))) * 100;
  el.style.setProperty("--needle", pct.toFixed(1) + "%");
  el.style.setProperty("--thresh", tpct.toFixed(1) + "%");
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Shared inline styles for the speaker-verification widgets. Injected once per page. */
export const SPEAKER_CSS = `
.dropzone { border: 2px dashed var(--border-strong); border-radius: var(--radius);
  background: var(--bg-raised); padding: .7rem; text-align: center; cursor: pointer;
  transition: border-color .15s, background .15s; font-size: .82rem; }
.dropzone.drag { border-color: var(--accent); background: var(--bg-secondary); }
.dropzone:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.pair-grid { display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start; margin: .6rem 0; }
.voice-col { flex: 1 1 260px; min-inline-size: 0; }
.voice-col h4 { font-family: var(--font-body); font-size: .8rem; color: var(--muted); margin: 0 0 .3rem; text-transform: uppercase; letter-spacing: .05em; }
.wave { inline-size: 100%; block-size: 60px; display: block; background: var(--bg-raised);
  border: 1px solid var(--border); border-radius: var(--radius); }
.audio-row { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; margin: .4rem 0; }
.audio-row audio { block-size: 34px; max-inline-size: 100%; }
.sample-row { display: flex; gap: .4rem; flex-wrap: wrap; margin: .5rem 0; }
.chip { font-size: .8rem; padding: .3rem .6rem; border-radius: 999px; border: 1px solid var(--border);
  background: var(--bg-raised); cursor: pointer; min-block-size: 34px; }
.chip[aria-pressed="true"] { outline: 2px solid var(--accent); outline-offset: 1px; border-color: var(--accent); }
.chip:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.rec-dot { inline-size: .7rem; block-size: .7rem; border-radius: 50%; background: var(--bad,#c0392b);
  display: inline-block; margin-inline-end: .4rem; animation: recpulse 1s ease-in-out infinite; }
@keyframes recpulse { 50% { opacity: .25; } }
@media (prefers-reduced-motion: reduce) { .rec-dot { animation: none; } }
.controls-row { display: flex; flex-wrap: wrap; gap: .6rem; align-items: center; margin: .6rem 0; }
.big-verdict { font-size: 1.35rem; font-weight: 600; margin: .3rem 0; display: flex; gap: .5rem; align-items: baseline; flex-wrap: wrap; }
.sim-num { font-family: var(--font-mono); font-size: 1.6rem; font-weight: 700; }
.tone-good { color: var(--good); } .tone-warn { color: var(--warn); } .tone-bad { color: var(--bad); }
.gauge { position: relative; block-size: 1rem; border-radius: 999px; margin: .8rem 0 .3rem;
  background: linear-gradient(to right, var(--bad) 0%, var(--warn) 55%, var(--good) 80%); border: 1px solid var(--border); }
.gauge::before { content: ""; position: absolute; inset-block: -4px; inline-size: 3px; left: var(--thresh, 80%);
  background: var(--color); border-radius: 2px; opacity: .8; }
.gauge::after { content: ""; position: absolute; inset-block: -5px; inline-size: 4px; left: var(--needle, 0%);
  transform: translateX(-50%); background: var(--accent); border-radius: 2px; box-shadow: 0 0 0 2px var(--background); }
.gauge-scale { display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: .68rem; color: var(--muted); }
.emb-strip { display: flex; align-items: center; gap: 1px; block-size: 60px; margin: .3rem 0; padding: 0 1px; }
.emb-cell { flex: 1 1 0; block-size: 100%; display: flex; flex-direction: column; justify-content: center; }
.emb-bar { display: block; inline-size: 100%; min-block-size: 1px; border-radius: 1px; }
.emb-bar.pos { background: var(--accent); align-self: flex-end; }
.emb-bar.neg { background: var(--muted); align-self: flex-start; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.inside-table { inline-size: 100%; border-collapse: collapse; font-size: .82rem; margin-top: .5rem; }
.inside-table th, .inside-table td { text-align: left; padding: .3rem .5rem; border-bottom: 1px solid var(--border); font-family: var(--font-mono); }
.inside-table th { color: var(--muted); font-weight: 600; }
.privacy-note { font-size: .78rem; color: var(--muted); margin: .35rem 0 0; }
.fallback { border: 1px solid var(--warn); border-radius: var(--radius); background: var(--bg-raised); padding: .8rem; margin: .5rem 0; font-size: .85rem; }
.param-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 240px), 1fr)); gap: .8rem 1.2rem; margin: .6rem 0; }
.param { display: flex; flex-direction: column; gap: .25rem; min-inline-size: 0; }
.param label { font-size: .8rem; color: var(--muted); }
.enroll-list { list-style: none; padding: 0; margin: .5rem 0; display: flex; flex-direction: column; gap: .3rem; }
.enroll-list li { display: flex; gap: .5rem; align-items: center; font-size: .82rem; padding: .25rem .4rem; border: 1px solid var(--border); border-radius: 6px; }
`;
