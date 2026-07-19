// Front-end helpers for the Speech Emotion Recognition pages. Keeps each page thin: it owns the worker
// handshake, decodes/records audio into the 16 kHz mono Float32Array the model wants, draws the
// waveform, and renders the emotion confidence bars. All inference lives in worker.js (off the main
// thread).

const WORKER_URL = "/web-ai-showcase/models/speech-emotion-recognition/worker.js";
const TARGET_RATE = 16000;

// Emotion → emoji + a stable accent hue, so the same emotion always looks the same across pages.
export const EMOJI = {
  ANGRY: "😠",
  DISGUST: "🤢",
  FEAR: "😨",
  HAPPY: "😄",
  NEUTRAL: "😐",
  SAD: "😢",
  ANGER: "😠",
  HAPPINESS: "😄",
  SADNESS: "😢",
  CALM: "😌",
  SURPRISED: "😲",
  SURPRISE: "😲",
};
export const HUE = {
  ANGRY: 8,
  DISGUST: 96,
  FEAR: 270,
  HAPPY: 45,
  NEUTRAL: 210,
  SAD: 220,
  ANGER: 8,
  HAPPINESS: 45,
  SADNESS: 220,
  CALM: 160,
  SURPRISED: 300,
  SURPRISE: 300,
};
export function emoji(label) {
  return EMOJI[String(label).toUpperCase()] || "🎭";
}
export function hue(label) {
  const h = HUE[String(label).toUpperCase()];
  return h == null ? 210 : h;
}
export function prettyLabel(label) {
  const s = String(label);
  return s.charAt(0).toUpperCase() + s.slice(1).toLowerCase();
}

export class EmotionEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.onProgress = null;
    this.onWindow = null;
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
    } else if (msg.type === "result" || msg.type === "windows-done") {
      const p = this._pending.get(msg.id);
      if (p) {
        this._pending.delete(msg.id);
        p.resolve(msg);
      }
    } else if (msg.type === "window") {
      this.onWindow?.(msg);
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

  /** Classify a 16 kHz mono Float32Array. Returns { labels, ms, device, durationS }. */
  classify(audio, opts) {
    const id = ++this._id;
    const copy = audio.slice();
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, audio: copy, opts }, [copy.buffer]);
    });
  }

  /** Classify a series of 16 kHz windows (emotion-over-time). Per-window results arrive via onWindow. */
  classifyWindows(windows, opts) {
    const id = ++this._id;
    const copies = windows.map((w) => w.slice());
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage(
        { type: "run-windows", id, windows: copies, opts },
        copies.map((w) => w.buffer),
      );
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

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Draw a mono waveform into a <canvas>, matching the design system's accent colour. */
export function drawWaveform(canvas, pcm) {
  const accent = getComputedStyle(document.body).getPropertyValue("--accent").trim() || "#4b3aff";
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

/** Render emotion labels as accessible confidence bars (emoji + label + score), top first. */
export function renderEmotionBars(container, labels) {
  container.replaceChildren(
    ...labels.map(({ label, score }, i) => {
      const row = document.createElement("div");
      row.className = "emo-row" + (i === 0 ? " top" : "");
      const pct = (score * 100).toFixed(1);
      row.style.setProperty("--emo-hue", String(hue(label)));
      row.innerHTML =
        `<span class="emo-label"><span class="emo-emoji" aria-hidden="true">${
          emoji(label)
        }</span> ${escapeHTML(prettyLabel(label))}</span>` +
        `<span class="emo-track"><span class="emo-fill" style="inline-size:${
          Math.max(2, score * 100)
        }%"></span></span>` +
        `<span class="emo-val">${pct}%</span>`;
      return row;
    }),
  );
}

/** Draw an emotion-over-time strip: one coloured cell per window, hue = top emotion, opacity = confidence. */
export function drawEmotionTimeline(canvas, windowLabels) {
  const dpr = self.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600, h = canvas.clientHeight || 40;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!windowLabels || !windowLabels.length) return;
  const cw = w / windowLabels.length;
  for (let i = 0; i < windowLabels.length; i++) {
    const top = windowLabels[i][0];
    if (!top) continue;
    ctx.fillStyle = `hsl(${hue(top.label)} 70% 55% / ${(0.35 + 0.6 * top.score).toFixed(2)})`;
    ctx.fillRect(i * cw, 0, Math.ceil(cw) + 1, h);
  }
}

export const EMOTION_CSS = `
.wave-wrap { margin:.6rem 0; }
.wave { inline-size:100%; block-size:80px; display:block; background:var(--bg-raised);
  border:1px solid var(--border); border-radius:var(--radius); }
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
.big-emotion { display:flex; align-items:center; gap:.7rem; margin:.5rem 0; }
.big-emotion .face { font-size:2.4rem; line-height:1; }
.big-emotion .name { font-size:1.5rem; font-weight:700; font-family:var(--font-display); }
.big-emotion .conf { font-family:var(--font-mono); font-size:.85rem; color:var(--muted); }
.emo-bars { display:flex; flex-direction:column; gap:.35rem; margin:.6rem 0; }
.emo-row { display:grid; grid-template-columns:minmax(8ch,10rem) 1fr 4.5ch; align-items:center; gap:.5rem; font-size:.85rem; }
.emo-row.top .emo-label { font-weight:700; }
.emo-label { display:flex; align-items:center; gap:.3rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.emo-emoji { font-size:1rem; }
.emo-track { block-size:.85rem; background:var(--bg-secondary); border-radius:999px; overflow:hidden; border:1px solid var(--border); min-inline-size:0; }
.emo-fill { display:block; block-size:100%; border-radius:999px; background:hsl(var(--emo-hue) 70% 52%); }
.emo-val { font-family:var(--font-mono); font-size:.78rem; color:var(--muted); text-align:right; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem; color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.emo-timeline { inline-size:100%; block-size:40px; display:block; border:1px solid var(--border); border-radius:var(--radius); }
.param-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,220px),1fr)); gap:.8rem 1.2rem; margin:.6rem 0; }
.param { display:flex; flex-direction:column; gap:.25rem; min-inline-size:0; }
.inside-table { inline-size:100%; border-collapse:collapse; font-size:.82rem; margin-top:.5rem; }
.inside-table th, .inside-table td { text-align:left; padding:.3rem .5rem; border-bottom:1px solid var(--border); font-family:var(--font-mono); }
.inside-table th { color:var(--muted); font-weight:600; }
.fallback { border:1px solid var(--warn); border-radius:var(--radius); background:var(--bg-raised); padding:1rem; }
`;
