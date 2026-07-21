// Front-end helpers for the Music Genre Classification page. Keeps the page thin: owns the worker
// handshake, decodes/records audio into the 16 kHz mono Float32Array the model wants, draws the waveform,
// and renders the 10-genre confidence bars. All inference lives in worker.js (off the main thread).

const WORKER_URL = "/web-ai-showcase/models/music-genre-classification/worker.js";
const TARGET_RATE = 16000;

// The 10 GTZAN genres → an emoji + a stable accent hue, so a genre always looks the same.
export const EMOJI = {
  blues: "🎸",
  classical: "🎻",
  country: "🤠",
  disco: "🕺",
  hiphop: "🎤",
  jazz: "🎷",
  metal: "🤘",
  pop: "🎧",
  reggae: "🌴",
  rock: "🎹",
};
export const HUE = {
  blues: 220,
  classical: 275,
  country: 35,
  disco: 320,
  hiphop: 12,
  jazz: 45,
  metal: 0,
  pop: 300,
  reggae: 130,
  rock: 190,
};
export const emoji = (label) => EMOJI[String(label).toLowerCase()] || "🎵";
export const hue = (label) => {
  const h = HUE[String(label).toLowerCase()];
  return h == null ? 210 : h;
};
export const prettyLabel = (label) => {
  const s = String(label).toLowerCase();
  return s === "hiphop" ? "Hip-hop" : s.charAt(0).toUpperCase() + s.slice(1);
};

export class GenreEngine {
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
        for (const w of this._loadWaiters) w.reject(new Error(msg.message));
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
  /** Classify a 16 kHz mono Float32Array → { labels, ms, device, durationS }. */
  classify(audio) {
    const id = ++this._id;
    const copy = audio.slice();
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, audio: copy }, [copy.buffer]);
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

/** Render genre labels as accessible confidence bars (emoji + label + score), top first. */
export function renderGenreBars(container, labels) {
  container.replaceChildren(
    ...labels.map(({ label, score }, i) => {
      const row = document.createElement("div");
      row.className = "gen-row" + (i === 0 ? " top" : "");
      const pct = (score * 100).toFixed(1);
      row.style.setProperty("--gen-hue", String(hue(label)));
      const name = document.createElement("span");
      name.className = "gen-label";
      name.innerHTML = `<span class="gen-emoji" aria-hidden="true">${emoji(label)}</span>${
        escapeHTML(prettyLabel(label))
      }`;
      const track = document.createElement("span");
      track.className = "gen-track";
      const fill = document.createElement("i");
      fill.style.width = Math.max(1.5, score * 100) + "%";
      track.append(fill);
      const val = document.createElement("span");
      val.className = "gen-pct";
      val.textContent = pct + "%";
      row.append(name, track, val);
      return row;
    }),
  );
}

export const GENRE_CSS = `
.gen-drop { border: 2px dashed var(--border); border-radius: 12px; padding: 1.1rem; text-align: center;
  background: var(--bg-raised); transition: border-color .15s, background .15s; }
.gen-drop.drag { border-color: var(--accent); background: color-mix(in srgb, var(--accent) 10%, transparent); }
.gen-tools { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; justify-content: center; margin: .3rem 0; }
.gen-btn { font: inherit; font-size: .85rem; padding: .35rem .8rem; border-radius: 8px; border: 1px solid var(--border);
  background: var(--bg-raised); color: var(--color); cursor: pointer; }
.gen-btn:hover, .gen-btn:focus-visible { border-color: var(--accent); }
.gen-btn[disabled] { opacity: .5; cursor: default; }
.gen-samples { display: flex; flex-wrap: wrap; gap: .4rem; justify-content: center; margin-top: .4rem; }
.gen-chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.gen-chip:hover, .gen-chip:focus-visible { border-color: var(--accent); }
#wave { width: 100%; height: 80px; display: block; margin: .6rem 0 .2rem; }
.gen-verdict { font-size: 1.4rem; font-weight: 700; margin: .3rem 0 .6rem; min-height: 1.7rem; }
.gen-verdict .gen-emoji { font-size: 1.6rem; margin-inline-end: .3rem; }
.gen-bars { display: flex; flex-direction: column; gap: .3rem; max-width: 32rem; }
.gen-row { display: grid; grid-template-columns: 8.5rem 1fr 3rem; align-items: center; gap: .5rem; font-size: .85rem; }
.gen-row.top .gen-label { font-weight: 700; }
.gen-label { display: inline-flex; align-items: center; gap: .35rem; white-space: nowrap; }
.gen-emoji { font-size: 1rem; }
.gen-track { height: 9px; border-radius: 5px; background: color-mix(in srgb, var(--color) 12%, transparent); overflow: hidden; }
.gen-track > i { display: block; height: 100%; border-radius: 5px;
  background: hsl(var(--gen-hue) 70% 55%); }
.gen-pct { font-family: var(--font-mono, monospace); text-align: right; font-size: .78rem; color: var(--muted); }
.gen-row.top .gen-pct { color: var(--color); }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono, monospace);
  font-size: .78rem; color: var(--muted); margin-top: .7rem; }
.readout b { color: var(--color); font-weight: 600; }
.gen-status { font-family: var(--font-mono, monospace); font-size: .9rem; margin: .4rem 0; }
`;
