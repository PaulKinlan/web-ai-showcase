// Front-end helpers for the Whisper-large-v3-turbo pages. Keeps each page thin: owns the worker
// handshake, turns samples / uploads / mic recordings into the 16 kHz mono Float32Array Whisper wants,
// draws the waveform, and renders the transcript, detected language, and timestamped segments. ALL
// inference runs in worker.js (off the main thread).
//
// whisper-large-v3-turbo is the large-v3 encoder with a pruned 4-layer decoder (down from 32): near
// large-v3 accuracy, several times faster to decode. WebGPU (q4f16) is preferred; a WASM q8 fallback
// runs anywhere (slower, but real).

const WORKER_URL = "/web-ai-showcase/models/whisper-large-v3-turbo/worker.js";
const TARGET_RATE = 16000;

// The ~99 languages Whisper knows (code -> English name), for the forced-language picker (sorted).
export const WHISPER_LANGS = [
  ["af", "Afrikaans"],
  ["sq", "Albanian"],
  ["am", "Amharic"],
  ["ar", "Arabic"],
  ["hy", "Armenian"],
  ["as", "Assamese"],
  ["az", "Azerbaijani"],
  ["ba", "Bashkir"],
  ["eu", "Basque"],
  ["be", "Belarusian"],
  ["bn", "Bengali"],
  ["bs", "Bosnian"],
  ["br", "Breton"],
  ["bg", "Bulgarian"],
  ["my", "Burmese"],
  ["yue", "Cantonese"],
  ["ca", "Catalan"],
  ["zh", "Chinese"],
  ["hr", "Croatian"],
  ["cs", "Czech"],
  ["da", "Danish"],
  ["nl", "Dutch"],
  ["en", "English"],
  ["et", "Estonian"],
  ["fo", "Faroese"],
  ["fi", "Finnish"],
  ["fr", "French"],
  ["gl", "Galician"],
  ["ka", "Georgian"],
  ["de", "German"],
  ["el", "Greek"],
  ["gu", "Gujarati"],
  ["ht", "Haitian Creole"],
  ["ha", "Hausa"],
  ["haw", "Hawaiian"],
  ["he", "Hebrew"],
  ["hi", "Hindi"],
  ["hu", "Hungarian"],
  ["is", "Icelandic"],
  ["id", "Indonesian"],
  ["it", "Italian"],
  ["ja", "Japanese"],
  ["jw", "Javanese"],
  ["kn", "Kannada"],
  ["kk", "Kazakh"],
  ["km", "Khmer"],
  ["ko", "Korean"],
  ["lo", "Lao"],
  ["la", "Latin"],
  ["lv", "Latvian"],
  ["ln", "Lingala"],
  ["lt", "Lithuanian"],
  ["lb", "Luxembourgish"],
  ["mk", "Macedonian"],
  ["mg", "Malagasy"],
  ["ms", "Malay"],
  ["ml", "Malayalam"],
  ["mt", "Maltese"],
  ["mi", "Maori"],
  ["mr", "Marathi"],
  ["mn", "Mongolian"],
  ["ne", "Nepali"],
  ["no", "Norwegian"],
  ["nn", "Nynorsk"],
  ["oc", "Occitan"],
  ["ps", "Pashto"],
  ["fa", "Persian"],
  ["pl", "Polish"],
  ["pt", "Portuguese"],
  ["pa", "Punjabi"],
  ["ro", "Romanian"],
  ["ru", "Russian"],
  ["sa", "Sanskrit"],
  ["sr", "Serbian"],
  ["sn", "Shona"],
  ["sd", "Sindhi"],
  ["si", "Sinhala"],
  ["sk", "Slovak"],
  ["sl", "Slovenian"],
  ["so", "Somali"],
  ["es", "Spanish"],
  ["su", "Sundanese"],
  ["sw", "Swahili"],
  ["sv", "Swedish"],
  ["tl", "Tagalog"],
  ["tg", "Tajik"],
  ["ta", "Tamil"],
  ["tt", "Tatar"],
  ["te", "Telugu"],
  ["th", "Thai"],
  ["bo", "Tibetan"],
  ["tr", "Turkish"],
  ["tk", "Turkmen"],
  ["uk", "Ukrainian"],
  ["ur", "Urdu"],
  ["uz", "Uzbek"],
  ["vi", "Vietnamese"],
  ["cy", "Welsh"],
  ["yi", "Yiddish"],
  ["yo", "Yoruba"],
];

export const WLANG_NAME = Object.fromEntries(WHISPER_LANGS);

/** Fill a <select> with an Auto-detect option plus every Whisper language. */
export function fillWhisperLangs(select) {
  const auto = document.createElement("option");
  auto.value = "";
  auto.textContent = "Auto-detect";
  select.replaceChildren(
    auto,
    ...WHISPER_LANGS.map(([c, n]) => {
      const o = document.createElement("option");
      o.value = c;
      o.textContent = `${n} (${c})`;
      return o;
    }),
  );
}

export class TurboEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.dtype = "q8";
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
      this.dtype = msg.dtype;
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

  load(onProgress) {
    if (onProgress) this.onProgress = onProgress;
    if (this.ready) return Promise.resolve({ device: this.device, dtype: this.dtype });
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load" });
    });
  }

  /**
   * Transcribe a 16 kHz mono Float32Array.
   * opts: { language: "" | code, task: "transcribe" | "translate", detect: bool }
   * Returns { text, segments, detectedLang, detectedProb, langProbs, tokens, tokPerSec, ms, device }.
   */
  transcribe(audio, opts = {}) {
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

/** Decode any browser-supported audio ArrayBuffer to a 16 kHz mono Float32Array (what Whisper wants). */
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

/** A tiny mic recorder: start() then stop() -> { blob, url }. Honest about missing mic support. */
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

export function fmtTime(s) {
  if (s == null || Number.isNaN(s)) return "\u2013";
  const m = Math.floor(s / 60);
  const sec = (s - m * 60).toFixed(1).padStart(4, "0");
  return `${m}:${sec}`;
}

/** Draw a mono waveform into a <canvas>, matching light/dark from CSS custom properties. */
export function drawWaveform(canvas, pcm, progress = 0) {
  const cs = getComputedStyle(document.body);
  const accent = cs.getPropertyValue("--accent").trim() || "#4b3aff";
  const muted = cs.getPropertyValue("--muted").trim() || "#888";
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

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Render the detected-language probability bars into a container. */
export function renderLangBars(container, langProbs, nameFor) {
  container.replaceChildren(
    ...(langProbs || []).slice(0, 5).map(([code, p]) => {
      const row = document.createElement("div");
      row.className = "lang-bar";
      const pct = (p * 100).toFixed(1);
      row.innerHTML = `<span class="lb-name">${
        escapeHTML(nameFor(code))
      } <span class="muted">(${code})</span></span>
      <span class="lb-track"><span class="lb-fill" style="inline-size:${pct}%"></span></span>
      <span class="lb-pct">${pct}%</span>`;
      return row;
    }),
  );
}

export const TURBO_CSS = `
.wave-wrap { margin:.6rem 0; }
.wave { inline-size:100%; block-size:96px; display:block; background:var(--bg-raised);
  border:1px solid var(--border); border-radius:var(--radius); }
.audio-row { display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; margin:.5rem 0; }
.audio-row audio { block-size:34px; }
.rec-dot { inline-size:.7rem; block-size:.7rem; border-radius:50%; background:var(--bad);
  display:inline-block; margin-inline-end:.4rem; animation:recpulse 1s ease-in-out infinite; }
@keyframes recpulse { 50% { opacity:.25; } }
@media (prefers-reduced-motion: reduce) { .rec-dot { animation:none; } }
.transcript { font-family:var(--font-body); font-size:1.05rem; line-height:1.8; margin:.4rem 0;
  padding:.8rem; background:var(--bg-raised); border:1px solid var(--border); border-radius:var(--radius); }
.seg-table { inline-size:100%; border-collapse:collapse; font-size:.85rem; margin-top:.5rem; }
.seg-table th, .seg-table td { text-align:left; padding:.3rem .5rem; border-bottom:1px solid var(--border);
  vertical-align:top; }
.seg-table th { color:var(--muted); font-weight:600; font-family:var(--font-mono); }
.seg-table td.t { font-family:var(--font-mono); color:var(--muted); white-space:nowrap; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem;
  color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.sample-row { display:flex; flex-wrap:wrap; gap:.4rem; margin:.4rem 0; }
.chip { font:inherit; font-size:.78rem; padding:.2rem .6rem; border-radius:999px; border:1px solid var(--border);
  background:var(--bg-raised); color:var(--color); cursor:pointer; }
.chip:hover { border-color:var(--accent); }
.opt-row { display:grid; gap:.9rem 1.2rem; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); align-items:end; margin:.6rem 0; }
.opt-row label { display:flex; flex-direction:column; gap:.3rem; font-size:.82rem; }
.opt-row select { inline-size:100%; }
.fallback { border:1px solid var(--warn); border-radius:var(--radius); background:var(--bg-raised); padding:1rem; }
.lang-bars { display:flex; flex-direction:column; gap:.35rem; margin-top:.5rem; }
.lang-bar { display:grid; grid-template-columns:minmax(120px,1fr) 2fr auto; gap:.6rem; align-items:center; font-size:.85rem; }
.lb-track { block-size:.7rem; background:var(--bg-secondary); border-radius:999px; overflow:hidden; border:1px solid var(--border); }
.lb-fill { display:block; block-size:100%; background:var(--accent); }
.lb-pct { font-family:var(--font-mono); font-size:.78rem; color:var(--muted); }
.stat-row { display:grid; gap:.6rem; grid-template-columns:repeat(auto-fit, minmax(120px, 1fr)); margin-top:.6rem; }
.stat { border:1px solid var(--border); border-radius:8px; background:var(--bg-secondary); padding:.5rem .7rem; }
.stat .k { font-family:var(--font-mono); font-size:.66rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
.stat .v { font-family:var(--font-display); font-size:1.3rem; }
`;
