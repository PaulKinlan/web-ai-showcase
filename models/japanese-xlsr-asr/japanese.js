// Front-end helpers for the wav2vec2-XLSR-53 (Japanese) pages. Keeps each page thin: worker
// handshake, audio decode to the 16 kHz mono Float32Array the model wants, a tiny mic recorder, waveform
// draw, the per-frame CTC strip renderer, and shared CSS. All inference lives in worker.js (off the main
// thread).
//
// DISTINCT from the English wav2vec2, Russian XLSR and the French/Spanish/Italian/Portuguese XLSR demos:
// this model is XLSR-53 fine-tuned for JAPANESE — a CHARACTER-level recogniser over a ~2,300-symbol
// character vocab (hiragana + katakana + kanji) — so it is language-specialised for Japanese. The engine
// is the same CTC shape (per-frame
// argmax → collapse), which lets the page visualise the collapse over Japanese characters.

const WORKER_URL = "/web-ai-showcase/models/japanese-xlsr-asr/worker.js";
const TARGET_RATE = 16000;

export class JapaneseXlsrEngine {
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

  /** Transcribe a 16 kHz mono Float32Array. Returns
   *  { text, strip:[{c,blank,boundary}], collapsed:[chars], chars:[{c,start,end}], words:[{text,start,end}],
   *    frameMs, frames, ms, device }. */
  transcribe(audio, audioDur) {
    const id = ++this._id;
    const copy = audio.slice(); // copy so re-runs keep their samples (we transfer the buffer)
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, audio: copy, audioDur }, [copy.buffer]);
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

export function fmtTime(s) {
  if (s == null || Number.isNaN(s)) return "–";
  const m = Math.floor(s / 60);
  const sec = (s - m * 60).toFixed(1).padStart(4, "0");
  return `${m}:${sec}`;
}

/** Draw a mono waveform into a <canvas>, matching the light/dark design system. progress in 0..1. */
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

/**
 * Render the CTC per-frame strip into a container. Each frame becomes a cell: blank frames muted,
 * emitted characters in the accent, a word boundary marked. `strip` = [{ c, blank, boundary }].
 */
export function renderStrip(el, strip) {
  el.replaceChildren(...strip.map((f) => {
    const span = document.createElement("span");
    span.className = "ctc-cell" + (f.blank ? " blank" : "") + (f.boundary ? " bound" : "");
    span.textContent = f.blank ? "·" : (f.boundary ? "␣" : f.c);
    return span;
  }));
}

/** Render per-character forced-alignment chips (Japanese character + start–end seconds). */
export function renderChars(el, chars) {
  el.replaceChildren(...chars.map((c) => {
    const span = document.createElement("span");
    span.className = "align-word";
    span.lang = "ja";
    span.textContent = c.c;
    const t = document.createElement("span");
    t.className = "t";
    t.textContent = `${c.start.toFixed(1)}–${c.end.toFixed(1)}s`;
    span.append(t);
    return span;
  }));
}

/** XLSR widget styles, injected once per page (keeps us on the shared design system). CJK-aware. */
export const JA_XLSR_CSS = `
.wave-wrap { margin:.6rem 0; }
.wave { inline-size:100%; block-size:96px; display:block; background:var(--bg-raised);
  border:1px solid var(--border); border-radius:var(--radius); }
.audio-row { display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; margin:.5rem 0; }
.audio-row audio { block-size:34px; max-inline-size:100%; }
.rec-dot { inline-size:.7rem; block-size:.7rem; border-radius:50%; background:var(--bad,#d33);
  display:inline-block; margin-inline-end:.4rem; animation:recpulse 1s ease-in-out infinite; }
@keyframes recpulse { 50% { opacity:.25; } }
@media (prefers-reduced-motion: reduce) { .rec-dot { animation:none; } }
.transcript { font-family:var(--font-body); font-size:1.35rem; line-height:2; margin:.4rem 0;
  padding:.8rem; background:var(--bg-raised); border:1px solid var(--border); border-radius:var(--radius);
  overflow-wrap:anywhere; word-break:normal; line-break:auto; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem;
  color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.chip { font:inherit; font-size:.82rem; padding:.5rem .8rem; min-block-size:40px; border-radius:999px;
  border:1px solid var(--border); background:var(--bg-raised); color:var(--color); cursor:pointer; }
.chip:hover { border-color:var(--accent); }
.sample-row { display:flex; flex-wrap:wrap; gap:.4rem; margin:.4rem 0; align-items:center; }
.audio-row button { min-block-size:44px; }
.fallback { border:1px solid var(--warn,#c90); border-radius:var(--radius); background:var(--bg-raised); padding:1rem; }
.fallback code { background:var(--bg-secondary); padding:.05rem .3rem; border-radius:4px; }
.ctc-strip { display:flex; flex-wrap:wrap; gap:2px; padding:.6rem; background:var(--bg-raised);
  border:1px solid var(--border); border-radius:var(--radius); font-family:var(--font-mono);
  font-size:.92rem; line-height:1.5; max-block-size:12rem; overflow-y:auto; }
.ctc-cell { min-inline-size:1.15em; text-align:center; padding:.05rem .12rem; border-radius:3px;
  color:var(--accent-ink); background:var(--accent); font-weight:600; }
.ctc-cell.blank { color:var(--muted); background:transparent; font-weight:400; }
.ctc-cell.bound { color:var(--color); background:color-mix(in srgb, var(--accent) 18%, transparent); }
.ctc-collapsed { font-family:var(--font-body); font-size:1.4rem; letter-spacing:.02em; margin:.4rem 0;
  padding:.6rem .8rem; background:var(--bg-raised); border:1px solid var(--accent); border-radius:var(--radius);
  overflow-wrap:anywhere; word-break:normal; }
.ctc-legend { display:flex; flex-wrap:wrap; gap:1rem; font-size:.76rem; color:var(--muted); margin:.4rem 0; }
.ctc-legend span { display:inline-flex; align-items:center; gap:.3rem; }
.ctc-swatch { inline-size:.9em; block-size:.9em; border-radius:3px; display:inline-block; }
.align-row { display:flex; flex-wrap:wrap; gap:.3rem; margin:.4rem 0; }
.align-word { display:inline-block; padding:.2rem .55rem; border-radius:999px;
  border:1px solid var(--border); background:var(--bg-raised); font-family:var(--font-body); font-size:1.1rem; }
.align-word .t { color:var(--muted); font-size:.7rem; margin-inline-start:.35rem; font-family:var(--font-mono); }
`;
