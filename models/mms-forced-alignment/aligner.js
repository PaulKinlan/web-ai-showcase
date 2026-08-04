// Front-end helpers for the MMS forced-alignment pages.
// All inference (CTC forward pass + monotonic alignment) runs in worker.js, off the main thread.

const WORKER_URL = "/web-ai-showcase/models/mms-forced-alignment/worker.js";
const TARGET_RATE = 16000;

export class AlignerEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
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

  load(onProgress) {
    if (onProgress) this.onProgress = onProgress;
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load" });
    });
  }

  /** Align a 16 kHz mono Float32Array against a known transcript. */
  align(audio, transcript, audioDur) {
    const id = ++this._id;
    const copy = audio.slice();
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "align", id, audio: copy, transcript, audioDur }, [copy.buffer]);
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
        for (const t of this.stream.getTracks()) t.stop();
        const blob = new Blob(this.chunks, { type: this.rec.mimeType || "audio/webm" });
        resolve({ blob, url: URL.createObjectURL(blob) });
      });
      this.rec.stop();
    });
  }
}

/** Format seconds as SRT timestamp (HH:MM:SS,mmm). */
export function srtTime(s) {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const mm = Math.floor(sec);
  const ms = Math.round((sec - mm) * 1000);
  const pad = (n, w = 2) => String(n).padStart(w, "0");
  return `${pad(h)}:${pad(m)}:${pad(mm)},${pad(ms, 3)}`;
}

/** Render words as clickable chips with timestamps; onPick(word) on click/keyboard. */
export function renderWordChips(container, words, onPick) {
  container.replaceChildren(
    ...words.map((w, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "wchip";
      b.dataset.i = String(i);
      const timed = w.start !== null && w.end !== null;
      b.disabled = !timed;
      b.classList.toggle("untimed", !timed);
      b.innerHTML = timed
        ? `<span class="w">${w.text}</span> <span class="t">${w.start.toFixed(2)}–${w.end.toFixed(2)}s</span>`
        : `<span class="w">${w.text}</span> <span class="t">no match</span>`;
      if (onPick) b.addEventListener("click", () => onPick(w, i));
      return b;
    }),
  );
}

export const ALIGN_CSS = `
textarea.prompt { inline-size: 100%; font: inherit; padding: .6rem .7rem; border-radius: 8px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); min-block-size: 4.2rem; resize: vertical; }
.wchip { display: inline-flex; align-items: center; gap: .35rem; font: inherit; font-size: .85rem;
  padding: .3rem .55rem; margin: .15rem; border-radius: 999px; border: 1px solid var(--border);
  background: var(--bg-raised); color: var(--color); cursor: pointer; }
.wchip .w { font-weight: 600; }
.wchip .t { font-family: var(--font-mono); font-size: .72rem; color: var(--muted); }
.wchip.active, .wchip:hover, .wchip:focus-visible { border-color: var(--accent); background: color-mix(in srgb, var(--accent) 14%, transparent); }
.wchip.untimed { opacity: .55; cursor: default; }
.wchip.active .t { color: var(--color); }
.words { display: flex; flex-wrap: wrap; gap: .1rem; margin-top: .6rem; }
.dropzone { border: 1px dashed var(--border); border-radius: 8px; padding: .7rem; text-align: center;
  color: var(--muted); cursor: pointer; margin: .5rem 0; }
.dropzone.drag { border-color: var(--accent); color: var(--accent); }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.sample-row { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0; }
.chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.chip:hover, .chip:focus-visible { border-color: var(--accent); }
pre.srt { font-family: var(--font-mono); font-size: .8rem; white-space: pre-wrap; background: var(--bg-raised);
  border: 1px solid var(--border); border-radius: 8px; padding: .6rem .7rem; }
.license-note { border: 1px solid var(--border); border-inline-start: 4px solid var(--accent); border-radius: var(--radius, 8px);
  background: var(--bg-raised); color: var(--color); padding: .7rem .8rem; margin: .9rem 0; font-size: .86rem; line-height: 1.55; }
.license-note strong { color: var(--color); }
audio { inline-size: 100%; margin-top: .5rem; }
`;
