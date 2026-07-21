// Front-end helpers for the Music source separation page. Owns the worker handshake, turns audio into the
// 44.1 kHz stereo segment Demucs wants, synthesises the built-in "spoken word over a beat" sample, and plays
// back any mix-and-match of the four stems (uncheck vocals → karaoke; only vocals → acapella). All inference
// lives in worker.js, off the main thread.

const WORKER_URL = "/web-ai-showcase/models/music-source-separation/worker.js";
export const SEG = 343980;
export const SR = 44100;
export const STEMS = ["drums", "bass", "other", "vocals"];
export const STEM_META = {
  drums: { emoji: "🥁", hue: 12 },
  bass: { emoji: "🎸", hue: 275 },
  other: { emoji: "🎹", hue: 190 },
  vocals: { emoji: "🎤", hue: 130 },
};

export class SepEngine {
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
  /** Separate a 44.1 kHz stereo segment ({ch0,ch1,len}) → { stems:[{name,l,r}], len, ms, device }. */
  separate({ ch0, ch1, len }) {
    const id = ++this._id;
    const a = ch0.slice(), b = ch1.slice();
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "separate", id, ch0: a, ch1: b, len }, [a.buffer, b.buffer]);
    });
  }
}

let _ctx = null;
export function audioCtx() {
  if (!_ctx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    _ctx = new AC({ sampleRate: SR });
  }
  return _ctx;
}

/** Decode any audio ArrayBuffer → 44.1 kHz STEREO, trimmed to the first SEG samples (≈7.8 s). */
export async function decodeToSegment(arrayBuffer) {
  const decoded = await audioCtx().decodeAudioData(arrayBuffer.slice(0));
  const frames = Math.min(SEG, Math.ceil(decoded.duration * SR));
  const off = new OfflineAudioContext(2, frames, SR);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const r = await off.startRendering();
  const ch0 = Float32Array.from(r.getChannelData(0));
  const ch1 = Float32Array.from(r.numberOfChannels > 1 ? r.getChannelData(1) : r.getChannelData(0));
  return { ch0, ch1, len: ch0.length };
}

/** Build the built-in sample: JFK's (public-domain) spoken voice over a synthesised beat — a first-party,
 *  clean "vocals + backing" mix. (Demucs's vocals stem is trained on SUNG vocals, so a spoken voice lands in
 *  "other" — the page says so; upload a song with singing to isolate real vocals.) */
export async function makeSampleMix() {
  const buf = await (await fetch("jfk.wav")).arrayBuffer();
  const decoded = await audioCtx().decodeAudioData(buf);
  const off = new OfflineAudioContext(1, SEG, SR);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const voice = (await off.startRendering()).getChannelData(0);
  const mix = new Float32Array(SEG);
  const spb = 60 / 100; // 100 bpm
  for (let i = 0; i < SEG; i++) {
    const t = i / SR;
    const beat = t / spb, ph = beat - Math.floor(beat), b = Math.floor(beat) % 4;
    const kick = ph < 0.08
      ? Math.sin(2 * Math.PI * (60 - 30 * ph) * t) * Math.exp(-ph * 12) * 0.7
      : 0;
    const hp = (beat * 2) % 1;
    const hat = (Math.random() * 2 - 1) * Math.exp(-hp * spb * 50) * 0.1;
    const bass = Math.sin(2 * Math.PI * (b % 2 ? 98 : 65) * t) * 0.22;
    mix[i] = (voice[i] || 0) * 0.9 + kick + hat + bass;
  }
  const ch = Float32Array.from(mix);
  return { ch0: ch, ch1: ch.slice(), len: SEG };
}

/** A little multi-stem player: play any subset of stems together (checkboxes = the live mix). */
export class StemPlayer {
  constructor() {
    this.ctx = audioCtx();
    this.buffers = {}; // name -> AudioBuffer
    this.sources = [];
    this.playing = false;
    this.onEnded = null;
  }
  setStems(stems, len) {
    this.buffers = {};
    for (const s of stems) {
      const buf = this.ctx.createBuffer(2, len, SR);
      buf.copyToChannel(s.l, 0);
      buf.copyToChannel(s.r, 1);
      this.buffers[s.name] = buf;
    }
  }
  stop() {
    for (const s of this.sources) {
      try {
        s.stop();
      } catch { /* already stopped */ }
    }
    this.sources = [];
    this.playing = false;
  }
  async play(enabledNames) {
    this.stop();
    if (this.ctx.state === "suspended") await this.ctx.resume();
    const t0 = this.ctx.currentTime + 0.05;
    let live = 0;
    for (const name of enabledNames) {
      const buf = this.buffers[name];
      if (!buf) continue;
      const src = this.ctx.createBufferSource();
      src.buffer = buf;
      src.connect(this.ctx.destination);
      src.start(t0);
      this.sources.push(src);
      live++;
    }
    if (live) {
      this.playing = true;
      this.sources[0].addEventListener("ended", () => {
        this.playing = false;
        this.onEnded?.();
      }, { once: true });
    }
  }
}

/** Encode a stereo pair as a 16-bit PCM WAV Blob (for stem download). */
export function encodeWav(l, r, sr = SR) {
  const n = l.length;
  const buf = new ArrayBuffer(44 + n * 4);
  const v = new DataView(buf);
  const ws = (o, s) => {
    for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i));
  };
  ws(0, "RIFF");
  v.setUint32(4, 36 + n * 4, true);
  ws(8, "WAVE");
  ws(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 2, true);
  v.setUint32(24, sr, true);
  v.setUint32(28, sr * 4, true);
  v.setUint16(32, 4, true);
  v.setUint16(34, 16, true);
  ws(36, "data");
  v.setUint32(40, n * 4, true);
  let o = 44;
  for (let i = 0; i < n; i++) {
    v.setInt16(o, Math.max(-1, Math.min(1, l[i])) * 32767, true);
    o += 2;
    v.setInt16(o, Math.max(-1, Math.min(1, r[i])) * 32767, true);
    o += 2;
  }
  return new Blob([buf], { type: "audio/wav" });
}

export function drawWaveform(canvas, ch) {
  const accent = getComputedStyle(document.body).getPropertyValue("--accent").trim() || "#4b3aff";
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 300, h = canvas.clientHeight || 40;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!ch || !ch.length) return;
  const mid = h / 2, step = Math.max(1, Math.floor(ch.length / w));
  ctx.strokeStyle = accent;
  ctx.globalAlpha = 0.85;
  for (let x = 0; x < w; x++) {
    let min = 1, max = -1;
    for (let i = 0; i < step; i++) {
      const val = ch[x * step + i] ?? 0;
      if (val < min) min = val;
      if (val > max) max = val;
    }
    ctx.beginPath();
    ctx.moveTo(x + 0.5, mid + min * mid * 0.95);
    ctx.lineTo(x + 0.5, mid + max * mid * 0.95);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

export const SEP_CSS = `
.ms-drop { border: 2px dashed var(--border); border-radius: 12px; padding: 1.1rem; text-align: center;
  background: var(--bg-raised); transition: border-color .15s, background .15s; }
.ms-drop.drag { border-color: var(--accent); background: color-mix(in srgb, var(--accent) 10%, transparent); }
.ms-tools { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; justify-content: center; margin: .3rem 0; }
.ms-btn { font: inherit; font-size: .85rem; padding: .35rem .8rem; border-radius: 8px; border: 1px solid var(--border);
  background: var(--bg-raised); color: var(--color); cursor: pointer; }
.ms-btn:hover:not([disabled]), .ms-btn:focus-visible { border-color: var(--accent); }
.ms-btn[disabled] { opacity: .5; cursor: default; }
.ms-btn.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
.ms-hint { font-size: .82rem; color: var(--muted); margin: .3rem 0; }
.ms-transport { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; margin: .8rem 0 .4rem; }
.ms-preset { display: flex; flex-wrap: wrap; gap: .4rem; }
.ms-stems { display: flex; flex-direction: column; gap: .5rem; margin-top: .4rem; }
.ms-stem { display: grid; grid-template-columns: auto 8rem 1fr auto; align-items: center; gap: .6rem;
  border: 1px solid var(--border); border-radius: 10px; padding: .5rem .7rem; background: var(--bg-raised); }
.ms-stem.off { opacity: .5; }
.ms-stem input[type=checkbox] { inline-size: 1.1rem; block-size: 1.1rem; }
.ms-name { display: inline-flex; align-items: center; gap: .4rem; font-weight: 600; }
.ms-name .em { font-size: 1.15rem; }
.ms-wave { width: 100%; height: 40px; display: block; }
.ms-dl { font: inherit; font-size: .74rem; padding: .15rem .5rem; border-radius: 6px; border: 1px solid var(--border);
  background: transparent; color: var(--color); cursor: pointer; text-decoration: none; }
.ms-dl:hover { border-color: var(--accent); }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono, monospace);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
`;
