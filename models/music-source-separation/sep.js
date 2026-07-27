// Front-end helpers for the Music source separation page. Owns the worker handshake, turns uploaded audio
// into the 44.1 kHz stereo segment Demucs wants, and plays back any mix-and-match of the four stems (uncheck
// vocals → karaoke; only vocals → acapella). All inference lives in worker.js, off the main thread.
// Bundled sample: an actual openly licensed SONG (SONG below); visitors can also upload their own audio.

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

/** Display name for a stem (shared by every route — do not copy per page). */
export const prettyName = (n) =>
  n === "other" ? "Other (keys/guitar)" : n.charAt(0).toUpperCase() + n.slice(1);

/** Measured energy of a stereo pair (shared by every route): RMS, absolute peak, and peak in dBFS
 *  ("-∞" for digital silence). */
export function measureStereo(l, r) {
  let sum = 0, peak = 0;
  for (let i = 0; i < l.length; i++) {
    sum += l[i] * l[i] + r[i] * r[i];
    const a = Math.abs(l[i]), b = Math.abs(r[i]);
    if (a > peak) peak = a;
    if (b > peak) peak = b;
  }
  return {
    rms: Math.sqrt(sum / (2 * l.length)),
    peak,
    peakDb: peak > 0 ? (20 * Math.log10(peak)).toFixed(1) : "-∞",
  };
}

/** Sum stems sample-by-sample with optional per-stem linear gains (a stem missing from `gains` is left
 *  OUT when a map is given; without a map every stem sums at unity). The same gain maths the live
 *  GainNodes do — shared by the karaoke/acapella exports and the remix-deck download. */
export function sumStems(stems, len, gains = null) {
  const l = new Float32Array(len), r = new Float32Array(len);
  for (const s of stems) {
    const g = gains ? (gains[s.name] ?? 0) : 1;
    if (!g) continue;
    for (let i = 0; i < len; i++) {
      l[i] += s.l[i] * g;
      r[i] += s.r[i] * g;
    }
  }
  return { l, r };
}

export class SepEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.onProgress = null;
    this._loadWaiters = [];
    this._pending = new Map();
    this._id = 0;
    this._disposed = false;
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
  isDisposed() {
    return this._disposed === true;
  }
  /** GENUINE teardown: reject all pending work, tell the worker to release the ONNX session, terminate
   *  the worker, and drop the reference. After disposal the engine cannot load or separate again. */
  async dispose() {
    if (this._disposed) return true;
    this._disposed = true;
    this.ready = false;
    const err = new Error("engine disposed");
    for (const w of this._loadWaiters) w.reject(err);
    this._loadWaiters = [];
    for (const [, p] of this._pending) p.reject(err);
    this._pending.clear();
    try {
      this.worker.postMessage({ type: "dispose" });
      await new Promise((r) => setTimeout(r, 150)); // let the worker release the session first
    } catch { /* worker already gone */ }
    try {
      this.worker.terminate();
    } catch { /* ignore */ }
    this.worker = null;
    return true;
  }
  load(onProgress) {
    if (this._disposed) return Promise.reject(new Error("engine disposed"));
    if (onProgress) this.onProgress = onProgress;
    if (this.ready) return Promise.resolve(this.device);
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load" });
    });
  }
  /** Separate a 44.1 kHz stereo segment ({ch0,ch1,len}) → { stems:[{name,l,r}], len, ms, device }. */
  separate({ ch0, ch1, len }) {
    if (this._disposed || !this.worker) return Promise.reject(new Error("engine disposed"));
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

/** Close the shared AudioContext + drop the decoded song buffer (genuine resource release). */
export async function closeAudioCtx() {
  _songBuf = null;
  if (_ctx) {
    try {
      await _ctx.close();
    } catch { /* already closed */ }
    _ctx = null;
  }
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

/** The bundled sample song: "The CC BY Song" by loveshadow, additional lyrics by Victor Stone
 *  (CC BY 3.0, via ccMixter) — an actual openly licensed SONG with sung vocals, fittingly about Creative
 *  Commons licences. The complete 2:22 track ships UNMODIFIED (the original MP3 upload,
 *  sha256 9ae38593020674f9f89879f79163031392f00a937b5332365f504be24b2e91aa). The default sample window is
 *  exposed honestly on the page. Provenance: CREDITS.md. */
export const SONG = {
  url: "/web-ai-showcase/models/music-source-separation/song.mp3",
  title: "The CC BY Song",
  artist: "loveshadow",
  license: "CC BY 3.0",
  source: "https://ccmixter.org/files/Loveshadow/29635",
  offsetSec: 0,
  durationSec: 141.67,
};

let _songBuf = null;
async function songBuffer() {
  if (!_songBuf) {
    const res = await fetch(SONG.url);
    if (!res.ok) throw new Error(`song fetch failed (${res.status})`);
    _songBuf = await audioCtx().decodeAudioData(await res.arrayBuffer());
  }
  return _songBuf;
}

/** A SEG-length window of the bundled song at `offsetSec` (default: the opening at 0.0 s — the song
 *  sings from the first bar, so no offset is needed). */
export async function songSegment(offsetSec = SONG.offsetSec) {
  const buf = await songBuffer();
  const start = Math.max(0, Math.min(Math.floor(offsetSec * SR), buf.length - SEG));
  const len = Math.min(SEG, buf.length - start);
  const ch = (i) => buf.getChannelData(Math.min(i, buf.numberOfChannels - 1));
  return {
    ch0: Float32Array.from(ch(0).slice(start, start + len)),
    ch1: Float32Array.from(ch(1).slice(start, start + len)),
    len,
    offsetSec: start / SR,
  };
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
  async play(enabledNames, rate = 1) {
    this.stop();
    if (this.ctx.state === "suspended") await this.ctx.resume();
    const t0 = this.ctx.currentTime + 0.05;
    let live = 0;
    for (const name of enabledNames) {
      const buf = this.buffers[name];
      if (!buf) continue;
      const src = this.ctx.createBufferSource();
      src.buffer = buf;
      src.playbackRate.value = rate; // real resampling playback (practice mode slows a part down)
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
  /** Stop playback and drop the stem buffers (genuine teardown; the ctx closes via closeAudioCtx). */
  dispose() {
    this.stop();
    this.buffers = {};
  }
}

/** A live stem mixer for the Wild demo: loops the separated window with one GainNode per stem, so pads
 *  can mute/solo stems WHILE the loop plays (a real remix, not re-rendered playback). */
export class StemMixer {
  constructor() {
    this.ctx = audioCtx();
    this.buffers = {};
    this.gains = {};
    this.sources = [];
    this.playing = false;
  }
  setStems(stems, len) {
    this.stop();
    this.buffers = {};
    for (const s of stems) {
      const buf = this.ctx.createBuffer(2, len, SR);
      buf.copyToChannel(s.l, 0);
      buf.copyToChannel(s.r, 1);
      this.buffers[s.name] = buf;
    }
  }
  /** Fade a stem in/out live (20 ms ramp, no clicks). Works during playback. */
  setStemOn(name, on) {
    const g = this.gains[name];
    if (g) g.gain.setTargetAtTime(on ? 1 : 0, this.ctx.currentTime, 0.02);
  }
  /** Set a stem's gain live (20 ms ramp, no clicks). `v` is linear: 1 = unity, >1 boosts, 0 kills. */
  setStemGain(name, v) {
    const g = this.gains[name];
    if (g) g.gain.setTargetAtTime(Math.max(0, v), this.ctx.currentTime, 0.02);
  }
  isOn(name) {
    const g = this.gains[name];
    return g ? g.gain.value > 0.5 : true;
  }
  /** A live AnalyserNode tapped off a stem's gain (for metering / envelope-driven effects like
   *  auto-ducking). Created lazily; reused across play/stop cycles (the stem gain reconnects on play). */
  analyserFor(name) {
    this._analysers ||= {};
    if (!this._analysers[name]) {
      const a = this.ctx.createAnalyser();
      a.fftSize = 2048;
      this._analysers[name] = a;
    }
    // (Re)tap: the gain node is recreated on each play(), so (re)connect if needed.
    const g = this.gains[name];
    if (g && this._analysers[name]._tap !== g) {
      try {
        g.connect(this._analysers[name]);
      } catch { /* already connected */ }
      this._analysers[name]._tap = g;
    }
    return this._analysers[name];
  }
  stop() {
    for (const s of this.sources) {
      try {
        s.stop();
      } catch { /* already stopped */ }
    }
    this.sources = [];
    this.gains = {};
    this.playing = false;
  }
  async play() {
    this.stop();
    if (this.ctx.state === "suspended") await this.ctx.resume();
    const t0 = this.ctx.currentTime + 0.05;
    for (const name of STEMS) {
      const buf = this.buffers[name];
      if (!buf) continue;
      const src = this.ctx.createBufferSource();
      src.buffer = buf;
      src.loop = true;
      const g = this.ctx.createGain();
      src.connect(g).connect(this.ctx.destination);
      src.start(t0);
      this.sources.push(src);
      this.gains[name] = g;
    }
    this.playing = this.sources.length > 0;
  }
  /** Stop the loop and drop buffers/gains (genuine teardown; the ctx closes via closeAudioCtx). */
  dispose() {
    this.stop();
    this.buffers = {};
  }
}

/** Resample a stereo segment to 16 kHz mono — what the instrument classifier (multi-model demo) wants. */
export function toMono16k(l, r, len) {
  const n = Math.max(1, Math.floor((len * 16000) / SR));
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const j = Math.min(len - 1, Math.floor((i * SR) / 16000));
    out[i] = (l[j] + r[j]) * 0.5;
  }
  return out;
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
