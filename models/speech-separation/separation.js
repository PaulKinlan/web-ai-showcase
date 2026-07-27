// Front-end helpers for the speech-separation page: the worker handshake, audio decode/mix/encode, and
// waveform drawing. All inference lives in worker.js (off the main thread).

export const SR = 16000;

export class SeparationEngine {
  constructor() {
    this.worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    this.ready = false;
    this._loadWaiters = [];
    this._pending = new Map();
    this._id = 0;
    this.onProgress = null;
    this.device = "wasm";
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
  /** Separate a mono 16 kHz mixture → { s1, s2, sr, ms }. */
  separate(mix) {
    const id = ++this._id;
    const m = new Float32Array(mix);
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "separate", id, mix: m }, [m.buffer]);
    });
  }
}

/** Decode encoded audio (any format) to mono Float32 at 16 kHz. */
export async function decodeTo16kMono(arrayBuffer) {
  const ctx = new OfflineAudioContext(1, 1, SR);
  const buf = await ctx.decodeAudioData(arrayBuffer);
  if (buf.numberOfChannels === 1) return buf.getChannelData(0).slice();
  const out = new Float32Array(buf.length);
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const ch = buf.getChannelData(c);
    for (let i = 0; i < ch.length; i++) out[i] += ch[i] / buf.numberOfChannels;
  }
  return out;
}

function peakNorm(x, peak = 0.98) {
  let m = 1e-9;
  for (const v of x) m = Math.max(m, Math.abs(v));
  const k = peak / m;
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] * k;
  return out;
}

/** Encode a mono Float32 clip to a 16-bit PCM WAV Blob (for <audio> playback / download). */
export function floatToWavBlob(samples, sr = SR) {
  const n = samples.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const dv = new DataView(buf);
  const W = (o, s) => {
    for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i));
  };
  W(0, "RIFF");
  dv.setUint32(4, 36 + n * 2, true);
  W(8, "WAVE");
  W(12, "fmt ");
  dv.setUint32(16, 16, true);
  dv.setUint16(20, 1, true);
  dv.setUint16(22, 1, true);
  dv.setUint32(24, sr, true);
  dv.setUint32(28, sr * 2, true);
  dv.setUint16(32, 2, true);
  dv.setUint16(34, 16, true);
  W(36, "data");
  dv.setUint32(40, n * 2, true);
  const norm = peakNorm(samples, 0.98);
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, norm[i]));
    dv.setInt16(44 + i * 2, s * 32767, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

/** Draw a waveform of a mono clip onto a canvas. */
export function drawWave(canvas, samples, color = "#2bb59a") {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height, mid = H / 2;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  const step = Math.max(1, Math.floor(samples.length / W));
  for (let x = 0; x < W; x++) {
    let min = 1, max = -1;
    for (let i = 0; i < step; i++) {
      const v = samples[x * step + i] || 0;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    ctx.moveTo(x + 0.5, mid + min * mid);
    ctx.lineTo(x + 0.5, mid + max * mid);
  }
  ctx.stroke();
}

export const SEP_CSS = `
  .sep-block { margin: 0.8rem 0; }
  .sep-block h4 { margin: 0 0 0.3rem; font-size: 0.95rem; }
  .sep-row { display: flex; flex-wrap: wrap; gap: 0.8rem; align-items: safe center; }
  .sep-wave { inline-size: 100%; max-inline-size: 100%; block-size: 64px; background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 8px; display: block; }
  .sep-block audio { inline-size: 100%; max-inline-size: 26rem; }
  .sep-out { display: flex; flex-wrap: wrap; gap: 1rem; }
  .sep-out > div { flex: 1 1 15rem; min-inline-size: 0; }
  .sep-dropzone { border: 2px dashed var(--border-strong); border-radius: 10px; padding: 0.9rem; margin-block: 0.6rem; font-size: 0.9rem; }
  .sep-dropzone label { display: block; margin-block-end: 0.5rem; }
  .sep-dropzone input { max-inline-size: 100%; min-block-size: 44px; }
  .sep-dropzone.is-dragging { border-color: var(--accent); background: var(--bg-secondary); }
  #separateBtn { min-block-size: 44px; }
  .sep-status { font-family: var(--font-mono, monospace); font-size: 0.9rem; margin: 0.6rem 0; }
  .sep-status.err { color: var(--bad); }
`;
