// Front-end helpers for the OuteTTS pages. Keeps each page thin: the worker handshake, a waveform
// renderer, and the shared widget CSS. All synthesis (the Qwen2 LLM + the WavTokenizer codec decode)
// happens in the worker, off the main thread.

/** Client for the OuteTTS worker. */
export class OuteClient {
  constructor(workerUrl = "/web-ai-showcase/models/outetts/worker.js") {
    this.worker = new Worker(workerUrl, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.speakers = [];
    this._loadWaiters = [];
    this._pending = new Map();
    this._id = 0;
    this.onProgress = null;
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
      this.speakers = msg.speakers || [];
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
    if (this.ready) return Promise.resolve({ device: this.device, speakers: this.speakers });
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load" });
    });
  }

  /** Generate speech → { wav:ArrayBuffer, pcm:Float32Array, rate, durSec, audioTokens, ms, rtf, device } */
  generate(opts) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, ...opts });
    });
  }
}

/** Turn a WAV ArrayBuffer into an object URL for an <audio> element. */
export function wavURL(wav) {
  return URL.createObjectURL(new Blob([wav], { type: "audio/wav" }));
}

/** Split a passage into sentence-ish chunks for sequential narration. */
export function splitSentences(text) {
  return text
    .replace(/\s+/g, " ")
    .match(/[^.!?]+[.!?]*/g)
    ?.map((s) => s.trim())
    .filter(Boolean) ?? [];
}

/** Concatenate several Float32 PCM chunks (optionally with a short silent gap) into one buffer. */
export function concatPcm(chunks, rate, gapSec = 0.15) {
  const gap = Math.round(rate * gapSec);
  const total = chunks.reduce((n, c) => n + c.length + gap, 0);
  const out = new Float32Array(total);
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length + gap;
  }
  return out.subarray(0, Math.max(0, o - gap));
}

/** Encode a mono Float32 waveform to a 16-bit PCM WAV ArrayBuffer (client-side, for stitched clips). */
export function encodeWav(samples, rate) {
  const n = samples.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const dv = new DataView(buf);
  const wr = (o, s) => {
    for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i));
  };
  wr(0, "RIFF");
  dv.setUint32(4, 36 + n * 2, true);
  wr(8, "WAVE");
  wr(12, "fmt ");
  dv.setUint32(16, 16, true);
  dv.setUint16(20, 1, true);
  dv.setUint16(22, 1, true);
  dv.setUint32(24, rate, true);
  dv.setUint32(28, rate * 2, true);
  dv.setUint16(32, 2, true);
  dv.setUint16(34, 16, true);
  wr(36, "data");
  dv.setUint32(40, n * 2, true);
  let o = 44;
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    dv.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    o += 2;
  }
  return buf;
}

/**
 * Draw a downsampled peak waveform of a Float32 PCM buffer into an SVG inside `container`.
 * A real picture of the generated audio — not decorative.
 */
export function renderWaveform(container, pcm, { width = 640, height = 120 } = {}) {
  const buckets = Math.min(width, 480);
  const per = Math.max(1, Math.floor(pcm.length / buckets));
  const mid = height / 2;
  let path = "";
  for (let b = 0; b < buckets; b++) {
    let min = 1, max = -1;
    const start = b * per;
    for (let i = start; i < start + per && i < pcm.length; i++) {
      const v = pcm[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const x = (b / buckets) * width;
    const y1 = mid - max * mid * 0.95;
    const y2 = mid - min * mid * 0.95;
    path += `M${x.toFixed(1)},${y1.toFixed(1)} L${x.toFixed(1)},${y2.toFixed(1)} `;
  }
  container.innerHTML =
    `<svg viewBox="0 0 ${width} ${height}" class="wave-svg" role="img" aria-label="Waveform of the generated speech">
      <line x1="0" y1="${mid}" x2="${width}" y2="${mid}" class="wave-axis"></line>
      <path d="${path}" class="wave-path" fill="none" stroke-width="1.2"></path>
    </svg>`;
}

/** Shared inline styles for the OuteTTS widgets. Injected once per page. */
export const OUTE_CSS = `
.field { display: flex; flex-direction: column; gap: .3rem; font-size: .82rem; min-inline-size: 0; }
.controls-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; align-items: end; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.wave-svg { inline-size: 100%; max-inline-size: 100%; block-size: auto; background: var(--bg-raised);
  border: 1px solid var(--border); border-radius: 8px; margin-top: .5rem; }
.wave-axis { stroke: var(--border); stroke-width: 1; }
.wave-path { stroke: var(--accent); }
.audio-row { display: flex; flex-wrap: wrap; gap: .6rem; align-items: center; margin-top: .6rem; }
.audio-row audio { max-inline-size: 100%; }
.range-row { display: flex; align-items: center; gap: .6rem; }
.range-row input[type=range] { flex: 1; min-inline-size: 0; }
.range-val { font-family: var(--font-mono); min-inline-size: 2.6rem; text-align: end; }
.chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px; border: 1px solid var(--border);
  background: var(--bg-raised); color: var(--color); cursor: pointer; }
.chip:hover { border-color: var(--accent); }
.chip-row { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0; }
.pipe { display: flex; flex-wrap: wrap; gap: .5rem; align-items: stretch; margin: .6rem 0; }
.pipe-stage { flex: 1 1 160px; min-inline-size: 0; border: 1px solid var(--border); border-radius: 8px;
  background: var(--bg-raised); padding: .6rem .7rem; }
.pipe-stage h4 { margin: 0 0 .3rem; font-size: .82rem; }
.pipe-stage p { margin: 0; font-size: .76rem; color: var(--muted); }
.pipe-arrow { align-self: center; color: var(--muted); font-size: 1.2rem; }
`;
