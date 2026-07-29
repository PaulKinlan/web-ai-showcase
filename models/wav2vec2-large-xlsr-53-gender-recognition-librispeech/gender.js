// Front-end helpers for the wav2vec2 XLS-R voice-label pages. Keeps each page thin: it owns the
// worker handshake, decodes/records audio into the 16 kHz mono Float32Array the model wants, draws
// the waveform, and renders the two-class score bars + decision margin. All inference lives in
// worker.js (off the main thread).

const WORKER_URL =
  "/web-ai-showcase/models/wav2vec2-large-xlsr-53-gender-recognition-librispeech/worker.js";
const TARGET_RATE = 16000;

export class GenderEngine {
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

  /** Classify a 16 kHz mono Float32Array. Returns { labels, logits, margin, ms, device, durationS }. */
  classify(audio, opts) {
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

/**
 * Re-render a decoded clip at a different playback rate (pitch AND tempo shift together — the
 * classic "chipmunk / slow-mo" resampling), returned as a 16 kHz mono Float32Array. Used by the
 * Wild page to probe how the acoustic label tracks pitch/timbre. Honest: this is rate resynthesis,
 * not a formant-preserving pitch shifter.
 */
export async function renderAtRate(pcm, rate) {
  const src = new OfflineAudioContext(1, pcm.length, TARGET_RATE);
  const buf = src.createBuffer(1, pcm.length, TARGET_RATE);
  buf.copyToChannel(pcm, 0);
  const off = new OfflineAudioContext(1, Math.ceil(pcm.length / rate), TARGET_RATE);
  const node = off.createBufferSource();
  node.buffer = buf;
  node.playbackRate.value = rate;
  node.connect(off.destination);
  node.start();
  const rendered = await off.startRendering();
  return rendered.getChannelData(0);
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Draw a mono waveform into a <canvas>, matching the design system's accent colour. */
export function drawWaveform(canvas, pcm) {
  const cs = getComputedStyle(document.body);
  const accent = cs.getPropertyValue("--accent").trim() || "#4b3aff";
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
  ctx.globalAlpha = 0.9;
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

/**
 * Render the checkpoint's two acoustic voice labels as accessible score bars (BOTH classes always,
 * most confident first), plus the decision margin. `threshold` (0..1) is the abstention bar: when
 * the winning margin is below it the verdict line says so instead of naming a label.
 */
export function renderScoreBars(container, labels, threshold = 0) {
  const rows = labels.map(({ label, score }, i) => {
    const row = document.createElement("div");
    row.className = "score-row" + (i === 0 ? " top" : "");
    const pct = (score * 100).toFixed(1);
    row.innerHTML = `<span class="score-label">“${escapeHTML(label)}”</span>` +
      `<span class="score-track"><span class="score-fill" style="inline-size:${
        Math.max(2, score * 100)
      }%"></span></span>` +
      `<span class="score-val">${pct}%</span>`;
    return row;
  });
  const margin = Math.abs(labels[0].score - labels[1].score);
  const verdict = document.createElement("p");
  verdict.className = "verdict";
  if (threshold > 0 && margin < threshold) {
    verdict.classList.add("abstain");
    verdict.textContent = `Below the review threshold — margin ${(margin * 100).toFixed(1)}% < ${
      (threshold * 100).toFixed(0)
    }%. A human would need to look at this one; the model is not confident enough to label it.`;
  } else {
    verdict.textContent =
      `Acoustic voice label: “${labels[0].label}” — margin ${(margin * 100).toFixed(1)}%. ` +
      `This describes how the clip SOUNDS to the checkpoint, not who the speaker is.`;
  }
  container.replaceChildren(...rows, verdict);
}

export const GENDER_CSS = `
.wave-wrap { margin:.6rem 0; }
.wave { inline-size:100%; block-size:80px; display:block; background:var(--bg-raised);
  border:1px solid var(--border); border-radius:var(--radius); }
.audio-row { display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; margin:.5rem 0; }
.audio-row audio { block-size:34px; max-inline-size:100%; }
.sample-row { display:flex; flex-wrap:wrap; gap:.4rem; margin:.5rem 0; }
.scores { display:flex; flex-direction:column; gap:.35rem; margin:.5rem 0; }
.score-row { display:grid; grid-template-columns:minmax(8ch,10rem) 1fr 5ch; align-items:center; gap:.5rem;
  font-size:.85rem; }
.score-row.top .score-label { font-weight:700; color:var(--accent); }
.score-label { text-align:right; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.score-track { block-size:.85rem; background:var(--bg-secondary); border-radius:999px; overflow:hidden;
  border:1px solid var(--border); }
.score-fill { display:block; block-size:100%; background:var(--accent); border-radius:999px; }
.score-val { font-family:var(--font-mono); font-size:.78rem; color:var(--muted); text-align:right; }
.verdict { font-size:.88rem; margin:.5rem 0 0; }
.verdict.abstain { color:var(--warn, #b45309); font-weight:600; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem;
  color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; overflow-wrap:anywhere; }
.field-row { display:flex; flex-wrap:wrap; gap:.6rem; align-items:center; margin:.6rem 0; }
.ethics { border:1px solid var(--warn, #b45309); border-inline-start-width:4px; border-radius:var(--radius);
  background:var(--bg-raised); padding:.8rem 1rem; }
.ethics h2, .ethics h3 { margin-top:0; }
.inside-table { inline-size:100%; border-collapse:collapse; font-size:.82rem; margin-top:.5rem; }
.inside-table th, .inside-table td { text-align:left; padding:.3rem .5rem; border-bottom:1px solid var(--border);
  font-family:var(--font-mono); overflow-wrap:anywhere; }
.inside-table th { color:var(--muted); font-weight:600; }
.logit-line { font-family:var(--font-mono); font-size:.82rem; margin-top:.5rem; overflow-wrap:anywhere; }
.batch-table { inline-size:100%; border-collapse:collapse; font-size:.82rem; margin-top:.5rem; display:block;
  overflow-x:auto; }
.batch-table table { inline-size:100%; border-collapse:collapse; }
.batch-table th, .batch-table td { text-align:left; padding:.35rem .5rem; border-bottom:1px solid var(--border);
  white-space:nowrap; }
.batch-table th { color:var(--muted); font-weight:600; }
.batch-table .flag { color:var(--warn, #b45309); font-weight:700; }
.sweep-chart { inline-size:100%; block-size:auto; display:block; background:var(--bg-raised);
  border:1px solid var(--border); border-radius:var(--radius); margin-top:.5rem; }
.provenance { font-size:.8rem; color:var(--muted); }
`;
