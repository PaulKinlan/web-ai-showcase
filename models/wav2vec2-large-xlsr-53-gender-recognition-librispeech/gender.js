// Front-end helpers for the wav2vec2 XLS-R voice-label pages. Keeps each page thin: it owns the
// worker handshake, decodes/records audio into the 16 kHz mono Float32Array the model wants, draws
// the waveform, and renders the two-class score bars + decision margin. All inference lives in
// worker.js (off the main thread).

import { WorkerClient } from "/web-ai-showcase/lib/worker-protocol.js";

const WORKER_URL =
  "/web-ai-showcase/models/wav2vec2-large-xlsr-53-gender-recognition-librispeech/worker.js";
const TARGET_RATE = 16000;

export class GenderEngine {
  constructor() {
    this.ready = false;
    this.device = "wasm";
    this._controller = null;
    this.client = new WorkerClient({
      url: WORKER_URL,
      name: "xlsr-acoustic-label",
      maxInFlight: 1,
      maxQueue: 0,
      disposeGraceMs: 300,
      onState: (state) => {
        if (state === "error" || state === "terminated") this.ready = false;
      },
    });
    // Kept as a read-only compatibility handle for diagnostics; callers must use dispose().
    this.worker = this.client.worker;
  }

  async load(onProgress, device) {
    await this.client.ready;
    const { result } = await this.client.request("load", { device }, {
      channel: "load",
      onProgress,
    });
    this.ready = true;
    this.device = result.device;
    this.runtimeEvidence = result;
    return result.device;
  }

  /**
   * Classify one waveform. A private copy is transferred (not cloned), so the page may safely reuse
   * its selected clip. Single-flight backpressure rejects overlapping runs deterministically.
   */
  async classify(audio, opts = {}) {
    if (!this.ready) throw new Error("Model worker is not ready");
    const transferred = audio.slice();
    this._controller = new AbortController();
    try {
      const { result } = await this.client.request(
        "classify",
        { audio: transferred.buffer, device: opts.device },
        {
          transfer: [transferred.buffer],
          signal: this._controller.signal,
          channel: opts.channel || "inference",
        },
      );
      // Inspectable acceptance/debug signal bound to the genuine versioned-worker response. It
      // contains raw logits + pinned runtime evidence, never a synthetic UI-only reconstruction.
      globalThis.dispatchEvent?.(new CustomEvent("xlsr-worker-result", { detail: result }));
      return result;
    } finally {
      this._controller = null;
    }
  }

  cancel(reason = "Inference cancelled") {
    this._controller?.abort(new DOMException(reason, "AbortError"));
  }

  async dispose(reason = "Model worker disposed") {
    this.ready = false;
    this.cancel(reason);
    await this.client.terminate(new Error(reason));
  }

  get pending() {
    return this.client.pending;
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
 * Wild page sensitivity probe. Rate resampling changes pitch, spectrum/formants, tempo, duration,
 * and temporal structure together; it isolates neither pitch nor timbre.
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
  const accent = cs.getPropertyValue("--accent").trim();
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
 * highest score first), plus the score margin. `threshold` (0..1) is the abstention bar: when
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
    verdict.textContent = `Below the display threshold — score margin ${
      (margin * 100).toFixed(1)
    }% < ${
      (threshold * 100).toFixed(0)
    }%. The page abstains because the two uncalibrated scores are too close.`;
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
.verdict.abstain { color:var(--warn); font-weight:600; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem;
  color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; overflow-wrap:anywhere; }
.field-row { display:flex; flex-wrap:wrap; gap:.6rem; align-items:center; margin:.6rem 0; }
.ethics { border:1px solid var(--warn); border-inline-start-width:4px; border-radius:var(--radius);
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
.batch-table .flag { color:var(--warn); font-weight:700; }
.sweep-chart { inline-size:100%; block-size:auto; display:block; background:var(--bg-raised);
  border:1px solid var(--border); border-radius:var(--radius); margin-top:.5rem; }
.provenance { font-size:.8rem; color:var(--muted); }
.provenance code, .inside-table code { overflow-wrap:anywhere; word-break:break-all; }
`;
