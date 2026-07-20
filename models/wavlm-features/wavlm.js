// Front-end helpers for the WavLM feature pages. Keeps each page thin: worker handshake, audio decode
// to the 16 kHz mono Float32Array the model wants, waveform draw, heatmap paint (from the RGBA buffer
// the worker transfers back), similarity matrix + PCA scatter, and shared CSS. ALL model inference and
// the dense heatmap composite live in worker.js (off the main thread).
//
// WavLM emits per-frame 768-dim embeddings, not text. We mean-pool + L2-normalise a clip's frames into
// one utterance vector and compare vectors with cosine similarity — the basis of audio-audio
// similarity, "find a similar segment", and clustering by self-supervised features.

const WORKER_URL = "/web-ai-showcase/models/wavlm-features/worker.js";
const TARGET_RATE = 16000;

export class WavLMEngine {
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
    } else if (msg.type === "error") {
      if (msg.id != null && this._pending.has(msg.id)) {
        this._pending.get(msg.id).reject(new Error(msg.message));
        this._pending.delete(msg.id);
      } else {
        const err = new Error(msg.message);
        for (const w of this._loadWaiters) w.reject(err);
        this._loadWaiters = [];
      }
    } else if (msg.id != null && this._pending.has(msg.id)) {
      this._pending.get(msg.id).resolve(msg);
      this._pending.delete(msg.id);
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

  /** Embed one clip → { pooled:[768], frames, dim, frameMs, audioSec, ms, device, heat:{w,h,rgba} }. */
  embed(audio, audioSec) {
    const id = ++this._id;
    const copy = audio.slice();
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "embed", id, audio: copy, audioSec }, [copy.buffer]);
    });
  }

  /** Embed a bank of clips → { names, matrix (pairwise cosine), coords ([x,y] PCA), ms, device }. */
  similarity(clips) {
    const id = ++this._id;
    const payload = clips.map((c) => ({ name: c.name, audio: c.audio.slice() }));
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage(
        { type: "similarity", id, clips: payload },
        payload.map((c) => c.audio.buffer),
      );
    });
  }

  /** Slide a window over a haystack, cosine each window vs the query → { times, sims, best, ... }. */
  search(query, haystack, haySec, windowSec, hopSec) {
    const id = ++this._id;
    const q = query.slice(), h = haystack.slice();
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage(
        { type: "search", id, query: q, haystack: h, haySec, windowSec, hopSec },
        [q.buffer, h.buffer],
      );
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

/** Cosine of two equal-length arrays (assumes they're L2-normalised — pooled vectors from the worker). */
export function cosine(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

export function fmtTime(s) {
  if (s == null || Number.isNaN(s)) return "–";
  const m = Math.floor(s / 60);
  const sec = (s - m * 60).toFixed(1).padStart(4, "0");
  return `${m}:${sec}`;
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Draw a mono waveform into a <canvas>, matching the light/dark design system. `sel` = [s0,s1] in 0..1
 *  optionally highlights a selected/active region in the accent colour. */
export function drawWaveform(canvas, pcm, sel = null) {
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
  const x0 = sel ? Math.floor(w * sel[0]) : -1;
  const x1 = sel ? Math.ceil(w * sel[1]) : -1;
  for (let x = 0; x < w; x++) {
    let min = 1, max = -1;
    for (let i = 0; i < step; i++) {
      const v = pcm[x * step + i] ?? 0;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const inSel = sel && x >= x0 && x <= x1;
    ctx.strokeStyle = inSel ? accent : muted;
    ctx.globalAlpha = inSel ? 0.95 : 0.5;
    ctx.beginPath();
    ctx.moveTo(x + 0.5, mid + min * mid * 0.95);
    ctx.lineTo(x + 0.5, mid + max * mid * 0.95);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

/** Paint the worker's RGBA heatmap into a <canvas>, scaled to fit (crisp, nearest-neighbour). */
export function drawHeatmap(canvas, heat) {
  const { w, h, rgba } = heat;
  const src = document.createElement("canvas");
  src.width = w;
  src.height = h;
  src.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(rgba), w, h), 0, 0);
  const dpr = self.devicePixelRatio || 1;
  const cw = canvas.clientWidth || 600;
  const ch = canvas.clientHeight || 200;
  canvas.width = cw * dpr;
  canvas.height = ch * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, cw, ch);
  ctx.drawImage(src, 0, 0, cw, ch);
}

/** Render an N×N cosine-similarity matrix as an accessible table with heat-tinted cells. */
export function renderMatrix(el, names, matrix) {
  const cs = getComputedStyle(document.body);
  const accent = cs.getPropertyValue("--accent").trim() || "#4b3aff";
  const cell = (v) => {
    // map cosine (roughly -0.2..1) to accent alpha
    const a = Math.max(0, Math.min(1, (v + 0.2) / 1.2));
    return `background:color-mix(in srgb, ${accent} ${(a * 90).toFixed(0)}%, transparent)`;
  };
  const head = '<tr><th scope="col"><span class="vh">clip</span></th>' +
    names.map((n) => `<th scope="col">${escapeHTML(n)}</th>`).join("") + "</tr>";
  const rows = names.map((n, i) =>
    `<tr><th scope="row">${escapeHTML(n)}</th>` +
    matrix[i].map((v) => `<td style="${cell(v)}"><span class="mv">${v.toFixed(2)}</span></td>`)
      .join("") +
    "</tr>"
  ).join("");
  el.innerHTML = `<table class="sim-matrix">${head}${rows}</table>`;
}

/** Draw a PCA scatter of pooled vectors, coloured + labelled by clip. groups: name→colorIndex. */
export function drawScatter(canvas, coords, labels, colorOf) {
  const cs = getComputedStyle(document.body);
  const color = cs.getPropertyValue("--color").trim() || "#222";
  const border = cs.getPropertyValue("--border").trim() || "#ccc";
  const dpr = self.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600;
  const h = canvas.clientHeight || 320;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const pad = 44;
  const xs = coords.map((c) => c[0]), ys = coords.map((c) => c[1]);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const sx = (x) => pad + ((x - xmin) / ((xmax - xmin) || 1)) * (w - 2 * pad);
  const sy = (y) => (h - pad) - ((y - ymin) / ((ymax - ymin) || 1)) * (h - 2 * pad);
  ctx.strokeStyle = border;
  ctx.globalAlpha = 0.6;
  ctx.strokeRect(pad - 8, pad - 8, w - 2 * pad + 16, h - 2 * pad + 16);
  ctx.globalAlpha = 1;
  ctx.font = "12px var(--font-body, system-ui)";
  coords.forEach((c, i) => {
    const x = sx(c[0]), y = sy(c[1]);
    ctx.fillStyle = colorOf(labels[i]);
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = color;
    ctx.fillText(labels[i], x + 10, y + 4);
  });
}

/** WavLM widget styles, injected once per page (keeps us on the shared design system). */
export const WAVLM_CSS = `
.wave-wrap { margin:.6rem 0; }
.wave { inline-size:100%; block-size:96px; display:block; background:var(--bg-raised);
  border:1px solid var(--border); border-radius:var(--radius); touch-action:none; }
.audio-row { display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; margin:.5rem 0; }
.audio-row audio { block-size:34px; max-inline-size:100%; }
.chip { font:inherit; font-size:.82rem; padding:.35rem .7rem; border-radius:999px; min-block-size:2.4rem;
  border:1px solid var(--border); background:var(--bg-raised); color:var(--color); cursor:pointer; }
.chip:hover { border-color:var(--accent); }
.chip[aria-pressed="true"] { border-color:var(--accent); background:color-mix(in srgb,var(--accent) 16%,transparent); }
.sample-row { display:flex; flex-wrap:wrap; gap:.4rem; margin:.4rem 0; align-items:center; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem;
  color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.fallback { border:1px solid var(--warn); border-radius:var(--radius); background:var(--bg-raised); padding:1rem; }
.heat { inline-size:100%; block-size:220px; display:block; background:var(--bg-raised);
  border:1px solid var(--border); border-radius:var(--radius); image-rendering:pixelated; }
.heat-axes { display:flex; justify-content:space-between; font-size:.72rem; color:var(--muted);
  font-family:var(--font-mono); margin-top:.2rem; }
.vecbar { display:flex; gap:1px; block-size:26px; margin:.4rem 0; border:1px solid var(--border);
  border-radius:6px; overflow:hidden; }
.vecbar i { flex:1 1 0; min-inline-size:0; }
.simline { display:flex; align-items:center; gap:.6rem; margin:.35rem 0; font-family:var(--font-mono); font-size:.82rem; }
.simline .lab { inline-size:8.5rem; flex:0 0 auto; color:var(--color); }
.simline .track { flex:1 1 0; min-inline-size:0; block-size:.85rem; border-radius:999px;
  background:var(--bg-secondary); overflow:hidden; }
.simline .fill { display:block; block-size:100%; background:var(--accent); }
.simline .num { inline-size:3.2rem; flex:0 0 auto; text-align:end; color:var(--muted); }
.pairgrid { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,180px),1fr)); gap:.8rem; margin:.6rem 0; }
.picker { border:1px solid var(--border); border-radius:var(--radius); padding:.7rem; background:var(--bg-raised); }
.picker h4 { margin:.1rem 0 .5rem; font-size:.85rem; }
.scatter { inline-size:100%; block-size:340px; max-inline-size:100%; display:block; background:var(--bg-raised);
  border:1px solid var(--border); border-radius:var(--radius); }
.sim-matrix { border-collapse:collapse; font-family:var(--font-mono); font-size:.76rem; inline-size:100%; }
.sim-matrix th, .sim-matrix td { border:1px solid var(--border); padding:.3rem .4rem; text-align:center; }
.sim-matrix th[scope="row"], .sim-matrix th[scope="col"] { color:var(--muted); font-weight:600; white-space:nowrap; }
.sim-matrix .mv { color:var(--color); }
.matrix-scroll { overflow-x:auto; }
.vh { position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); }
.curve { inline-size:100%; block-size:140px; display:block; background:var(--bg-raised);
  border:1px solid var(--border); border-radius:var(--radius); }
.legend { display:flex; flex-wrap:wrap; gap:.8rem; font-size:.76rem; color:var(--muted); margin:.5rem 0; }
.legend span { display:inline-flex; align-items:center; gap:.35rem; }
.dot { inline-size:.8em; block-size:.8em; border-radius:50%; display:inline-block; }
.transcript { font-family:var(--font-body); font-size:1.05rem; line-height:1.7; margin:.4rem 0;
  padding:.8rem; background:var(--bg-raised); border:1px solid var(--border); border-radius:var(--radius); }
`;

// A small, stable categorical palette for scatter/legend (indigo-anchored, works light+dark).
export const PALETTE = [
  "#4b3aff",
  "#e8590c",
  "#2b8a3e",
  "#c2255c",
  "#1098ad",
  "#9c36b5",
  "#f08c00",
];
