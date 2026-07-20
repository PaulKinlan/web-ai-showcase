// Front-end helpers for the Speaker-diarization pages: the worker handshake and the renderers (a
// colour-coded speaker timeline + a per-frame activity strip). Inference (pyannote segmentation +
// powerset decode → speaker segments) lives in worker.js, off the main thread. Model:
// onnx-community/pyannote-segmentation-3.0 (audio-frame-classification), WASM, ~6 MB.

const WORKER_URL = "/web-ai-showcase/models/speaker-diarization/worker.js";

// A small, WCAG-friendly categorical palette for speaker lanes (works in light + dark).
export const SPEAKER_COLORS = [
  "#4f46e5", // indigo
  "#0a7c66", // teal
  "#b45309", // amber-brown
  "#9a3b8e", // magenta
  "#2563eb", // blue
  "#7c3aed", // violet
];

export function speakerColor(id) {
  return SPEAKER_COLORS[id % SPEAKER_COLORS.length];
}

export class DiarizeEngine {
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

  load(onProgress) {
    if (onProgress) this.onProgress = onProgress;
    if (this.ready) return Promise.resolve(this.device);
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load" });
    });
  }

  /** Diarize a 16 kHz mono Float32Array. Returns { segments:[{id,start,end,confidence}], speakerIds,
   *  durationSec, frames, activity, speechFrames, overlapFrames, ms, device }. */
  diarize(audio, sampleRate = 16000) {
    const id = ++this._id;
    const copy = audio.slice();
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, audio: copy, sampleRate }, [copy.buffer]);
    });
  }
}

export function fmtTime(s) {
  s = Math.max(0, s);
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(1);
  return `${m}:${sec.padStart(4, "0")}`;
}

// Render a colour-coded speaker timeline: one lane per speaker, blocks positioned by time.
export function renderTimeline(el, segments, speakerIds, durationSec) {
  el.replaceChildren();
  if (!segments.length) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = "No speech segments detected.";
    el.append(p);
    return;
  }
  const dur = durationSec || Math.max(...segments.map((s) => s.end));
  for (const sid of speakerIds) {
    const lane = document.createElement("div");
    lane.className = "spk-lane";
    const label = document.createElement("span");
    label.className = "spk-label";
    label.textContent = "Speaker " + (sid + 1);
    label.style.color = speakerColor(sid);
    const track = document.createElement("div");
    track.className = "spk-track";
    for (const s of segments.filter((x) => x.id === sid)) {
      const block = document.createElement("div");
      block.className = "spk-block";
      block.style.insetInlineStart = (100 * s.start / dur).toFixed(2) + "%";
      block.style.inlineSize = Math.max(0.4, 100 * (s.end - s.start) / dur).toFixed(2) + "%";
      block.style.background = speakerColor(sid);
      block.style.opacity = String(0.35 + 0.6 * Math.min(1, s.confidence));
      block.title = `Speaker ${sid + 1}: ${fmtTime(s.start)}–${fmtTime(s.end)} (${
        (s.confidence * 100).toFixed(0)
      }%)`;
      track.append(block);
    }
    lane.append(label, track);
    el.append(lane);
  }
  // A time axis under the lanes.
  const axis = document.createElement("div");
  axis.className = "spk-axis";
  axis.innerHTML = `<span>0:00.0</span><span>${fmtTime(dur / 2)}</span><span>${
    fmtTime(dur)
  }</span>`;
  el.append(axis);
}

// Render the per-frame activity strip (silence / one speaker / overlap) for the "see inside" surface.
export function renderActivity(canvas, activity) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const n = activity.length || 1;
  const bw = W / n;
  for (let i = 0; i < activity.length; i++) {
    const a = activity[i];
    ctx.fillStyle = a >= 2 ? "#b45309" : a === 1 ? "#4f46e5" : "rgba(120,120,120,0.25)";
    ctx.fillRect(i * bw, 0, Math.ceil(bw), H);
  }
}

export const DIARIZE_CSS = `
.diar-controls { display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; margin:.6rem 0; }
.diar-controls button, .diar-controls label.upload { min-block-size:38px; }
.sample-strip { display:flex; gap:.4rem; flex-wrap:wrap; margin:.5rem 0; }
.chip { font:inherit; font-size:.8rem; padding:.3rem .7rem; border-radius:999px; min-block-size:34px;
  border:1px solid var(--border); background:var(--bg-raised); color:var(--color); cursor:pointer; }
.chip:hover { border-color:var(--accent); }
.chip:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.timeline { display:flex; flex-direction:column; gap:.4rem; margin-top:.6rem; }
.spk-lane { display:flex; align-items:center; gap:.6rem; }
.spk-label { flex:0 0 6.5rem; font-size:.8rem; font-weight:600; font-family:var(--font-mono); }
.spk-track { position:relative; flex:1 1 auto; block-size:22px; border-radius:5px;
  background:var(--bg-secondary); border:1px solid var(--border); overflow:hidden; min-inline-size:0; }
.spk-block { position:absolute; inset-block:2px; border-radius:3px; }
.spk-axis { display:flex; justify-content:space-between; font-family:var(--font-mono); font-size:.7rem;
  color:var(--muted); margin-inline-start:7.1rem; }
.act-canvas { inline-size:100%; block-size:26px; border:1px solid var(--border); border-radius:5px;
  background:var(--bg-raised); image-rendering:pixelated; }
.act-legend { display:flex; gap:1rem; flex-wrap:wrap; font-size:.75rem; color:var(--muted); margin-top:.35rem; }
.act-legend b { display:inline-block; inline-size:.8rem; block-size:.8rem; border-radius:2px; vertical-align:-1px; margin-inline-end:.3rem; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem;
  color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.seg-table { inline-size:100%; border-collapse:collapse; font-size:.8rem; margin-top:.5rem; }
.seg-table th, .seg-table td { text-align:left; padding:.25rem .5rem; border-bottom:1px solid var(--border); }
.seg-table th { color:var(--muted); font-weight:600; }
.seg-dot { display:inline-block; inline-size:.7rem; block-size:.7rem; border-radius:50%; margin-inline-end:.4rem; vertical-align:-1px; }
`;
