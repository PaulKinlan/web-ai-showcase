// Front-end helpers for the MediaPipe PoseLandmarker pages. Draws the 33-point body skeleton onto a
// canvas, and — the off-main-thread path (invariant 15) — provides createMediaPipeWorkerTask(): a
// main-thread client that loads the MediaPipe task inside models/pose-landmarker/worker.js and runs every
// detect there, so the heavy inference never blocks the UI. The model still loads through
// lib/model-loader.js createModelLoader (auto-init/explicit-download policy unchanged); only WHERE the
// MediaPipe task lives moved (main thread → dedicated worker). The worker is CLASSIC (see worker.js /
// module:false below) because MediaPipe's WASM loader needs importScripts, which module workers forbid.

import { WorkerClient } from "/web-ai-showcase/lib/worker-protocol.js";

export const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

// MediaPipe's 33 body landmarks and how they connect into a skeleton.
export const POSE_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 7],
  [0, 4],
  [4, 5],
  [5, 6],
  [6, 8],
  [9, 10],
  [11, 12],
  [11, 13],
  [13, 15],
  [15, 17],
  [15, 19],
  [15, 21],
  [17, 19],
  [12, 14],
  [14, 16],
  [16, 18],
  [16, 20],
  [16, 22],
  [18, 20],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [24, 26],
  [25, 27],
  [26, 28],
  [27, 29],
  [28, 30],
  [29, 31],
  [30, 32],
  [27, 31],
  [28, 32],
];

export const POSE_LANDMARK_NAMES = [
  "Nose",
  "Left eye (inner)",
  "Left eye",
  "Left eye (outer)",
  "Right eye (inner)",
  "Right eye",
  "Right eye (outer)",
  "Left ear",
  "Right ear",
  "Mouth (left)",
  "Mouth (right)",
  "Left shoulder",
  "Right shoulder",
  "Left elbow",
  "Right elbow",
  "Left wrist",
  "Right wrist",
  "Left pinky",
  "Right pinky",
  "Left index",
  "Right index",
  "Left thumb",
  "Right thumb",
  "Left hip",
  "Right hip",
  "Left knee",
  "Right knee",
  "Left ankle",
  "Right ankle",
  "Left heel",
  "Right heel",
  "Left foot index",
  "Right foot index",
];

const POSE_COLORS = ["#1565c0", "#b3261e", "#1a7a3a", "#6a1b9a"];

export class PoseTask {
  constructor(task) {
    this.task = task;
    this.mode = task?.runningMode || "IMAGE";
  }
  async _ensure(mode) {
    if (this.mode !== mode) {
      await this.task.setOptions({ runningMode: mode });
      this.mode = mode;
    }
  }
  async detectImage(imgEl) {
    await this._ensure("IMAGE");
    return normalize(this.task.detect(imgEl));
  }
  async detectVideo(videoEl, tsMs) {
    await this._ensure("VIDEO");
    return normalize(this.task.detectForVideo(videoEl, tsMs));
  }
  get delegate() {
    return this.task?.__delegate || "CPU";
  }
}

function normalize(res) {
  return { landmarks: res.landmarks || [], worldLandmarks: res.worldLandmarks || [] };
}

// ── Off-main-thread MediaPipe client ──────────────────────────────────────────────────────────────
// Loads a MediaPipe vision task inside the dedicated module worker (worker.js) and runs detect there.
// Mirrors the PoseTask/HandTask surface (detectImage / detectVideo / delegate) so the pages barely
// change: the client grabs a frame with createImageBitmap on the main thread (cheap, async, no pixel
// copy on the JS thread) and TRANSFERS it to the worker; the worker returns only the tiny landmark
// arrays. The webcam loop must keep at most one detect in flight (see the pages' rVFC loops); the "live"
// channel makes any stray extra frame supersede the older one rather than pile up (backpressure).
//
// @param {object} o
// @param {string} o.taskClass  e.g. "PoseLandmarker" | "HandLandmarker"
// @param {string} o.modelUrl   the canonical .task asset URL
// @param {object} [o.options]  { runningMode, taskOptions } — taskOptions is passed to createFromOptions
// @param {(p:any)=>void} [o.onProgress]
// @returns {Promise<MediaPipeWorkerTask>}
export async function createMediaPipeWorkerTask({ taskClass, modelUrl, options = {}, onProgress }) {
  const client = new WorkerClient({
    url: new URL("./worker.js", import.meta.url),
    name: taskClass.toLowerCase(),
    // CLASSIC worker (not module): MediaPipe's Emscripten WASM loader calls importScripts(), which only
    // exists in classic workers — a module worker throws "Module scripts don't support importScripts()".
    // worker.js dynamically import()s the ESM protocol + MediaPipe, both allowed in a classic worker.
    module: false,
    maxInFlight: 1,
    maxQueue: 1,
  });
  await client.ready;
  const { result } = await client.request(
    "load",
    {
      taskClass,
      modelUrl,
      options,
      preferGpu: typeof navigator !== "undefined" && "gpu" in navigator,
    },
    { onProgress },
  );
  return new MediaPipeWorkerTask(client, result?.delegate || "CPU", options.runningMode || "IMAGE");
}

export class MediaPipeWorkerTask {
  constructor(client, delegate, mode) {
    this.client = client;
    this._delegate = delegate;
    this.mode = mode;
  }
  get delegate() {
    return this._delegate;
  }
  async _detect(source, mode, timestamp, channel) {
    // Grab the current frame as an ImageBitmap and transfer ownership to the worker (no pixel copy on
    // the main thread). MediaPipe accepts an ImageBitmap as an image source.
    const bitmap = await createImageBitmap(source);
    try {
      const { result } = await this.client.request(
        "detect",
        { bitmap, mode, timestamp },
        { transfer: [bitmap], ...(channel ? { channel } : {}) },
      );
      if (result?.delegate) this._delegate = result.delegate;
      return { landmarks: result.landmarks || [], worldLandmarks: result.worldLandmarks || [] };
    } catch (err) {
      // If the request never dispatched (superseded/overflow) the bitmap wasn't transferred — release it.
      try {
        bitmap.close?.();
      } catch { /* already neutered by a successful transfer */ }
      throw err;
    }
  }
  detectImage(source) {
    return this._detect(source, "IMAGE", undefined, null);
  }
  detectVideo(source, tsMs) {
    return this._detect(source, "VIDEO", tsMs, "live");
  }
  terminate() {
    return this.client.terminate();
  }
}

export function poseColor(i) {
  return POSE_COLORS[i % POSE_COLORS.length];
}

/** Draw an image/video frame + detected body skeleton(s) on `canvas` at the source's natural size. */
export function drawPoses(canvas, source, result, opts = {}) {
  const w = source.naturalWidth || source.videoWidth || source.width;
  const h = source.naturalHeight || source.videoHeight || source.height;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (opts.mirror) {
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(source, 0, 0, w, h);
    ctx.restore();
  } else {
    ctx.drawImage(source, 0, 0, w, h);
  }

  const r = Math.max(3, w / 200);
  const lw = Math.max(2.5, w / 300);
  (result.landmarks || []).forEach((pose, pi) => {
    const color = opts.color || poseColor(pi);
    const minVis = opts.minVisibility ?? 0.5;
    const pts = pose.map((p) => ({
      x: (opts.mirror ? 1 - p.x : p.x) * w,
      y: p.y * h,
      v: p.visibility ?? 1,
    }));
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.lineCap = "round";
    for (const [a, b] of POSE_CONNECTIONS) {
      if (!pts[a] || !pts[b]) continue;
      if (pts[a].v < minVis || pts[b].v < minVis) continue;
      ctx.beginPath();
      ctx.moveTo(pts[a].x, pts[a].y);
      ctx.lineTo(pts[b].x, pts[b].y);
      ctx.stroke();
    }
    for (const p of pts) {
      if (p.v < minVis) continue;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fillStyle = "#fff";
      ctx.fill();
      ctx.lineWidth = lw;
      ctx.strokeStyle = color;
      ctx.stroke();
    }
  });
  return result;
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

export const LANDMARK_CSS = `
.dropzone { border:2px dashed var(--border-strong); border-radius:var(--radius); background:var(--bg-raised);
  padding:1rem; text-align:center; cursor:pointer; transition:border-color .15s, background .15s; }
.dropzone.drag { border-color:var(--accent); background:var(--bg-secondary); }
.dropzone:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.sample-strip { display:flex; gap:.5rem; flex-wrap:wrap; margin:.5rem 0; }
.sample-thumb { inline-size:76px; block-size:56px; object-fit:cover; border-radius:6px; border:2px solid transparent;
  cursor:pointer; padding:0; background:var(--bg-raised); }
.sample-thumb.active { border-color:var(--accent); }
.sample-thumb:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.canvas-wrap { position:relative; display:block; margin-top:.5rem; border-radius:8px; overflow:hidden;
  background:var(--bg-raised); border:1px solid var(--border); }
.stage-canvas { display:block; inline-size:100%; block-size:auto; max-block-size:64vh; object-fit:contain; }
.stage-canvas:focus-visible { outline:3px solid var(--accent); outline-offset:-3px; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem;
  color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.chip { font:inherit; font-size:.82rem; padding:.2rem .6rem; border-radius:999px; border:1px solid var(--border);
  background:var(--bg-raised); color:var(--color); cursor:pointer; }
.chip:hover { border-color:var(--accent); }
.chip[aria-pressed=true] { border-color:var(--accent); background:var(--bg-secondary); }
.field-row { display:flex; flex-wrap:wrap; gap:.6rem; align-items:center; margin:.6rem 0; }
.count-chips { display:flex; flex-wrap:wrap; gap:.4rem; margin:.5rem 0; }
.count-chip { display:inline-flex; align-items:center; gap:.35rem; font-size:.82rem; padding:.15rem .55rem;
  border-radius:999px; border:1px solid var(--border); background:var(--bg-raised); }
.count-chip .swatch { inline-size:.7rem; block-size:.7rem; border-radius:3px; }
.slider-row { display:flex; align-items:center; gap:.6rem; margin:.6rem 0; flex-wrap:wrap; }
.slider-row input[type=range] { flex:1 1 180px; accent-color:var(--accent); }
.slider-row output { font-family:var(--font-mono); font-size:.82rem; min-inline-size:3ch; }
.inside-table { inline-size:100%; border-collapse:collapse; font-size:.82rem; margin-top:.5rem; }
.inside-table th, .inside-table td { text-align:left; padding:.3rem .5rem; border-bottom:1px solid var(--border);
  font-family:var(--font-mono); }
.inside-table th { color:var(--muted); font-weight:600; }
.rec-dot { inline-size:.7rem; block-size:.7rem; border-radius:50%; background:var(--bad,#c0392b);
  display:inline-block; margin-inline-end:.4rem; animation:recpulse 1s ease-in-out infinite; }
@keyframes recpulse { 50% { opacity:.25; } }
@media (prefers-reduced-motion: reduce) { .rec-dot { animation:none; } }
.fallback { border:1px solid var(--warn); border-radius:var(--radius); background:var(--bg-raised); padding:1rem; }
.fallback code { background:var(--bg-secondary); padding:.05rem .3rem; border-radius:4px; }
`;
