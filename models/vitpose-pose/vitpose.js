// Front-end helpers for the ViTPose pages. Keeps each page thin: it owns the worker handshake, turns
// files/samples into data URLs, decodes images, and draws the 17-point COCO skeleton onto a canvas.
// ALL inference happens off the main thread in worker.js (a transformer top-down pose estimator).
//
// Model: onnx-community/vitpose-base-simple (VitPoseForPoseEstimation), WASM backend, q8.
// This is a DIFFERENT approach from the MediaPipe PoseLandmarker: ViTPose is a top-down heatmap
// transformer — you feed it a person box, it emits one 64x48 heatmap per keypoint, and the keypoint is
// the arg-max of that heatmap. We expose those raw heatmaps for the "see inside" surface.

const WORKER_URL = "/web-ai-showcase/models/vitpose-pose/worker.js";

// The 17 COCO keypoints, in the model's id2label order.
export const COCO_KEYPOINTS = [
  "Nose",
  "Left eye",
  "Right eye",
  "Left ear",
  "Right ear",
  "Left shoulder",
  "Right shoulder",
  "Left elbow",
  "Right elbow",
  "Left wrist",
  "Right wrist",
  "Left hip",
  "Right hip",
  "Left knee",
  "Right knee",
  "Left ankle",
  "Right ankle",
];

// COCO skeleton: how the 17 keypoints connect into limbs (0-indexed).
export const COCO_SKELETON = [
  [5, 6], // shoulders
  [5, 7],
  [7, 9], // left arm
  [6, 8],
  [8, 10], // right arm
  [5, 11],
  [6, 12],
  [11, 12], // torso
  [11, 13],
  [13, 15], // left leg
  [12, 14],
  [14, 16], // right leg
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 4], // face
  [0, 5],
  [0, 6], // neck-ish
];

// Distinct colours per detected person (multi-person / multi-model pages).
const PERSON_COLORS = ["#1565c0", "#b3261e", "#1a7a3a", "#6a1b9a", "#b06a00"];
export function personColor(i) {
  return PERSON_COLORS[i % PERSON_COLORS.length];
}

export class VitPoseEngine {
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

  /**
   * Estimate pose(s). `boxes` = array of [x, y, w, h] person boxes in image-pixel coords; omit to run
   * one full-image box. Returns { persons:[{keypoints:[[x,y]…], scores:[…], bbox}], heatmapDims,
   * heatmaps (Float32Array for person 0, 17×H×W), ms, device }.
   */
  estimate(imageURL, boxes) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, image: imageURL, boxes });
    });
  }
}

/** Read a File into a data URL usable by the worker. */
export function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}

/** Decode a URL/dataURL into an <img> (natural size known). */
export function decodeImage(url) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error("Could not decode that image."));
    im.src = url;
  });
}

/**
 * Draw an image/video frame + detected skeleton(s) onto `canvas` at the source's natural size.
 * `persons` = [{ keypoints:[[x,y]…17], scores:[…17] }]. `minScore` hides low-confidence joints.
 */
export function drawPoses(canvas, source, persons, opts = {}) {
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
  } else if (opts.dim) {
    ctx.drawImage(source, 0, 0, w, h);
    ctx.fillStyle = "rgba(0,0,0,0.45)";
    ctx.fillRect(0, 0, w, h);
  } else if (source) {
    ctx.drawImage(source, 0, 0, w, h);
  }

  const r = Math.max(3, w / 220);
  const lw = Math.max(2.5, w / 320);
  const minScore = opts.minScore ?? 0.3;
  (persons || []).forEach((person, pi) => {
    const color = opts.color || personColor(pi);
    const pts = person.keypoints.map((kp, i) => ({
      x: opts.mirror ? w - kp[0] : kp[0],
      y: kp[1],
      s: person.scores?.[i] ?? 1,
    }));
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.lineCap = "round";
    for (const [a, b] of COCO_SKELETON) {
      if (pts[a].s < minScore || pts[b].s < minScore) continue;
      ctx.beginPath();
      ctx.moveTo(pts[a].x, pts[a].y);
      ctx.lineTo(pts[b].x, pts[b].y);
      ctx.stroke();
    }
    for (const p of pts) {
      if (p.s < minScore) continue;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fillStyle = "#fff";
      ctx.fill();
      ctx.lineWidth = lw;
      ctx.strokeStyle = color;
      ctx.stroke();
    }
  });
  return persons;
}

/**
 * Render one keypoint's raw heatmap (H×W of a Float32Array laid out 17×H×W) into `canvas`, marking the
 * arg-max. This is the literal thing ViTPose decodes: the joint is the hottest cell of its heatmap.
 * Returns { argX, argY, max } in heatmap-cell coords.
 */
export function drawHeatmap(canvas, heatmaps, dims, kIndex) {
  const [, H, W] = dims; // dims = [17, H, W]
  const off = kIndex * H * W;
  let max = -Infinity, min = Infinity, argX = 0, argY = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const v = heatmaps[off + y * W + x];
      if (v > max) {
        max = v;
        argX = x;
        argY = y;
      }
      if (v < min) min = v;
    }
  }
  const scale = 6; // upscale each cell for legibility
  canvas.width = W * scale;
  canvas.height = H * scale;
  const ctx = canvas.getContext("2d");
  const range = max - min || 1;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const t = (heatmaps[off + y * W + x] - min) / range; // 0..1
      // Warm ramp: dark indigo -> hot amber for the peak.
      const rr = Math.round(30 + t * 225);
      const gg = Math.round(20 + t * 150);
      const bb = Math.round(90 - t * 70);
      ctx.fillStyle = `rgb(${rr},${gg},${bb})`;
      ctx.fillRect(x * scale, y * scale, scale, scale);
    }
  }
  // Mark the arg-max (the decoded joint location).
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(argX * scale + scale / 2, argY * scale + scale / 2, scale * 1.6, 0, Math.PI * 2);
  ctx.stroke();
  return { argX, argY, max };
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// Shared inline styles for the ViTPose widgets (injected once per page). Uses the design-system tokens.
export const VITPOSE_CSS = `
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
.field-row { display:flex; flex-wrap:wrap; gap:.6rem; align-items:center; margin:.6rem 0; }
.field-row label { display:flex; flex-direction:column; gap:.25rem; font-size:.82rem; }
.slider-row { display:flex; align-items:center; gap:.6rem; margin:.6rem 0; flex-wrap:wrap; }
.slider-row input[type=range] { flex:1 1 180px; accent-color:var(--accent); min-inline-size:0; }
.slider-row output { font-family:var(--font-mono); font-size:.82rem; min-inline-size:3ch; }
.hm-wrap { display:flex; gap:1rem; flex-wrap:wrap; align-items:flex-start; margin-top:.5rem; }
.hm-canvas { image-rendering:pixelated; border-radius:6px; border:1px solid var(--border);
  max-inline-size:100%; block-size:auto; }
.inside-table { inline-size:100%; border-collapse:collapse; font-size:.82rem; margin-top:.5rem; }
.inside-table th, .inside-table td { text-align:left; padding:.3rem .5rem; border-bottom:1px solid var(--border);
  font-family:var(--font-mono); }
.inside-table th { color:var(--muted); font-weight:600; }
.conf-cell { position:relative; }
.conf-bar { display:inline-block; block-size:.55rem; border-radius:999px; background:var(--accent);
  vertical-align:middle; margin-inline-end:.4rem; }
.rec-dot { inline-size:.7rem; block-size:.7rem; border-radius:50%; background:var(--bad,#c0392b);
  display:inline-block; margin-inline-end:.4rem; animation:recpulse 1s ease-in-out infinite; }
@keyframes recpulse { 50% { opacity:.25; } }
@media (prefers-reduced-motion: reduce) { .rec-dot { animation:none; } }
.metric-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.6rem; margin:.6rem 0; }
.metric { border:1px solid var(--border); border-radius:8px; padding:.6rem .7rem; background:var(--bg-raised); }
.metric .k { font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.03em; }
.metric .v { font-family:var(--font-mono); font-size:1.15rem; font-weight:600; }
`;
