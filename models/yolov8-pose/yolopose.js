// Front-end helpers for the YOLOv8-pose pages. Keeps each page thin: it owns the worker handshake
// (transferring ImageBitmaps so nothing is copied), turns files/samples/video frames into ImageBitmaps,
// and draws the 17-point COCO skeleton(s) onto a canvas. ALL inference AND letterbox preprocessing
// happen off the main thread in worker.js (raw ONNX Runtime Web). Privacy by construction: the image
// and every keypoint it produces never leave the device.
//
// Model: Xenova/yolov8n-pose (onnx/model.onnx, fp32, ~13 MB, AGPL-3.0). YOLOv8-pose is a SINGLE-STAGE
// detector+pose head: one forward pass returns a box + 17 COCO keypoints for EVERY person, so it poses
// a whole crowd at once — distinct from the top-down ViTPose demo (one crop per person) and MediaPipe.

const WORKER_URL = "/web-ai-showcase/models/yolov8-pose/worker.js";

// The 17 COCO keypoints, in the model's output order.
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
  [5, 6],
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
  [5, 11],
  [6, 12],
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 4],
  [0, 5],
  [0, 6],
];

const PERSON_COLORS = [
  "#1565c0",
  "#b3261e",
  "#1a7a3a",
  "#6a1b9a",
  "#b06a00",
  "#00838f",
  "#c2185b",
  "#5d4037",
];
export function personColor(i) {
  return PERSON_COLORS[i % PERSON_COLORS.length];
}

export class PoseEngine {
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

  /** Estimate every pose in an ImageBitmap (transferred → zero-copy). Returns
   * { persons:[{box:[x,y,w,h], score, keypoints:[[x,y]…17], scores:[…17]}], imageSize, gridConf,
   *   gridSize, candidates, ms, infMs, device }. */
  estimate(bitmap, opts) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, bitmap, opts }, [bitmap]);
    });
  }
}

/** Read a File into a data URL. */
export function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}
/** Decode a URL/dataURL into an <img>. */
export function decodeImage(url) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error("Could not decode that image."));
    im.src = url;
  });
}
export function toBitmap(source) {
  return createImageBitmap(source);
}
export async function urlToBitmap(src) {
  return createImageBitmap(await (await fetch(src)).blob());
}

/**
 * Draw a source image/frame + detected skeleton(s) onto `canvas` at the source's natural size.
 * `persons` = [{ keypoints:[[x,y]…17], scores:[…17], box }]. `minScore` hides low-confidence joints.
 * opts: { minScore, boxes (draw bbox), labels ([str] per person), mirror, dim, color }.
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
  } else if (source) {
    ctx.drawImage(source, 0, 0, w, h);
    if (opts.dim) {
      ctx.fillStyle = "rgba(0,0,0,0.45)";
      ctx.fillRect(0, 0, w, h);
    }
  }

  const r = Math.max(3, w / 220);
  const lw = Math.max(2.5, w / 320);
  const minScore = opts.minScore ?? 0.3;
  const mx = (x) => (opts.mirror ? w - x : x);
  (persons || []).forEach((person, pi) => {
    const color = opts.color || personColor(pi);
    const pts = person.keypoints.map((kp, i) => ({
      x: mx(kp[0]),
      y: kp[1],
      s: person.scores?.[i] ?? 1,
    }));
    if (opts.boxes && person.box) {
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.setLineDash([lw * 3, lw * 2]);
      const [bx, by, bw, bh] = person.box;
      ctx.strokeRect(opts.mirror ? w - bx - bw : bx, by, bw, bh);
      ctx.setLineDash([]);
    }
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
    const label = opts.labels?.[pi];
    if (label && person.box) {
      const [bx, by] = person.box;
      const lx = opts.mirror ? w - bx - person.box[2] : bx;
      ctx.font = `600 ${Math.max(12, w / 45)}px system-ui, sans-serif`;
      const tw = ctx.measureText(label).width;
      const fh = Math.max(16, w / 34);
      ctx.fillStyle = color;
      ctx.fillRect(lx, Math.max(0, by - fh), tw + 10, fh);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, lx + 5, Math.max(fh - 5, by - 5));
    }
  });
  return persons;
}

/** Render the objectness grid (80×80 from the largest anchor stride) into `canvas` as a heatmap — this
 * is literally where YOLOv8 detects person-ness before NMS. Returns { max }. */
export function drawConfGrid(canvas, grid, size) {
  let max = 0;
  for (let i = 0; i < grid.length; i++) if (grid[i] > max) max = grid[i];
  const scale = 4;
  canvas.width = size * scale;
  canvas.height = size * scale;
  const ctx = canvas.getContext("2d");
  const denom = max || 1;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const t = Math.min(1, grid[y * size + x] / denom);
      const rr = Math.round(30 + t * 225),
        gg = Math.round(20 + t * 150),
        bb = Math.round(90 - t * 70);
      ctx.fillStyle = `rgb(${rr},${gg},${bb})`;
      ctx.fillRect(x * scale, y * scale, scale, scale);
    }
  }
  return { max };
}

// Simple, honest joint-angle helpers (degrees) shared by the practical/wild pages. Rule-based, on
// top of the model's real keypoints — NOT a second model.
export function angleAt(a, b, c) {
  // angle ABC in degrees, or null if any joint missing
  if (!a || !b || !c) return null;
  const v1x = a[0] - b[0], v1y = a[1] - b[1];
  const v2x = c[0] - b[0], v2y = c[1] - b[1];
  const dot = v1x * v2x + v1y * v2y;
  const m1 = Math.hypot(v1x, v1y), m2 = Math.hypot(v2x, v2y);
  if (m1 < 1e-3 || m2 < 1e-3) return null;
  return Math.acos(Math.max(-1, Math.min(1, dot / (m1 * m2)))) * 180 / Math.PI;
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// Shared inline styles for the YOLOv8-pose widgets (design-system tokens only → light/dark for free).
export const POSE_CSS = `
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
.conf-bar { display:inline-block; block-size:.55rem; border-radius:999px; background:var(--accent);
  vertical-align:middle; margin-inline-end:.4rem; }
.metric-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.6rem; margin:.6rem 0; }
.metric { border:1px solid var(--border); border-radius:8px; padding:.6rem .7rem; background:var(--bg-raised); }
.metric .k { font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.03em; }
.metric .v { font-family:var(--font-mono); font-size:1.15rem; font-weight:600; }
.feedback { border:1px solid var(--border); border-radius:8px; padding:.7rem .8rem; margin:.5rem 0; background:var(--bg-raised); }
.feedback.good { border-color:var(--ok,#1a7a3a); }
.feedback.warn { border-color:var(--warn,#b06a00); }
`;
