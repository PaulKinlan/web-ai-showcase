// Front-end helpers for the MediaPipe FaceLandmarker pages. Like hand.js / pose.js: wraps one task
// instance, switches IMAGE/VIDEO running mode on demand, and draws the 478-point face mesh (tesselation
// + feature contours + irises) onto a canvas. It also exposes the 52 blendshape expression coefficients
// and the 4×4 facial transformation matrix the model returns. The model loads through
// lib/model-loader.js via lib/mediapipe.js createVisionTask.

import { TASKS_VISION_VERSION } from "/web-ai-showcase/lib/mediapipe.js";

export const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}`;

// The canonical face-mesh connection sets ship as static arrays of {start,end} on the FaceLandmarker
// class. We pull them from the (already-cached) tasks-vision module rather than hand-transcribe 2.6k
// tesselation edges. Cached: createVisionTask imports the same URL, so this resolves from module cache.
let connPromise = null;
export function faceConnections() {
  connPromise ??= import(CDN).then((v) => {
    const F = v.FaceLandmarker;
    return {
      tesselation: F.FACE_LANDMARKS_TESSELATION,
      faceOval: F.FACE_LANDMARKS_FACE_OVAL,
      leftEye: F.FACE_LANDMARKS_LEFT_EYE,
      rightEye: F.FACE_LANDMARKS_RIGHT_EYE,
      leftBrow: F.FACE_LANDMARKS_LEFT_EYEBROW,
      rightBrow: F.FACE_LANDMARKS_RIGHT_EYEBROW,
      lips: F.FACE_LANDMARKS_LIPS,
      leftIris: F.FACE_LANDMARKS_LEFT_IRIS,
      rightIris: F.FACE_LANDMARKS_RIGHT_IRIS,
    };
  });
  return connPromise;
}

// A few named landmarks worth reading out (nose tip, eye corners, lip centres, chin).
export const KEY_LANDMARKS = [
  [1, "Nose tip"],
  [33, "Left eye (outer)"],
  [133, "Left eye (inner)"],
  [263, "Right eye (outer)"],
  [362, "Right eye (inner)"],
  [61, "Mouth (left)"],
  [291, "Mouth (right)"],
  [13, "Upper lip"],
  [14, "Lower lip"],
  [199, "Chin"],
  [10, "Forehead"],
  [468, "Left iris (centre)"],
  [473, "Right iris (centre)"],
];

/** Wraps a FaceLandmarker task so a page can detect on a still image OR a video frame safely. */
export class FaceTask {
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
  return {
    landmarks: res.faceLandmarks || [],
    blendshapes: res.faceBlendshapes || [],
    matrixes: res.facialTransformationMatrixes || [],
  };
}

/** Turn a blendshapes result entry into a plain {name: score} map (drops the leading "_neutral"). */
export function blendMap(blend) {
  const out = {};
  for (const c of blend?.categories || []) out[c.categoryName] = c.score;
  return out;
}

/** Left+right average of a paired blendshape (e.g. "eyeBlink" → mean of eyeBlinkLeft/Right). */
export function pair(map, base) {
  const l = map[base + "Left"] ?? 0, r = map[base + "Right"] ?? 0;
  return (l + r) / 2;
}

const ACCENT = "#6a1b9a",
  MESH = "rgba(120,130,160,0.55)",
  EYE = "#1565c0",
  LIP = "#b3261e",
  BROW = "#1a7a3a",
  IRIS = "#e8a300",
  OVAL = "#3949ab";

/**
 * Draw an image/video frame + the detected face mesh onto `canvas` at the source's natural size.
 * opts.mesh: "tesselation" (full mesh, good for stills) | "contours" (lighter, good for webcam).
 */
export function drawFaces(canvas, source, result, conns, opts = {}) {
  const w = source.naturalWidth || source.videoWidth || source.width;
  const h = source.naturalHeight || source.videoHeight || source.height;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const mirror = opts.mirror;
  if (mirror) {
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(source, 0, 0, w, h);
    ctx.restore();
  } else {
    ctx.drawImage(source, 0, 0, w, h);
  }
  if (!conns) return result;
  const X = (x) => (mirror ? 1 - x : x) * w, Y = (y) => y * h;
  const seg = (pts, list, color, lw) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.beginPath();
    for (const c of list) {
      const a = pts[c.start], b = pts[c.end];
      if (!a || !b) continue;
      ctx.moveTo(X(a.x), Y(a.y));
      ctx.lineTo(X(b.x), Y(b.y));
    }
    ctx.stroke();
  };
  const thin = Math.max(0.5, w / 900), mid = Math.max(1.2, w / 380);
  for (const pts of result.landmarks) {
    if ((opts.mesh ?? "tesselation") === "tesselation") {
      seg(pts, conns.tesselation, MESH, thin);
    }
    seg(pts, conns.faceOval, OVAL, mid);
    seg(pts, conns.leftBrow, BROW, mid);
    seg(pts, conns.rightBrow, BROW, mid);
    seg(pts, conns.leftEye, EYE, mid);
    seg(pts, conns.rightEye, EYE, mid);
    seg(pts, conns.lips, LIP, mid);
    seg(pts, conns.leftIris, IRIS, mid);
    seg(pts, conns.rightIris, IRIS, mid);
  }
  return result;
}

/**
 * Render a blendshapes bar panel into `el`. Shows the highest-scoring coefficients (drops "_neutral").
 * Reuses the same DOM nodes across calls so it's cheap enough for a live webcam loop.
 * @param {number} [top] how many bars to show (default: all 52).
 */
export function renderBlendPanel(el, blend, top = 52) {
  const cats = (blend?.categories || []).filter((c) => c.categoryName !== "_neutral");
  cats.sort((a, b) => b.score - a.score);
  const shown = cats.slice(0, top);
  if (el.childElementCount !== shown.length * 3) {
    el.replaceChildren();
    for (const c of shown) {
      const name = document.createElement("div");
      name.className = "bl-name";
      const track = document.createElement("div");
      track.className = "bl-track";
      const fill = document.createElement("div");
      fill.className = "bl-fill";
      track.append(fill);
      const val = document.createElement("div");
      val.className = "bl-val";
      el.append(name, track, val);
    }
  }
  shown.forEach((c, i) => {
    const name = el.children[i * 3],
      fill = el.children[i * 3 + 1].firstChild,
      val = el.children[i * 3 + 2];
    name.textContent = c.displayName || c.categoryName;
    fill.style.inlineSize = (c.score * 100).toFixed(0) + "%";
    val.textContent = c.score.toFixed(2);
  });
}

/** Approx head yaw/pitch/roll (degrees) from the 4×4 facial transformation matrix (column-major). */
export function headPose(matrix) {
  const m = matrix?.data;
  if (!m || m.length < 16) return null;
  // Rotation columns: r00=m[0], r10=m[1], r20=m[2] … (column-major 4×4).
  const r00 = m[0], r10 = m[1], r20 = m[2], r21 = m[6], r22 = m[10];
  const deg = 180 / Math.PI;
  const pitch = Math.atan2(-r20, Math.hypot(r21, r22)) * deg;
  const yaw = Math.atan2(r10, r00) * deg;
  const roll = Math.atan2(r21, r22) * deg;
  return { yaw, pitch, roll };
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// Shared widget styles (design-system variables only). Same base as hand.js/pose.js plus a blendshape
// bar panel and a simple SVG-avatar frame used by the wild demo.
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
.field-row { display:flex; flex-wrap:wrap; gap:.6rem; align-items:center; margin:.6rem 0; }
.count-chips { display:flex; flex-wrap:wrap; gap:.4rem; margin:.5rem 0; }
.count-chip { display:inline-flex; align-items:center; gap:.35rem; font-size:.82rem; padding:.15rem .55rem;
  border-radius:999px; border:1px solid var(--border); background:var(--bg-raised); }
.count-chip .swatch { inline-size:.7rem; block-size:.7rem; border-radius:3px; }
.inside-table { inline-size:100%; border-collapse:collapse; font-size:.82rem; margin-top:.5rem; }
.inside-table th, .inside-table td { text-align:left; padding:.3rem .5rem; border-bottom:1px solid var(--border);
  font-family:var(--font-mono); }
.inside-table th { color:var(--muted); font-weight:600; }
.rec-dot { inline-size:.7rem; block-size:.7rem; border-radius:50%; background:var(--bad,#c0392b);
  display:inline-block; margin-inline-end:.4rem; animation:recpulse 1s ease-in-out infinite; }
@keyframes recpulse { 50% { opacity:.25; } }
@media (prefers-reduced-motion: reduce) { .rec-dot { animation:none; } }
.blend-panel { display:grid; grid-template-columns:minmax(9rem,10rem) 1fr 3ch; gap:.25rem .6rem;
  align-items:center; margin-top:.6rem; font-size:.8rem; }
.blend-panel .bl-name { font-family:var(--font-mono); color:var(--muted); text-align:right;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.blend-panel .bl-track { block-size:.6rem; border-radius:999px; background:var(--bg-secondary);
  overflow:hidden; border:1px solid var(--border); }
.blend-panel .bl-fill { block-size:100%; border-radius:999px; background:var(--accent); inline-size:0;
  transition:inline-size .08s linear; }
@media (prefers-reduced-motion: reduce) { .blend-panel .bl-fill { transition:none; } }
.blend-panel .bl-val { font-family:var(--font-mono); color:var(--color); text-align:right; }
.avatar-frame { display:grid; place-items:center; padding:1rem; background:var(--bg-raised);
  border:1px solid var(--border); border-radius:var(--radius); }
.state-badge { display:inline-block; padding:.2rem .7rem; border-radius:999px; font-weight:600;
  font-size:.95rem; border:1px solid var(--border); }
.state-badge.ok { background:color-mix(in srgb, var(--accent) 15%, transparent); }
.state-badge.warn { background:color-mix(in srgb, #c0392b 18%, transparent); }
.fallback { border:1px solid var(--warn); border-radius:var(--radius); background:var(--bg-raised); padding:1rem; }
.fallback code { background:var(--bg-secondary); padding:.05rem .3rem; border-radius:4px; }
`;
