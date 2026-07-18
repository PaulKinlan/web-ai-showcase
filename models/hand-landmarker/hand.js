// Front-end helpers for the MediaPipe HandLandmarker pages. MediaPipe's vision tasks run on the main
// thread (they own their own WASM + optional GPU delegate), so these helpers wrap one task instance,
// switch it between IMAGE and VIDEO running modes on demand, and draw the 21 landmarks + hand skeleton
// onto a canvas. The model itself loads through lib/model-loader.js via lib/mediapipe.js createVisionTask.

export const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// The 21 hand landmarks and how they connect (MediaPipe's standard hand skeleton).
export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], // thumb
  [0, 5], [5, 6], [6, 7], [7, 8], // index
  [5, 9], [9, 10], [10, 11], [11, 12], // middle
  [9, 13], [13, 14], [14, 15], [15, 16], // ring
  [13, 17], [17, 18], [18, 19], [19, 20], // pinky
  [0, 17], // palm base
];

export const HAND_LANDMARK_NAMES = [
  "Wrist",
  "Thumb CMC", "Thumb MCP", "Thumb IP", "Thumb tip",
  "Index MCP", "Index PIP", "Index DIP", "Index tip",
  "Middle MCP", "Middle PIP", "Middle DIP", "Middle tip",
  "Ring MCP", "Ring PIP", "Ring DIP", "Ring tip",
  "Pinky MCP", "Pinky PIP", "Pinky DIP", "Pinky tip",
];

// One colour per detected hand, dark enough for white text / readable strokes in light and dark.
const HAND_COLORS = ["#1565c0", "#b3261e", "#1a7a3a", "#6a1b9a"];

/** Wraps a HandLandmarker task so a page can detect on a still image OR a video frame safely. */
export class HandTask {
  constructor(task) {
    this.task = task;
    this.mode = task?.runningMode || "IMAGE"; // createVisionTask defaults to IMAGE
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

/** Normalize the result shape across tasks-vision versions (handedness vs handednesses). */
function normalize(res) {
  return {
    landmarks: res.landmarks || [],
    worldLandmarks: res.worldLandmarks || [],
    handedness: res.handedness || res.handednesses || [],
  };
}

export function handColor(i) {
  return HAND_COLORS[i % HAND_COLORS.length];
}

/**
 * Draw an image/video frame + the detected hand skeletons onto `canvas` at the source's natural size.
 * Landmarks are normalized (0..1); we scale them to pixels. CSS scales the canvas responsively.
 */
export function drawHands(canvas, source, result, opts = {}) {
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

  const r = Math.max(3, w / 220);
  const lw = Math.max(2, w / 400);
  (result.landmarks || []).forEach((hand, hi) => {
    const color = handColor(hi);
    const pts = hand.map((p) => ({
      x: (opts.mirror ? 1 - p.x : p.x) * w,
      y: p.y * h,
    }));
    // connections
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.lineCap = "round";
    for (const [a, b] of HAND_CONNECTIONS) {
      if (!pts[a] || !pts[b]) continue;
      ctx.beginPath();
      ctx.moveTo(pts[a].x, pts[a].y);
      ctx.lineTo(pts[b].x, pts[b].y);
      ctx.stroke();
    }
    // points
    for (const p of pts) {
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

/** Distance between two normalized landmarks (0..1 space). Handy for pinch / gesture logic. */
export function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}

/** 2D distance in normalized space (ignores z) — for pinch detection on the image plane. */
export function dist2(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/**
 * Which of the five digits are extended, orientation-robustly: a finger is "up" when its tip is
 * farther from the wrist than its middle joint. Returns { fingers:[thumb,index,middle,ring,pinky], count }.
 */
export function fingersUp(landmarks) {
  const w = landmarks[0];
  const d = (i) => dist(landmarks[i], w);
  const tips = [4, 8, 12, 16, 20], pips = [3, 6, 10, 14, 18];
  const fingers = tips.map((t, i) => {
    if (i === 0) {
      // thumb: measure splay away from the index base (5) rather than curl toward the wrist
      return dist(landmarks[4], landmarks[5]) > dist(landmarks[2], landmarks[5]) * 1.35;
    }
    return d(t) > d(pips[i]) * 1.05;
  });
  return { fingers, count: fingers.filter(Boolean).length };
}

/** A friendly gesture name from the extended-finger pattern. */
export function gestureName({ fingers, count }) {
  const [thumb, index, middle, ring, pinky] = fingers;
  if (count === 0) return "Fist ✊";
  if (count === 5) return "Open palm ✋";
  if (index && middle && !ring && !pinky && !thumb) return "Peace ✌️";
  if (index && !middle && !ring && !pinky && !thumb) return "Pointing ☝️";
  if (thumb && !index && !middle && !ring && !pinky) return "Thumbs up 👍";
  if (thumb && index && !middle && !ring && pinky) return "Call me 🤙";
  if (!thumb && index && !middle && !ring && pinky) return "Rock 🤘";
  return `${count} finger${count === 1 ? "" : "s"}`;
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// Shared widget styles (reuse the design system). Canvas stage, dropzone, sample strip, tables.
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
