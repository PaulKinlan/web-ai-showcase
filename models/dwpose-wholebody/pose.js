// DWPose whole-body engine + helpers. The engine talks to worker.js (raw ORT-web); the helpers crop a
// person box to the model's top-down input, draw the 133-keypoint skeleton (body + feet + 68 face + 42
// hand points), and expose the crop transform. No fake output — every point is the model's decoded keypoint.

export const IN_W = 288;
export const IN_H = 384;

export class PoseEngine {
  constructor() {
    this.worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    this.onProgress = null;
    this.pending = new Map();
    this.seq = 0;
    this.worker.addEventListener("message", (e) => {
      const m = e.data;
      if (m.type === "progress") this.onProgress?.(m.p);
      else if (m.type === "ready") this._ready?.();
      else if (m.type === "pose") this.pending.get(m.id)?.resolve(m);
      else if (m.type === "error") {
        if (m.id != null && this.pending.has(m.id)) {
          this.pending.get(m.id).reject(new Error(m.message));
        } else this._readyReject?.(new Error(m.message));
      }
    });
  }
  load(onProgress) {
    if (onProgress) this.onProgress = onProgress;
    return new Promise((resolve, reject) => {
      this._ready = resolve;
      this._readyReject = reject;
      this.worker.postMessage({ type: "load" });
    });
  }
  /** Estimate whole-body keypoints from a person crop (RGBA ImageData at IN_W×IN_H) → { kpts, w, h, ms }. */
  estimate(crop) {
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      const c = new Uint8ClampedArray(crop.data);
      this.worker.postMessage({ type: "estimate", id, crop: c }, [c.buffer]);
    }).finally(() => this.pending.delete(id));
  }
  dispose() {
    this.worker.terminate();
  }
}

/**
 * Crop a person box [x,y,w,h] (source pixels) from a bitmap to the model's top-down input. The box is
 * expanded to the model's W/H aspect (0.75) so the whole body fits, then drawn to IN_W×IN_H → ImageData.
 */
export function cropPerson(bitmap, [x, y, w, h]) {
  const AR = IN_W / IN_H;
  const cx = x + w / 2, cy = y + h / 2;
  if (w / h < AR) w = h * AR;
  else h = w / AR;
  x = cx - w / 2;
  y = cy - h / 2;
  const c = new OffscreenCanvas(IN_W, IN_H);
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, x, y, w, h, 0, 0, IN_W, IN_H);
  return ctx.getImageData(0, 0, IN_W, IN_H);
}

export function putImageData(canvas, imageData) {
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  canvas.getContext("2d").putImageData(imageData, 0, 0);
}

export async function urlToBitmap(src) {
  return createImageBitmap(await (await fetch(src)).blob());
}

// COCO-wholebody-133 layout: 0-16 body, 17-22 feet, 23-90 face (68), 91-111 left hand, 112-132 right hand.
const BODY_EDGES = [
  [15, 13],
  [13, 11],
  [16, 14],
  [14, 12],
  [11, 12],
  [5, 11],
  [6, 12],
  [5, 6],
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
  [1, 2],
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 4],
];
const FEET_EDGES = [[15, 17], [15, 18], [15, 19], [16, 20], [16, 21], [16, 22]];
const HAND_LOCAL = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [0, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [0, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
];
const FACE = { start: 23, count: 68 };
const LHAND = 91, RHAND = 112;

/**
 * Draw the whole-body skeleton onto a canvas already showing the crop. Keypoints are in input (IN_W×IN_H)
 * space; they scale to the canvas' displayed size. `parts` toggles face + hands so the "whole-body" vs
 * "body-only" difference is visible. Only keypoints above `threshold` confidence are drawn.
 */
export function drawSkeleton(canvas, kpts, { threshold = 0.35, face = true, hands = true } = {}) {
  const ctx = canvas.getContext("2d");
  const sx = canvas.width / IN_W, sy = canvas.height / IN_H;
  const P = (i) => kpts[i];
  const line = (a, b, color, lw = 2) => {
    const pa = P(a), pb = P(b);
    if (!pa || !pb || pa.c < threshold || pb.c < threshold) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.beginPath();
    ctx.moveTo(pa.x * sx, pa.y * sy);
    ctx.lineTo(pb.x * sx, pb.y * sy);
    ctx.stroke();
  };
  const dot = (i, color, r = 2.2) => {
    const p = P(i);
    if (!p || p.c < threshold) return;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(p.x * sx, p.y * sy, r, 0, 7);
    ctx.fill();
  };
  // body + feet
  for (const [a, b] of BODY_EDGES) line(a, b, "#2bb59a", 3);
  for (const [a, b] of FEET_EDGES) line(a, b, "#2bb59a", 2);
  for (let i = 0; i <= 22; i++) dot(i, "#7fe8d2", 2.4);
  // face
  if (face) { for (let i = 0; i < FACE.count; i++) dot(FACE.start + i, "#f5b642", 1.3); }
  // hands
  if (hands) {
    for (const base of [LHAND, RHAND]) {
      const color = base === LHAND ? "#e05be0" : "#4ac6e0";
      for (const [a, b] of HAND_LOCAL) line(base + a, base + b, color, 1.4);
      for (let i = 0; i < 21; i++) dot(base + i, color, 1.4);
    }
  }
}

export function countConfident(kpts, threshold = 0.35) {
  return kpts.reduce((n, p) => n + (p.c >= threshold ? 1 : 0), 0);
}

export const POSE_CSS = `
  .dw-gallery { display: flex; flex-wrap: wrap; gap: 0.8rem; margin: 0.8rem 0; }
  .dw-person { text-align: center; }
  .dw-person canvas { width: 150px; height: 200px; border-radius: 10px; background: #0b0f14; display: block; }
  .dw-person figcaption { font-size: 0.76rem; margin-top: 0.25rem; }
  .dw-person .k { font-family: var(--font-mono, monospace); }
  .dw-controls { display: flex; flex-wrap: wrap; gap: 0.8rem 1.4rem; align-items: center; margin: 0.5rem 0; font-size: 0.9rem; }
  .dw-legend { display: flex; flex-wrap: wrap; gap: 0.4rem 1rem; font-size: 0.78rem; margin: 0.3rem 0; }
  .dw-legend span::before { content: ""; display: inline-block; width: 0.7rem; height: 0.7rem; border-radius: 2px; margin-right: 0.3rem; vertical-align: middle; }
  .dw-legend .body::before { background: #2bb59a; }
  .dw-legend .face::before { background: #f5b642; }
  .dw-legend .lh::before { background: #e05be0; }
  .dw-legend .rh::before { background: #4ac6e0; }
  .dw-status { font-family: var(--font-mono, monospace); font-size: 0.9rem; margin: 0.4rem 0; }
`;
