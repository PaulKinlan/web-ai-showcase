// Shared front-end helpers for the SlimSAM pages. Keeps each page thin: it owns the worker handshake
// (load → embed-once → segment-per-click), and the canvas helpers that turn a raw H×W mask into a
// coloured overlay or a transparent cut-out. All inference is in worker.js (off the main thread).

const WORKER_URL = "/web-ai-showcase/models/slimsam-segment-anything/worker.js";

export class SamEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this._loadWaiters = [];
    this._pending = new Map();
    this._id = 0;
    this.onProgress = null;
    this.device = "wasm";
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
    } else if (msg.type === "embedded" || msg.type === "result") {
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

  /** Compute + cache the vision-encoder embeddings for one image (data URL). Run once per image. */
  embed(imageURL) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "embed", id, image: imageURL });
    });
  }

  /** Decode a mask from point prompts: [{x,y,label}] with x,y normalized to [0,1], label 1=fg/0=bg. */
  segment(points) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "segment", id, points });
    });
  }
}

export function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

export function decodeImage(url) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error("Could not decode image"));
    im.src = url;
  });
}

// Indigo overlay by default — matches the design accent and passes over most photos. The coloured
// fill + crisp edge are baked in worker.js (off the main thread) and arrive as an ImageBitmap; this
// constant is kept only for parity/reference with the worker's MASK_RGB.
export const MASK_RGB = [75, 58, 255];

/**
 * Draw `imgEl` at natural resolution onto `canvas`, then blit the pre-composited overlay ImageBitmap
 * (coloured mask + crisp outline, built off the main thread in worker.js), plus a marker at each prompt
 * point. The dense per-pixel composite no longer runs here — the main thread only does drawImage(), so
 * click/keyboard redraws stay well under a frame. `overlay` is the worker's ImageBitmap (or null).
 */
export function drawMaskOverlay(canvas, imgEl, overlay, w, h, points = [], _opts = {}) {
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(imgEl, 0, 0, w, h);

  if (overlay) ctx.drawImage(overlay, 0, 0, w, h);

  for (const p of points) {
    const px = p.x * w, py = p.y * h;
    const rad = Math.max(5, w / 90);
    ctx.beginPath();
    ctx.arc(px, py, rad, 0, Math.PI * 2);
    ctx.fillStyle = p.label === 0 ? "#c0392b" : "#fff";
    ctx.fill();
    ctx.lineWidth = Math.max(2, w / 260);
    ctx.strokeStyle = p.label === 0 ? "#fff" : "#4b3aff";
    ctx.stroke();
  }
}

/** Return a canvas holding only the masked pixels of `imgEl` (transparent elsewhere) — a cut-out. */
export function cutout(imgEl, mask, w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  ctx.drawImage(imgEl, 0, 0, w, h);
  const img = ctx.getImageData(0, 0, w, h);
  const d = img.data;
  for (let i = 0; i < w * h; i++) if (!mask[i]) d[i * 4 + 3] = 0;
  ctx.putImageData(img, 0, 0);
  return c;
}

/** Convert a pointer event on a scaled canvas into normalized [0,1] image coordinates. */
export function eventToNorm(canvas, evt) {
  const rect = canvas.getBoundingClientRect();
  const x = (evt.clientX - rect.left) / rect.width;
  const y = (evt.clientY - rect.top) / rect.height;
  return { x: Math.min(1, Math.max(0, x)), y: Math.min(1, Math.max(0, y)) };
}

export const SAM_CSS = `
.dropzone {
  border: 2px dashed var(--border-strong); border-radius: var(--radius); background: var(--bg-raised);
  padding: 1rem; text-align: center; cursor: pointer; transition: border-color .15s, background .15s;
}
.dropzone.drag { border-color: var(--accent); background: var(--bg-secondary); }
.dropzone:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.sample-strip { display: flex; gap: .5rem; flex-wrap: wrap; margin: .5rem 0; }
.sample-thumb {
  inline-size: 76px; block-size: 56px; object-fit: cover; border-radius: 6px;
  border: 2px solid transparent; cursor: pointer; padding: 0;
}
.sample-thumb.active { border-color: var(--accent); }
.sample-thumb:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.canvas-wrap {
  position: relative; display: block; margin-top: .5rem; border-radius: 8px; overflow: hidden;
  background: var(--bg-raised); border: 1px solid var(--border);
}
.stage-canvas {
  display: block; inline-size: 100%; block-size: auto; max-block-size: 62vh; object-fit: contain;
  cursor: crosshair;
}
.stage-canvas:focus-visible { outline: 3px solid var(--accent); outline-offset: -3px; }
.checker {
  background-image:
    linear-gradient(45deg, var(--bg-secondary) 25%, transparent 25%),
    linear-gradient(-45deg, var(--bg-secondary) 25%, transparent 25%),
    linear-gradient(45deg, transparent 75%, var(--bg-secondary) 75%),
    linear-gradient(-45deg, transparent 75%, var(--bg-secondary) 75%);
  background-size: 20px 20px; background-position: 0 0, 0 10px, 10px -10px, -10px 0;
}
.readout {
  display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem;
}
.readout b { color: var(--color); font-weight: 600; }
.chip {
  font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer;
}
.chip:hover { border-color: var(--accent); }
.chip[aria-pressed=true] { border-color: var(--accent); background: var(--bg-secondary); }
.point-grid { display: grid; grid-template-columns: repeat(3, auto); gap: .3rem; inline-size: max-content; margin: .5rem 0; }
.point-grid button { padding: .3rem .5rem; font-size: .8rem; }
.inside-table { inline-size: 100%; border-collapse: collapse; font-size: .82rem; margin-top: .5rem; }
.inside-table th, .inside-table td {
  text-align: left; padding: .3rem .5rem; border-bottom: 1px solid var(--border); font-family: var(--font-mono);
}
.inside-table th { color: var(--muted); font-weight: 600; }
.iou-badge {
  display: inline-block; font-family: var(--font-mono); font-size: 1.1rem; font-weight: 600;
  padding: .1rem .5rem; border-radius: 6px; background: var(--bg-raised); border: 1px solid var(--border);
}
.fallback { border: 1px solid var(--warn); border-radius: var(--radius); background: var(--bg-raised); padding: 1rem; }
.fallback code { background: var(--bg-secondary); padding: .05rem .3rem; border-radius: 4px; }
`;
