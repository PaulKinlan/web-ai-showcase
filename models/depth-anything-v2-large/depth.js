// Front-end helpers shared by every Depth Anything page. Keeps pages thin: owns the worker handshake,
// turns files/samples into data URLs, colourises the depth map, and does the per-pixel parallax warp.
// All inference lives in worker.js (off the main thread).

const WORKER_URL = "/web-ai-showcase/models/depth-anything-v2-large/worker.js";

export class DepthEngine {
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
      this.dtype = msg.dtype;
      for (const w of this._loadWaiters) w.resolve(msg.device);
      this._loadWaiters = [];
    } else if (msg.type === "result") {
      const p = this._pending.get(msg.id);
      if (p) {
        this._pending.delete(msg.id);
        msg.depth = new Uint8Array(msg.depth); // rehydrate the transferred buffer
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

  estimate(imageURL) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, image: imageURL });
    });
  }
}

/** Read a File (upload or drop) into a data URL usable by the worker. */
export function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}

/** Fetch a same-origin sample and return a self-contained data URL for the worker. */
export async function urlToDataURL(src) {
  const blob = await (await fetch(src)).blob();
  return fileToDataURL(new File([blob], "sample", { type: blob.type }));
}

// --- Colour maps (perceptual ramps as control-point stops; interpolated in sRGB) ---------------
const MAPS = {
  turbo: [
    [48, 18, 59],
    [65, 69, 171],
    [57, 118, 209],
    [32, 163, 181],
    [48, 196, 120],
    [140, 208, 52],
    [216, 182, 29],
    [238, 116, 32],
    [165, 20, 24],
  ],
  viridis: [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
  magma: [[0, 0, 4], [81, 18, 124], [183, 55, 121], [252, 137, 97], [252, 253, 191]],
  gray: [[0, 0, 0], [255, 255, 255]],
};
export const COLORMAPS = Object.keys(MAPS);

/** Sample a named colour map at t∈[0,1] → [r,g,b]. */
export function sampleMap(name, t) {
  const stops = MAPS[name] ?? MAPS.turbo;
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.floor(x), f = x - i;
  const a = stops[i], b = stops[Math.min(stops.length - 1, i + 1)];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/** Draw a colourised depth map into a canvas. `depth` = Uint8Array (w*h), bright/warm = nearer. */
export function renderDepthColor(canvas, width, height, depth, mapName = "turbo") {
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(width, height);
  const px = img.data;
  for (let i = 0; i < depth.length; i++) {
    const [r, g, b] = sampleMap(mapName, depth[i] / 255);
    const o = i * 4;
    px[o] = r;
    px[o + 1] = g;
    px[o + 2] = b;
    px[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

/** A small legend strip for the active colour map (far → near). */
export function renderColorLegend(canvas, mapName = "turbo") {
  const w = canvas.width || 220, h = canvas.height || 14;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  for (let x = 0; x < w; x++) {
    const [r, g, b] = sampleMap(mapName, x / (w - 1));
    for (let y = 0; y < h; y++) {
      const o = (y * w + x) * 4;
      img.data[o] = r;
      img.data[o + 1] = g;
      img.data[o + 2] = b;
      img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

/**
 * Backward-mapped parallax: for each output pixel sample the source shifted by (depth-0.5)*offset.
 * Nearer pixels (higher depth) move more, so the photo appears to tilt in 3D. Nearest-neighbour so it
 * stays cheap enough to run every animation frame at working resolution.
 * @param {Uint8ClampedArray} out  destination RGBA buffer (w*h*4)
 * @param {Uint8ClampedArray} src  source RGBA buffer (w*h*4)
 * @param {Uint8Array} depth       per-pixel depth 0–255 (w*h)
 */
export function parallaxWarp(out, src, depth, w, h, offX, offY) {
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const d = depth[i] / 255 - 0.5;
      let sx = (x + d * offX) | 0;
      let sy = (y + d * offY) | 0;
      if (sx < 0) sx = 0;
      else if (sx >= w) sx = w - 1;
      if (sy < 0) sy = 0;
      else if (sy >= h) sy = h - 1;
      const s = (sy * w + sx) * 4, o = i * 4;
      out[o] = src[s];
      out[o + 1] = src[s + 1];
      out[o + 2] = src[s + 2];
      out[o + 3] = 255;
    }
  }
}

/** Draw an image element into a canvas at a capped working size; return {w,h} used. */
export function fitCanvas(canvas, imgEl, maxSide = 480) {
  const iw = imgEl.naturalWidth || imgEl.videoWidth || imgEl.width;
  const ih = imgEl.naturalHeight || imgEl.videoHeight || imgEl.height;
  const scale = Math.min(1, maxSide / Math.max(iw, ih));
  const w = Math.max(1, Math.round(iw * scale)), h = Math.max(1, Math.round(ih * scale));
  canvas.width = w;
  canvas.height = h;
  canvas.getContext("2d").drawImage(imgEl, 0, 0, w, h);
  return { w, h };
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Shared inline styles for the depth widgets (dropzone, samples, canvases, readouts). */
export const DEPTH_CSS = `
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
.canvas-grid { display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start; margin: .6rem 0; }
.canvas-card { flex: 1 1 300px; min-inline-size: 260px; }
.canvas-card h4 { font-family: var(--font-body); font-size: .8rem; color: var(--muted); margin: 0 0 .3rem; text-transform: uppercase; letter-spacing: .05em; }
.viz-canvas, .preview-img {
  inline-size: 100%; block-size: auto; max-block-size: 420px; object-fit: contain;
  border-radius: 8px; border: 1px solid var(--border); background: var(--bg-raised); display: block;
}
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.controls-row { display: flex; flex-wrap: wrap; gap: .6rem; align-items: center; margin: .6rem 0; }
.seg { display: inline-flex; border: 1px solid var(--border-strong); border-radius: 8px; overflow: hidden; }
.seg button { border: 0; border-radius: 0; background: var(--bg-raised); color: var(--color); font-size: .82rem; }
.seg button[aria-pressed=true] { background: var(--accent); color: var(--accent-ink); }
.legend-row { display: flex; align-items: center; gap: .5rem; font-family: var(--font-mono); font-size: .72rem; color: var(--muted); margin-top: .3rem; }
.legend-canvas { border-radius: 4px; border: 1px solid var(--border); block-size: 14px; inline-size: 220px; }
.histo { display: flex; align-items: flex-end; gap: 1px; block-size: 70px; margin-top: .4rem; }
.histo span { flex: 1 1 0; background: var(--accent); border-radius: 1px 1px 0 0; min-block-size: 1px; opacity: .85; }
.inside-table { inline-size: 100%; border-collapse: collapse; font-size: .82rem; margin-top: .5rem; }
.inside-table th, .inside-table td { text-align: left; padding: .3rem .5rem; border-bottom: 1px solid var(--border); font-family: var(--font-mono); }
.inside-table th { color: var(--muted); font-weight: 600; }
.tilt-stage { position: relative; max-inline-size: 520px; margin-inline: auto; touch-action: none; }
.fallback { border: 1px solid var(--warn); border-radius: var(--radius); background: var(--bg-raised); padding: 1rem; }
.fallback code { background: var(--bg-secondary); padding: .05rem .3rem; border-radius: 4px; }
`;
