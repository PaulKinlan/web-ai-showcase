// Shared front-end helpers for the Swin2SR super-resolution pages. Keeps each page thin: it owns the
// worker handshake, image<->canvas plumbing, the bicubic "before" reference, a sharpness proxy, and the
// before/after comparison slider. ALL inference is in worker.js (off the main thread).

const WORKER_URL = "/web-ai-showcase/models/swin2sr-super-resolution/worker.js";

// Long-edge cap on the INPUT. Swin2SR memory/latency grow with input area, so we bound it and tile.
// A 512-px input already becomes a 1024-px output. Larger uploads are downscaled to this first (noted
// honestly in the UI) so the demo stays runnable in a browser.
export const MAX_INPUT_EDGE = 512;

export class SuperResEngine {
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
    } else if (msg.type === "tile") {
      this._pending.get(msg.id)?.onTile?.(msg);
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

  /** Upscale a source canvas 2×. `onTile({done,total,tileMs})` fires per tile for honest progress. */
  upscale(sourceCanvas, { tile = 128, onTile } = {}) {
    const id = ++this._id;
    const ctx = sourceCanvas.getContext("2d", { willReadFrequently: true });
    const img = ctx.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height);
    const rgba = img.data; // Uint8ClampedArray, transferred to the worker
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject, onTile });
      this.worker.postMessage(
        { type: "run", id, rgba, width: img.width, height: img.height, tile },
        [rgba.buffer],
      );
    });
  }
}

/** Read a File (from upload or drop) into a data URL. */
export function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}

/** Decode a URL/data-URL into an HTMLImageElement. */
export function decodeImage(url) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.crossOrigin = "anonymous";
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error("Could not decode image"));
    im.src = url;
  });
}

/**
 * Draw an image element into a fresh canvas, capping the long edge to MAX_INPUT_EDGE.
 * Returns { canvas, capped:boolean, srcW, srcH }. Nearest-neighbour is NOT used — this is the true
 * low-res input the model will see.
 */
export function toInputCanvas(imgEl, maxEdge = MAX_INPUT_EDGE) {
  const sw = imgEl.naturalWidth || imgEl.width;
  const sh = imgEl.naturalHeight || imgEl.height;
  const scale = Math.min(1, maxEdge / Math.max(sw, sh));
  const w = Math.max(1, Math.round(sw * scale));
  const h = Math.max(1, Math.round(sh * scale));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = scale < 1; // only smooth when actually shrinking an oversized upload
  ctx.drawImage(imgEl, 0, 0, w, h);
  return { canvas, capped: scale < 1, srcW: sw, srcH: sh, w, h };
}

/** Bicubic-ish 2× upscale of a source canvas via the browser's smooth scaler — the "before" baseline. */
export function bicubicUpscale(sourceCanvas, scale = 2) {
  const canvas = document.createElement("canvas");
  canvas.width = sourceCanvas.width * scale;
  canvas.height = sourceCanvas.height * scale;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(sourceCanvas, 0, 0, canvas.width, canvas.height);
  return canvas;
}

/** Put a raw RGBA buffer into a (new) canvas at the given dimensions. */
export function rgbaToCanvas(rgba, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d").putImageData(new ImageData(rgba, width, height), 0, 0);
  return canvas;
}

/**
 * Sharpness proxy: mean absolute Laplacian (high-frequency edge energy) over a canvas, on luma.
 * Higher = crisper edges. We compare the model output against the bicubic baseline to show the gain.
 */
export function sharpnessEnergy(canvas) {
  const { width: w, height: h } = canvas;
  const d = canvas.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, w, h).data;
  const luma = new Float32Array(w * h);
  for (let i = 0, p = 0; i < luma.length; i++, p += 4) {
    luma[i] = 0.299 * d[p] + 0.587 * d[p + 1] + 0.114 * d[p + 2];
  }
  let sum = 0, n = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const lap = 4 * luma[i] - luma[i - 1] - luma[i + 1] - luma[i - w] - luma[i + w];
      sum += Math.abs(lap);
      n++;
    }
  }
  return n ? sum / n : 0;
}

/**
 * Wire an accessible before/after comparison. `beforeCanvas` sits on top, clipped from the right by the
 * range slider (keyboard-operable); `afterCanvas` shows through underneath. The visual divider tracks
 * the slider value. Both canvases must share pixel dimensions.
 */
export function setupCompare({ container, beforeCanvas, afterCanvas, slider, divider }) {
  afterCanvas.classList.add("cmp-layer", "cmp-after");
  beforeCanvas.classList.add("cmp-layer", "cmp-before");
  container.classList.add("cmp");
  const apply = () => {
    const v = +slider.value; // 0..100 — percent of the BEFORE image shown from the left
    beforeCanvas.style.clipPath = `inset(0 ${100 - v}% 0 0)`;
    if (divider) divider.style.insetInlineStart = `${v}%`;
    slider.setAttribute("aria-valuetext", `${v}% original, ${100 - v}% upscaled`);
  };
  slider.addEventListener("input", apply);
  apply();
  return { apply };
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Shared inline styles for the Swin2SR widgets (dropzone, sample strip, compare slider, readouts). */
export const SUPERRES_CSS = `
.dropzone {
  border: 2px dashed var(--border-strong);
  border-radius: var(--radius);
  background: var(--bg-raised);
  padding: 1rem;
  text-align: center;
  cursor: pointer;
  transition: border-color .15s, background .15s;
}
.dropzone.drag { border-color: var(--accent); background: var(--bg-secondary); }
.dropzone:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.sample-strip { display: flex; gap: .5rem; flex-wrap: wrap; margin: .5rem 0; }
.sample-thumb {
  inline-size: 76px; block-size: 56px; object-fit: cover; border-radius: 6px;
  border: 2px solid transparent; cursor: pointer; padding: 0; background: var(--bg-raised);
  image-rendering: pixelated;
}
.sample-thumb.active { border-color: var(--accent); }
.sample-thumb:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.cmp {
  position: relative; display: block; margin-top: .5rem; border-radius: 8px; overflow: hidden;
  background: var(--bg-raised); border: 1px solid var(--border); line-height: 0;
  max-inline-size: 100%;
}
.cmp-layer {
  display: block; inline-size: 100%; block-size: auto; max-block-size: 62vh; object-fit: contain;
  grid-area: 1 / 1;
}
.cmp { display: grid; }
.cmp-before { z-index: 2; }
.cmp-after { z-index: 1; }
.cmp-divider {
  position: absolute; inset-block: 0; inline-size: 2px; background: var(--accent); z-index: 3;
  transform: translateX(-1px); pointer-events: none;
}
.cmp-tag {
  position: absolute; inset-block-start: .4rem; font-size: .7rem; font-family: var(--font-mono);
  padding: .1rem .4rem; border-radius: 4px; background: color-mix(in srgb, var(--bg) 75%, transparent);
  color: var(--color); z-index: 4; pointer-events: none; border: 1px solid var(--border);
}
.cmp-tag.left { inset-inline-start: .4rem; }
.cmp-tag.right { inset-inline-end: .4rem; }
.single-canvas {
  display: block; inline-size: 100%; block-size: auto; max-block-size: 62vh; object-fit: contain;
  border-radius: 8px; border: 1px solid var(--border); background: var(--bg-raised);
  image-rendering: pixelated;
}
.single-canvas[hidden] { display: none; }
.slider-row { display: flex; align-items: center; gap: .6rem; margin: .6rem 0; flex-wrap: wrap; }
.slider-row input[type=range] { flex: 1 1 180px; accent-color: var(--accent); }
.slider-row output { font-family: var(--font-mono); font-size: .82rem; min-inline-size: 3ch; }
.readout {
  display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem;
}
.readout b { color: var(--color); font-weight: 600; }
.tile-grid { display: flex; flex-wrap: wrap; gap: .25rem; margin-top: .5rem; }
.tile-dot {
  inline-size: 1.4rem; block-size: 1.4rem; border-radius: 4px; border: 1px solid var(--border);
  background: var(--bg-raised); font-size: .6rem; display: grid; place-items: center;
  font-family: var(--font-mono); color: var(--muted);
}
.tile-dot.done { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
.inside-table { inline-size: 100%; border-collapse: collapse; font-size: .82rem; margin-top: .5rem; }
.inside-table th, .inside-table td {
  text-align: left; padding: .3rem .5rem; border-bottom: 1px solid var(--border);
  font-family: var(--font-mono);
}
.inside-table th { color: var(--muted); font-weight: 600; }
.fallback { border: 1px solid var(--warn); border-radius: var(--radius); background: var(--bg-raised); padding: 1rem; }
`;
