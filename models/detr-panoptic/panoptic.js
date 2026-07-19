// Front-end helpers shared by every DETR-panoptic page. Owns the worker handshake, turns files/samples
// into data URLs, and paints the colour-coded per-instance overlay + legend. All inference lives in
// worker.js (off the main thread — modern-web-guidance: break-up-long-tasks).
//
// Model: Xenova/detr-resnet-50-panoptic (task: image-segmentation). PANOPTIC parse = things (countable
// instances, each its own segment) + stuff (amorphous regions). Distinct from SegFormer's dense
// semantic map (no per-instance split) and SAM's single point-prompted mask.

const WORKER_URL = "/web-ai-showcase/models/detr-panoptic/worker.js";

export class PanopticEngine {
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
        msg.overlay = new Uint8ClampedArray(msg.overlay);
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

  segment(imageURL) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, image: imageURL });
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

export async function urlToDataURL(src) {
  const blob = await (await fetch(src)).blob();
  return fileToDataURL(new File([blob], "sample", { type: blob.type }));
}

export function loadImage(src) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error("image decode failed"));
    im.src = src;
  });
}

export function escapeHTML(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]),
  );
}

export function rgb(seg) {
  return `rgb(${seg.r} ${seg.g} ${seg.b})`;
}

/** Draw the raw photo into a canvas at its natural resolution. */
export function paintPhoto(canvas, img) {
  const w = img.naturalWidth, h = img.naturalHeight;
  canvas.width = w;
  canvas.height = h;
  canvas.getContext("2d").drawImage(img, 0, 0, w, h);
}

/**
 * Paint the panoptic overlay over the photo. `overlay` is an RGBA buffer (mapW×mapH) of the per-
 * segment colours; it's scaled to the photo and blended at `opacity` (0–1). At opacity 1 you get the
 * pure colour-blocked parse; lower it to see the photo beneath. Nearest-neighbour keeps crisp segment
 * boundaries. `onlyIndices` (optional Set) restricts which segments are drawn (used for hover/isolate).
 */
export function paintPanoptic(canvas, img, overlay, mapW, mapH, opacity = 0.55) {
  const w = img.naturalWidth, h = img.naturalHeight;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(img, 0, 0, w, h);

  const off = document.createElement("canvas");
  off.width = mapW;
  off.height = mapH;
  off.getContext("2d").putImageData(new ImageData(overlay, mapW, mapH), 0, 0);

  ctx.globalAlpha = opacity;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, w, h);
  ctx.globalAlpha = 1;
}

/** Paint just the colour-blocked parse (no photo) — the "turn a scene into a map" view. */
export function paintMapOnly(canvas, overlay, mapW, mapH, outW, outH) {
  canvas.width = outW;
  canvas.height = outH;
  const off = document.createElement("canvas");
  off.width = mapW;
  off.height = mapH;
  off.getContext("2d").putImageData(new ImageData(overlay, mapW, mapH), 0, 0);
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, outW, outH);
}

/**
 * Build an overlay buffer containing ONLY the segments whose index is in `keep` (a Set). Used to
 * isolate a single segment (multi-model page / hover), by masking every other segment's colour out.
 */
export function overlayForSegments(fullOverlay, segments, keep) {
  const colorKey = new Map();
  for (const s of segments) colorKey.set(`${s.r},${s.g},${s.b}`, s.index);
  const out = new Uint8ClampedArray(fullOverlay.length);
  for (let i = 0; i < fullOverlay.length; i += 4) {
    const key = `${fullOverlay[i]},${fullOverlay[i + 1]},${fullOverlay[i + 2]}`;
    const idx = colorKey.get(key);
    if (idx != null && keep.has(idx) && fullOverlay[i + 3] > 0) {
      out[i] = fullOverlay[i];
      out[i + 1] = fullOverlay[i + 1];
      out[i + 2] = fullOverlay[i + 2];
      out[i + 3] = 255;
    }
  }
  return out;
}

export const PANOPTIC_CSS = `
.dropzone { border: 2px dashed var(--border-strong); border-radius: var(--radius); background: var(--bg-raised); padding: 1rem; text-align: center; cursor: pointer; transition: border-color .15s, background .15s; }
.dropzone.drag { border-color: var(--accent); background: var(--bg-secondary); }
.dropzone:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.sample-strip { display: flex; gap: .5rem; flex-wrap: wrap; margin: .5rem 0; }
.sample-thumb { inline-size: 76px; block-size: 56px; object-fit: cover; border-radius: 6px; border: 2px solid transparent; cursor: pointer; padding: 0; }
.sample-thumb.active { border-color: var(--accent); }
.sample-thumb:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.canvas-wrap { position: relative; display: block; margin-top: .5rem; border-radius: 8px; overflow: hidden; background: var(--bg-raised); border: 1px solid var(--border); }
.stage-canvas { display: block; inline-size: 100%; block-size: auto; max-block-size: 62vh; object-fit: contain; }
.stage-canvas:focus-visible { outline: 3px solid var(--accent); outline-offset: -3px; }
.canvas-grid { display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start; margin: .6rem 0; }
.canvas-card { flex: 1 1 300px; min-inline-size: 0; }
.canvas-card h4 { font-family: var(--font-body); font-size: .8rem; color: var(--muted); margin: 0 0 .3rem; text-transform: uppercase; letter-spacing: .05em; }
.viz-canvas { inline-size: 100%; block-size: auto; max-block-size: 420px; object-fit: contain; border-radius: 8px; border: 1px solid var(--border); background: var(--bg-raised); display: block; }
.slider-row { display: flex; align-items: center; gap: .6rem; margin: .6rem 0; flex-wrap: wrap; }
.slider-row input[type=range] { flex: 1 1 180px; accent-color: var(--accent); min-inline-size: 0; }
.slider-row output { font-family: var(--font-mono); font-size: .82rem; min-inline-size: 3ch; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.split-chips { display: flex; flex-wrap: wrap; gap: .5rem; margin: .5rem 0; }
.split-chip { display: inline-flex; align-items: center; gap: .4rem; font-size: .85rem; padding: .25rem .7rem; border-radius: 999px; border: 1px solid var(--border); background: var(--bg-raised); }
.split-chip b { font-family: var(--font-mono); }
.legend { display: flex; flex-wrap: wrap; gap: .35rem; margin: .6rem 0; }
.seg-item { display: inline-flex; align-items: center; gap: .4rem; font-size: .82rem; padding: .2rem .6rem; border-radius: 999px; border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; min-block-size: 32px; }
.seg-item:hover, .seg-item:focus-visible { border-color: var(--accent); outline: none; }
.seg-item[aria-pressed=true] { border-color: var(--accent); background: var(--bg-secondary); }
.seg-item .swatch { inline-size: .85rem; block-size: .85rem; border-radius: 3px; flex: none; border: 1px solid rgba(0,0,0,.15); }
.seg-item .kind { font-size: .7rem; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
.seg-item b { font-family: var(--font-mono); color: var(--muted); font-weight: 600; }
.inside-table { inline-size: 100%; border-collapse: collapse; font-size: .82rem; margin-top: .5rem; }
.inside-table th, .inside-table td { text-align: left; padding: .3rem .5rem; border-bottom: 1px solid var(--border); font-family: var(--font-mono); }
.inside-table th { color: var(--muted); font-weight: 600; }
.inside-table .swatch { display: inline-block; inline-size: .8rem; block-size: .8rem; border-radius: 3px; vertical-align: middle; margin-inline-end: .35rem; border: 1px solid rgba(0,0,0,.15); }
.fallback { border: 1px solid var(--warn); border-radius: var(--radius); background: var(--bg-raised); padding: 1rem; }
.fallback code { background: var(--bg-secondary); padding: .05rem .3rem; border-radius: 4px; }
`;
