// Front-end helpers shared by every Clothing-Parsing page. Owns the worker handshake, turns
// files/samples into data URLs, and paints the colour-coded garment map (a per-pixel class overlay)
// plus the legend, per-class coverage bars, per-garment cut-out, and recolour. All inference lives in
// worker.js (off the main thread).
//
// Model: Xenova/segformer_b2_clothes (task: image-segmentation; SegFormer-B2 finetuned on human/ATR
// parsing). This is DENSE SEMANTIC segmentation over 18 CLOTHING + BODY classes — every pixel becomes
// upper-clothes / skirt / pants / dress / hat / bag / hair / face / arm / leg … — a different job from
// face parsing (facial regions), portrait matting (one subject cut-out), or scene semantics.

const WORKER_URL = "/web-ai-showcase/models/segformer-b2-clothes/worker.js";

export class ClothesEngine {
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

/** Draw the raw photo into a canvas at its natural resolution. */
export function paintPhoto(canvas, img) {
  const w = img.naturalWidth, h = img.naturalHeight;
  canvas.width = w;
  canvas.height = h;
  canvas.getContext("2d").drawImage(img, 0, 0, w, h);
}

/**
 * Turn an overlay SOURCE into something `drawImage` can scale WITHOUT a main-thread per-pixel readback.
 * The worker composites the colour-coded overlay off the main thread and transfers back an ImageBitmap
 * — the fast path here is a bare drawImage(bitmap). When only the raw RGBA buffer is available, we fall
 * back to a one-off putImageData at MAP resolution (invariant 15 measured fallback).
 */
function overlayToDrawable(source, mapW, mapH) {
  if (typeof ImageBitmap !== "undefined" && source instanceof ImageBitmap) return source;
  const off = document.createElement("canvas");
  off.width = mapW;
  off.height = mapH;
  off.getContext("2d").putImageData(new ImageData(source, mapW, mapH), 0, 0);
  return off;
}

/**
 * Paint the garment map over the photo. `overlay` is EITHER the worker-composited ImageBitmap
 * (preferred — the main thread only drawImages it) OR the raw RGBA buffer (mapW×mapH) fallback; it's
 * scaled to the photo and blended at `opacity` (0–1). Opacity drags re-drawImage the same bitmap.
 */
export function paintClothMap(canvas, img, overlay, mapW, mapH, opacity = 0.6) {
  const w = img.naturalWidth, h = img.naturalHeight;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(img, 0, 0, w, h);
  const src = overlayToDrawable(overlay, mapW, mapH);
  ctx.globalAlpha = opacity;
  ctx.imageSmoothingEnabled = false; // crisp garment boundaries, not blurred
  ctx.drawImage(src, 0, 0, w, h);
  ctx.globalAlpha = 1;
}

/** Paint just the colour-blocked garment map (no photo). */
export function paintMapOnly(canvas, overlay, mapW, mapH, outW, outH) {
  canvas.width = outW;
  canvas.height = outH;
  const src = overlayToDrawable(overlay, mapW, mapH);
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(src, 0, 0, outW, outH);
}

/**
 * Upscale the class-colour overlay to a target size with nearest-neighbour (crisp edges) and return
 * its RGBA pixel data — used to test each output pixel's winning class by colour.
 */
export function scaledOverlayData(overlay, mapW, mapH, outW, outH) {
  const src = document.createElement("canvas");
  src.width = mapW;
  src.height = mapH;
  src.getContext("2d").putImageData(new ImageData(overlay, mapW, mapH), 0, 0);
  const dst = document.createElement("canvas");
  dst.width = outW;
  dst.height = outH;
  const ctx = dst.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(src, 0, 0, outW, outH);
  return ctx.getImageData(0, 0, outW, outH).data;
}

/**
 * Keep ONLY the pixels whose winning class is in `keepRgbSet` (a Set of "r,g,b" overlay colours),
 * everything else transparent — the "extract the clothing" cut-out. Draws into `canvas` at photo
 * resolution. Optional checkerboard shows the transparency.
 */
export function cutoutClasses(
  canvas,
  photo,
  overlay,
  mapW,
  mapH,
  keepRgbSet,
  { checker = true } = {},
) {
  const w = photo.naturalWidth, h = photo.naturalHeight;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  if (checker) paintChecker(ctx, w, h);
  const cls = scaledOverlayData(overlay, mapW, mapH, w, h);
  // draw the photo, then knock out non-kept pixels via a scratch buffer
  const scratch = document.createElement("canvas");
  scratch.width = w;
  scratch.height = h;
  const sctx = scratch.getContext("2d");
  sctx.drawImage(photo, 0, 0, w, h);
  const img = sctx.getImageData(0, 0, w, h);
  const d = img.data;
  let kept = 0;
  for (let i = 0; i < w * h; i++) {
    const o = i * 4;
    const key = cls[o] + "," + cls[o + 1] + "," + cls[o + 2];
    if (keepRgbSet.has(key) && cls[o + 3] > 0) kept++;
    else d[o + 3] = 0;
  }
  sctx.putImageData(img, 0, 0);
  ctx.drawImage(scratch, 0, 0);
  return kept;
}

function paintChecker(ctx, w, h) {
  const s = 12;
  for (let y = 0; y < h; y += s) {
    for (let x = 0; x < w; x += s) {
      ctx.fillStyle = ((x / s + y / s) & 1) ? "#e9e6df" : "#cfcabf";
      ctx.fillRect(x, y, s, s);
    }
  }
}

/**
 * Tint one or more classes with a colour, keeping the photo everywhere else — the "recolour a garment"
 * compositor. `targets` = [{rgb:[r,g,b], tint:[r,g,b], alpha}]. Draws into `canvas` at photo resolution.
 */
export function tintClasses(canvas, photo, overlay, mapW, mapH, targets) {
  const w = photo.naturalWidth, h = photo.naturalHeight;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(photo, 0, 0, w, h);
  if (!targets || !targets.length) return;
  const cls = scaledOverlayData(overlay, mapW, mapH, w, h);
  const base = ctx.getImageData(0, 0, w, h);
  const d = base.data;
  const lut = new Map(targets.map((t) => [t.rgb.join(","), t]));
  for (let i = 0; i < w * h; i++) {
    const o = i * 4;
    const key = cls[o] + "," + cls[o + 1] + "," + cls[o + 2];
    const t = lut.get(key);
    if (!t || cls[o + 3] === 0) continue;
    const a = t.alpha ?? 0.5;
    d[o] = Math.round(d[o] * (1 - a) + t.tint[0] * a);
    d[o + 1] = Math.round(d[o + 1] * (1 - a) + t.tint[1] * a);
    d[o + 2] = Math.round(d[o + 2] * (1 - a) + t.tint[2] * a);
  }
  ctx.putImageData(base, 0, 0);
}

/** Coverage (0–1) of one class label from the classes array. */
export function classCoverage(classes, label) {
  const c = classes.find((x) => x.label === label);
  return c ? c.coverage : 0;
}

/** Sum coverage of a group of labels. */
export function groupCoverage(classes, labels) {
  const set = new Set(labels);
  return classes.filter((c) => set.has(c.label)).reduce((s, c) => s + c.coverage, 0);
}

export const CLOTHES_CSS = `
.dropzone { border: 2px dashed var(--border-strong); border-radius: var(--radius); background: var(--bg-raised); padding: 1rem; text-align: center; cursor: pointer; transition: border-color .15s, background .15s; }
.dropzone.drag { border-color: var(--accent); background: var(--bg-secondary); }
.dropzone:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.sample-strip { display: flex; gap: .5rem; flex-wrap: wrap; margin: .5rem 0; }
.sample-thumb { inline-size: 64px; block-size: 64px; object-fit: cover; border-radius: 8px; border: 2px solid transparent; cursor: pointer; padding: 0; }
.sample-thumb.active { border-color: var(--accent); }
.sample-thumb:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.canvas-wrap { position: relative; display: block; margin-top: .5rem; border-radius: 8px; overflow: hidden; background: var(--bg-raised); border: 1px solid var(--border); }
.stage-canvas { display: block; inline-size: 100%; block-size: auto; max-block-size: 62vh; object-fit: contain; }
.stage-canvas:focus-visible { outline: 3px solid var(--accent); outline-offset: -3px; }
.canvas-grid { display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start; margin: .6rem 0; }
.canvas-card { flex: 1 1 300px; min-inline-size: 240px; }
.canvas-card h4 { font-family: var(--font-body); font-size: .8rem; color: var(--muted); margin: 0 0 .3rem; text-transform: uppercase; letter-spacing: .05em; }
.viz-canvas { inline-size: 100%; block-size: auto; max-block-size: 420px; object-fit: contain; border-radius: 8px; border: 1px solid var(--border); background: var(--bg-raised); display: block; }
.slider-row { display: flex; align-items: center; gap: .6rem; margin: .6rem 0; flex-wrap: wrap; }
.slider-row input[type=range] { flex: 1 1 180px; accent-color: var(--accent); }
.slider-row output { font-family: var(--font-mono); font-size: .82rem; min-inline-size: 3ch; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.legend { display: flex; flex-wrap: wrap; gap: .35rem; margin: .6rem 0; }
.legend-item { display: inline-flex; align-items: center; gap: .4rem; font-size: .82rem; padding: .18rem .55rem; border-radius: 999px; border: 1px solid var(--border); background: var(--bg-raised); }
.legend-item .swatch { inline-size: .85rem; block-size: .85rem; border-radius: 3px; flex: none; border: 1px solid rgba(0,0,0,.15); }
.legend-item b { font-family: var(--font-mono); color: var(--muted); font-weight: 600; }
.cov-list { display: flex; flex-direction: column; gap: .3rem; margin: .5rem 0; }
.cov-row { display: grid; grid-template-columns: 8.5rem 1fr 3.2rem; align-items: center; gap: .5rem; font-size: .82rem; }
.cov-row .name { display: inline-flex; align-items: center; gap: .4rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cov-row .name .swatch { inline-size: .8rem; block-size: .8rem; border-radius: 3px; flex: none; border: 1px solid rgba(0,0,0,.15); }
.cov-row .bar { block-size: .7rem; border-radius: 4px; background: var(--bg-secondary); overflow: hidden; }
.cov-row .bar > span { display: block; block-size: 100%; border-radius: 4px; }
.cov-row .pct { font-family: var(--font-mono); text-align: right; color: var(--muted); }
.chip-toggle { display:inline-flex; align-items:center; gap:.4rem; padding:.45rem .7rem; min-block-size:44px; border:1px solid var(--border); border-radius:999px; background:var(--bg-raised); color:var(--color); cursor:pointer; font-size:.82rem; }
.chip-toggle[aria-pressed=true] { border-color:var(--accent); background:var(--bg-secondary); }
.chip-toggle .swatch { inline-size:.85rem; block-size:.85rem; border-radius:3px; flex:none; border:1px solid rgba(0,0,0,.2); }
.chip-wrap { display:flex; flex-wrap:wrap; gap:.4rem; margin:.5rem 0; }
.swatch-btn { display:inline-flex; align-items:center; gap:.4rem; padding:.4rem .7rem; min-block-size:44px; border:1px solid var(--border); border-radius:999px; background:var(--bg-raised); color:var(--color); cursor:pointer; font-size:.82rem; }
.swatch-btn[aria-pressed=true] { border-color:var(--accent); background:var(--bg-secondary); }
.swatch-btn .dot { inline-size:.9rem; block-size:.9rem; border-radius:50%; border:1px solid rgba(0,0,0,.2); }
.inside-table { inline-size: 100%; max-inline-size: 100%; table-layout: fixed; border-collapse: collapse; font-size: .82rem; margin-top: .5rem; }
.inside-table th, .inside-table td { text-align: left; padding: .3rem .5rem; border-bottom: 1px solid var(--border); font-family: var(--font-mono); overflow-wrap: anywhere; word-break: break-word; }
.inside-table th { color: var(--muted); font-weight: 600; inline-size: 7.5rem; }
.credit { font-size: .72rem; color: var(--muted); margin: .3rem 0 0; }
.fallback { border: 1px solid var(--warn); border-radius: var(--radius); background: var(--bg-raised); padding: 1rem; }
`;
