// Front-end helpers shared by every CLIPSeg page. Owns the worker handshake, turns files/samples
// into data URLs, and paints the text-prompted masks + the raw probability heatmap (the "see
// inside" surface). All inference lives in worker.js (off the main thread).
//
// Model: Xenova/clipseg-rd64-refined (CLIPSegForImageSegmentation). This is OPEN-VOCABULARY,
// TEXT-PROMPTED segmentation — you type a phrase ("the dog", "shadows", "the road") and the model
// returns a per-pixel score map for THAT phrase. Different from fixed-class semantic segmentation
// (SegFormer's 150 ADE20K labels) and from point-prompted masks (SAM). One forward pass scores every
// pixel against every prompt at the model's 352×352 working resolution; we upscale to your photo.

const WORKER_URL = "/web-ai-showcase/models/clipseg-text-segmentation/worker.js";

export class ClipsegEngine {
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
        msg.maps = msg.maps.map((m) => ({ prompt: m.prompt, data: new Float32Array(m.data) }));
        p.resolve(msg);
      }
    } else if (msg.type === "composite") {
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

  /** Score `prompts` (array of phrases) against `imageURL`. Resolves
   *  { mapW, mapH, maps:[{prompt, data:Float32Array}], ms, device }. `data` is RAW per-pixel logits.
   *  The worker also caches these logits so the composite helpers below re-threshold off the main
   *  thread on every slider input without re-running the model. */
  segment(imageURL, prompts) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, image: imageURL, prompts });
    });
  }

  // ── Off-main-thread compositing (invariant 15) ─────────────────────────────────────────────────
  // Each builds the coloured RGBA layer in the worker (OffscreenCanvas) against the cached logits and
  // resolves with a transferred ImageBitmap; the page only drawImage()s it. Keeps the threshold/opacity
  // sliders responsive without any main-thread per-pixel loop.
  composite(op, args) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "composite", id, op, args });
    });
  }
  /** Combined mask overlay for `items` = [{mapIndex, colorIndex}] → { bitmap, mapW, mapH }. */
  overlay(items, threshold, opacity) {
    return this.composite("overlay", { items, threshold, opacity });
  }
  /** Inferno probability heatmap for one prompt → { bitmap, mapW, mapH }. */
  heatmap(mapIndex) {
    return this.composite("heatmap", { mapIndex });
  }
  /** Binary alpha stencil (255 above threshold) for a GPU cut-out → { bitmap, above, mapW, mapH }. */
  maskAlpha(mapIndex, threshold) {
    return this.composite("maskAlpha", { mapIndex, threshold });
  }
  /** Per-image normalised relative render → { heat, overlay, above, mapW, mapH } (the "wild" page). */
  norm(mapIndex, threshold) {
    return this.composite("norm", { mapIndex, threshold });
  }
}

// ── Main-thread draw helpers (GPU compositing only — no per-pixel loops) ──────────────────────────

/** Draw the photo, then blit a worker-built overlay ImageBitmap scaled to the photo (nearest→smooth). */
export function drawOverlay(canvas, img, overlayBitmap) {
  const w = img.naturalWidth, h = img.naturalHeight;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0, w, h);
  if (overlayBitmap) {
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(overlayBitmap, 0, 0, w, h);
  }
}

/** Blit a worker-built map-resolution bitmap (e.g. the heatmap) into a canvas sized to the bitmap. */
export function drawBitmap(canvas, bitmap) {
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  canvas.getContext("2d").drawImage(bitmap, 0, 0);
}

/** Cut the object out of the photo using a worker-built alpha stencil as a GPU destination-in mask.
 *  Returns a photo-resolution canvas with everything outside the mask transparent (no per-pixel loop). */
export function cutoutWithMask(img, maskBitmap) {
  const w = img.naturalWidth, h = img.naturalHeight;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  ctx.drawImage(img, 0, 0, w, h);
  ctx.globalCompositeOperation = "destination-in";
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(maskBitmap, 0, 0, w, h);
  ctx.globalCompositeOperation = "source-over";
  return c;
}

// ---- image plumbing ----------------------------------------------------------------------------

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
export function parsePrompts(text, max = 5) {
  return text.split(/[,\n;]/).map((s) => s.trim()).filter(Boolean).slice(0, max);
}

// ---- colour ------------------------------------------------------------------------------------

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

/** Deterministic well-separated colour per phrase INDEX (golden-angle hue walk). */
export function colorForIndex(i) {
  const h = (i * 137.508 + 15) % 360;
  return hslToRgb(h / 360, i % 2 ? 0.72 : 0.62, i % 3 === 0 ? 0.55 : 0.5);
}
export function cssColor(i) {
  const [r, g, b] = colorForIndex(i);
  return `rgb(${r} ${g} ${b})`;
}
function hslToRgb(h, s, l) {
  if (s === 0) {
    const v = Math.round(l * 255);
    return [v, v, v];
  }
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const hk = (t) => {
    let x = t;
    if (x < 0) x += 1;
    if (x > 1) x -= 1;
    if (x < 1 / 6) return p + (q - p) * 6 * x;
    if (x < 1 / 2) return q;
    if (x < 2 / 3) return p + (q - p) * (2 / 3 - x) * 6;
    return p;
  };
  return [
    Math.round(hk(h + 1 / 3) * 255),
    Math.round(hk(h) * 255),
    Math.round(hk(h - 1 / 3) * 255),
  ];
}

// A compact perceptual "inferno"-style ramp for the probability heatmap (0 → dark, 1 → bright).
const RAMP = [
  [0, 0, 4],
  [40, 11, 84],
  [101, 21, 110],
  [159, 42, 99],
  [212, 72, 66],
  [245, 125, 21],
  [250, 193, 39],
  [252, 255, 164],
];
export function probColor(t) {
  const x = Math.min(1, Math.max(0, t)) * (RAMP.length - 1);
  const i = Math.floor(x), f = x - i;
  const a = RAMP[i], b = RAMP[Math.min(RAMP.length - 1, i + 1)];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

// ---- stats -------------------------------------------------------------------------------------

/** Fraction of pixels whose sigmoid(logit) exceeds `threshold`. */
export function maskCoverage(map, threshold) {
  let n = 0;
  for (let i = 0; i < map.length; i++) if (sigmoid(map[i]) >= threshold) n++;
  return n / map.length;
}
/** Peak probability + its (x,y) in map space — a "best hit" marker. */
export function peakOfMap(map, w, h) {
  let best = -Infinity, bi = 0;
  for (let i = 0; i < map.length; i++) {
    if (map[i] > best) {
      best = map[i];
      bi = i;
    }
  }
  return { prob: sigmoid(best), logit: best, x: bi % w, y: (bi / w) | 0 };
}
/** Tight bounding box (normalised 0–1) of the thresholded mask, or null if empty. For crop hand-off. */
export function bboxOfMask(map, w, h, threshold) {
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (sigmoid(map[y * w + x]) >= threshold) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  if (x1 < 0) return null;
  return { x: x0 / w, y: y0 / h, w: (x1 - x0 + 1) / w, h: (y1 - y0 + 1) / h };
}

// ---- canvas painters ---------------------------------------------------------------------------

/** Draw the raw photo into a canvas at its natural resolution. */
export function paintPhoto(canvas, img) {
  const w = img.naturalWidth, h = img.naturalHeight;
  canvas.width = w;
  canvas.height = h;
  canvas.getContext("2d").drawImage(img, 0, 0, w, h);
}

export const CLIPSEG_CSS = `
.dropzone { border: 2px dashed var(--border-strong); border-radius: var(--radius); background: var(--bg-raised); padding: 1rem; text-align: center; cursor: pointer; transition: border-color .15s, background .15s; }
.dropzone.drag { border-color: var(--accent); background: var(--bg-secondary); }
.dropzone:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.sample-strip { display: flex; gap: .5rem; flex-wrap: wrap; margin: .5rem 0; }
.sample-thumb { inline-size: 76px; block-size: 56px; object-fit: cover; border-radius: 6px; border: 2px solid transparent; cursor: pointer; padding: 0; }
.sample-thumb.active { border-color: var(--accent); }
.sample-thumb:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.prompt-input { font: inherit; inline-size: 100%; padding: .55rem .7rem; border-radius: 8px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); }
.prompt-input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.chip-row { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0; }
.seg-chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.seg-chip:hover, .seg-chip:focus-visible { border-color: var(--accent); }
.canvas-wrap { position: relative; display: block; margin-top: .6rem; border-radius: 8px; overflow: hidden; background: var(--bg-raised); border: 1px solid var(--border); }
.stage-canvas { display: block; inline-size: 100%; block-size: auto; max-block-size: 62vh; object-fit: contain; }
.canvas-grid { display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start; margin: .6rem 0; }
.canvas-card { flex: 1 1 300px; min-inline-size: 260px; }
.canvas-card h4 { font-family: var(--font-body); font-size: .8rem; color: var(--muted); margin: 0 0 .3rem; text-transform: uppercase; letter-spacing: .05em; }
.viz-canvas { inline-size: 100%; block-size: auto; max-block-size: 420px; object-fit: contain; border-radius: 8px; border: 1px solid var(--border); background: var(--bg-raised); display: block; image-rendering: auto; }
.slider-row { display: flex; align-items: center; gap: .6rem; margin: .6rem 0; flex-wrap: wrap; }
.slider-row label { font-size: .82rem; color: var(--muted); min-inline-size: 6.5rem; }
.slider-row input[type=range] { flex: 1 1 160px; accent-color: var(--accent); }
.slider-row output { font-family: var(--font-mono); font-size: .82rem; min-inline-size: 3.2ch; text-align: right; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.legend { display: flex; flex-wrap: wrap; gap: .35rem; margin: .6rem 0; }
.legend-item { display: inline-flex; align-items: center; gap: .4rem; font-size: .82rem; padding: .18rem .55rem; border-radius: 999px; border: 1px solid var(--border); background: var(--bg-raised); }
.legend-item .swatch { inline-size: .85rem; block-size: .85rem; border-radius: 3px; flex: none; border: 1px solid rgba(0,0,0,.15); }
.legend-item .pct { font-family: var(--font-mono); color: var(--muted); }
.colorbar { block-size: .7rem; border-radius: 4px; margin: .4rem 0 .2rem;
  background: linear-gradient(to right, rgb(0 0 4), rgb(40 11 84), rgb(101 21 110), rgb(159 42 99), rgb(212 72 66), rgb(245 125 21), rgb(250 193 39), rgb(252 255 164)); }
.colorbar-labels { display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: .68rem; color: var(--muted); }
.inside-table { inline-size: 100%; border-collapse: collapse; font-size: .82rem; margin-top: .5rem; }
.inside-table th, .inside-table td { text-align: left; padding: .3rem .5rem; border-bottom: 1px solid var(--border); font-family: var(--font-mono); }
.inside-table th { color: var(--muted); font-weight: 600; }
.fallback { border: 1px solid var(--warn); border-radius: var(--radius); background: var(--bg-raised); padding: 1rem; }
.pill-list { display: flex; flex-wrap: wrap; gap: .35rem; margin: .5rem 0; }
.result-pill { font-family: var(--font-mono); font-size: .82rem; padding: .15rem .55rem; border-radius: 999px; border: 1px solid var(--border); background: var(--bg-raised); }
`;
