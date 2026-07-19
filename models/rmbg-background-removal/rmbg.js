// Front-end helpers shared by every RMBG page. Owns the worker handshake, turns files/samples into
// data URLs, and composes the cutout (apply the alpha matte to the photo, over any backdrop). All
// inference lives in worker.js (off the main thread).

const WORKER_URL = "/web-ai-showcase/models/rmbg-background-removal/worker.js";

export class RmbgEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.dtype = "q8";
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
        msg.alpha = new Uint8Array(msg.alpha);
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

/** Draw a light/dark checkerboard (transparency indicator) into a canvas. */
export function paintCheckerboard(ctx, w, h, cell = 12) {
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const a = dark ? "#2a2723" : "#e9e6dd", b = dark ? "#201d1a" : "#f7f5ee";
  for (let y = 0; y < h; y += cell) {
    for (let x = 0; x < w; x += cell) {
      ctx.fillStyle = ((x / cell + y / cell) & 1) ? a : b;
      ctx.fillRect(x, y, cell, cell);
    }
  }
}

/**
 * Compose the subject cutout. Returns nothing; draws into `canvas`.
 * @param opts.backdrop  "checker" | CSS colour string | HTMLImageElement | null(transparent)
 * @param opts.alpha     Uint8Array foreground matte (w*h), 0=background 255=subject
 * @param opts.feather   extra alpha softening (0–1) applied multiplicatively
 * @param opts.cutout    optional worker-composited RGBA ImageBitmap (subject + alpha). When present
 *                       (and feather===1) the main thread just draws it — the per-pixel composite
 *                       already ran off the main thread in worker.js. Keeps INP low at high res.
 */
export function composeCutout(
  canvas,
  img,
  w,
  h,
  alpha,
  { backdrop = "checker", feather = 1, cutout = null } = {},
) {
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);

  // Backdrop first.
  if (backdrop === "checker") paintCheckerboard(ctx, w, h);
  else if (typeof backdrop === "string") {
    ctx.fillStyle = backdrop;
    ctx.fillRect(0, 0, w, h);
  } else if (backdrop && backdrop.tagName === "IMG") ctx.drawImage(backdrop, 0, 0, w, h);

  // Fast path: the worker already stamped the matte into the photo's alpha (off the main thread) and
  // handed back an ImageBitmap — just draw it over the backdrop. Main thread does zero per-pixel work.
  if (cutout && feather === 1) {
    ctx.drawImage(cutout, 0, 0);
    return;
  }

  // Fallback (no bitmap, or a feather multiplier that re-weights the baked alpha): draw the photo to an
  // offscreen buffer, stamp the matte into its alpha, blit on top.
  const off = document.createElement("canvas");
  off.width = w;
  off.height = h;
  const octx = off.getContext("2d");
  octx.drawImage(img, 0, 0, w, h);
  const id = octx.getImageData(0, 0, w, h);
  for (let i = 0; i < alpha.length; i++) {
    id.data[i * 4 + 3] = feather === 1 ? alpha[i] : Math.round(alpha[i] * feather);
  }
  octx.putImageData(id, 0, 0);
  ctx.drawImage(off, 0, 0);
}

/**
 * A transparent-background PNG of just the subject (for sticker export). When the worker-composited
 * `cutout` ImageBitmap is passed and no outline is requested, draw it directly (the alpha stamp already
 * ran off the main thread); otherwise fall back to the per-pixel path.
 */
export function cutoutToPNG(img, w, h, alpha, outline = 0, cutout = null) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  if (cutout && outline === 0) {
    ctx.drawImage(cutout, 0, 0);
    return c;
  }
  ctx.drawImage(img, 0, 0, w, h);
  const id = ctx.getImageData(0, 0, w, h);
  for (let i = 0; i < alpha.length; i++) id.data[i * 4 + 3] = alpha[i];
  ctx.putImageData(id, 0, 0);
  if (outline > 0) return addOutline(c, alpha, w, h, outline);
  return c;
}

/** Dilate the matte and paint a solid halo behind the subject — the classic sticker border. */
export function addOutline(subjectCanvas, alpha, w, h, radius, color = "#ffffff") {
  const dil = dilate(alpha, w, h, radius);
  const out = document.createElement("canvas");
  out.width = w;
  out.height = h;
  const ctx = out.getContext("2d");
  // Halo layer.
  const halo = ctx.createImageData(w, h);
  const [r, g, b] = hexToRGB(color);
  for (let i = 0; i < dil.length; i++) {
    const o = i * 4;
    halo.data[o] = r;
    halo.data[o + 1] = g;
    halo.data[o + 2] = b;
    halo.data[o + 3] = dil[i];
  }
  ctx.putImageData(halo, 0, 0);
  ctx.drawImage(subjectCanvas, 0, 0);
  return out;
}

// Simple square-kernel dilation of an alpha matte (thresholded), for the sticker outline.
function dilate(alpha, w, h, r) {
  const bin = new Uint8Array(w * h);
  for (let i = 0; i < alpha.length; i++) bin[i] = alpha[i] >= 128 ? 255 : 0;
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (bin[y * w + x]) {
        out[y * w + x] = 255;
        continue;
      }
      let hit = false;
      for (let dy = -r; dy <= r && !hit; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -r; dx <= r; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          if (bin[yy * w + xx]) {
            hit = true;
            break;
          }
        }
      }
      out[y * w + x] = hit ? 255 : 0;
    }
  }
  return out;
}

function hexToRGB(hex) {
  const m = hex.replace("#", "");
  const n = m.length === 3 ? m.split("").map((c) => c + c).join("") : m;
  return [parseInt(n.slice(0, 2), 16), parseInt(n.slice(2, 4), 16), parseInt(n.slice(4, 6), 16)];
}

/** Render the raw alpha matte as grayscale into a canvas ("See inside"). */
export function renderMatte(canvas, w, h, alpha) {
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const id = ctx.createImageData(w, h);
  for (let i = 0; i < alpha.length; i++) {
    const a = alpha[i], o = i * 4;
    id.data[o] = a;
    id.data[o + 1] = a;
    id.data[o + 2] = a;
    id.data[o + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
}

/** Highlight only the soft edge (partial alpha) — shows how clean the matte's boundary is. */
export function renderEdge(canvas, w, h, alpha) {
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const id = ctx.createImageData(w, h);
  for (let i = 0; i < alpha.length; i++) {
    const a = alpha[i], o = i * 4;
    const edge = a > 25 && a < 230; // partially transparent = the model's hair/edge detail
    id.data[o] = edge ? 0x8a : 0x14;
    id.data[o + 1] = edge ? 0xb4 : 0x12;
    id.data[o + 2] = edge ? 0xf8 : 0x10;
    id.data[o + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
}

export function escapeHTML(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]),
  );
}

export const RMBG_CSS = `
.dropzone { border: 2px dashed var(--border-strong); border-radius: var(--radius); background: var(--bg-raised); padding: 1rem; text-align: center; cursor: pointer; transition: border-color .15s, background .15s; }
.dropzone.drag { border-color: var(--accent); background: var(--bg-secondary); }
.dropzone:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.sample-strip { display: flex; gap: .5rem; flex-wrap: wrap; margin: .5rem 0; }
.sample-thumb { inline-size: 76px; block-size: 56px; object-fit: cover; border-radius: 6px; border: 2px solid transparent; cursor: pointer; padding: 0; }
.sample-thumb.active { border-color: var(--accent); }
.sample-thumb:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.canvas-grid { display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start; margin: .6rem 0; }
.canvas-card { flex: 1 1 300px; min-inline-size: 260px; }
.canvas-card h4 { font-family: var(--font-body); font-size: .8rem; color: var(--muted); margin: 0 0 .3rem; text-transform: uppercase; letter-spacing: .05em; }
.viz-canvas, .preview-img { inline-size: 100%; block-size: auto; max-block-size: 420px; object-fit: contain; border-radius: 8px; border: 1px solid var(--border); background: var(--bg-raised); display: block; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.controls-row { display: flex; flex-wrap: wrap; gap: .6rem; align-items: center; margin: .6rem 0; }
.swatches { display: flex; gap: .4rem; flex-wrap: wrap; align-items: center; }
.swatch { inline-size: 30px; block-size: 30px; border-radius: 50%; border: 2px solid var(--border); cursor: pointer; padding: 0; }
.swatch[aria-pressed=true] { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent); }
.swatch:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.seg { display: inline-flex; border: 1px solid var(--border-strong); border-radius: 8px; overflow: hidden; }
.seg button { border: 0; border-radius: 0; background: var(--bg-raised); color: var(--color); font-size: .82rem; }
.seg button[aria-pressed=true] { background: var(--accent); color: var(--accent-ink); }
.histo { display: flex; align-items: flex-end; gap: 1px; block-size: 70px; margin-top: .4rem; }
.histo span { flex: 1 1 0; background: var(--accent); border-radius: 1px 1px 0 0; min-block-size: 1px; opacity: .85; }
.inside-table { inline-size: 100%; border-collapse: collapse; font-size: .82rem; margin-top: .5rem; }
.inside-table th, .inside-table td { text-align: left; padding: .3rem .5rem; border-bottom: 1px solid var(--border); font-family: var(--font-mono); }
.inside-table th { color: var(--muted); font-weight: 600; }
.fallback { border: 1px solid var(--warn); border-radius: var(--radius); background: var(--bg-raised); padding: 1rem; }
.fallback code { background: var(--bg-secondary); padding: .05rem .3rem; border-radius: 4px; }
`;
