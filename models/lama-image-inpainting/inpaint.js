// Front-end helpers shared by every LaMa inpainting page. Keeps pages thin: owns the worker handshake
// (transferring ImageBitmaps so nothing is copied), turns files / gallery images into ImageBitmaps, and
// provides an accessible MASK EDITOR (brush + preset rectangle) so the visitor marks the region to
// remove/fill. ALL inference AND the dense output composite live in worker.js (off the main thread, raw
// ONNX Runtime Web). Privacy by construction: the image, the mask, and every filled pixel never leave
// the device.

const WORKER_URL = "/web-ai-showcase/models/lama-image-inpainting/worker.js";

export class InpaintEngine {
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

  /** Inpaint imageBmp using maskBmp (white/opaque = fill). Both transferred → zero-copy. Returns
   * { resultBmp, fillBmp, w, h, imgW, imgH, holeFraction, ms, infMs, device }. */
  inpaint(imageBmp, maskBmp, opts) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, imageBmp, maskBmp, opts }, [imageBmp, maskBmp]);
    });
  }
}

export function toBitmap(source) {
  return createImageBitmap(source);
}
export async function urlToBitmap(src) {
  return createImageBitmap(await (await fetch(src)).blob());
}

export function drawBitmapFit(canvas, bitmap, maxW = 640) {
  const scale = Math.min(maxW / bitmap.width, 1);
  const cw = Math.round(bitmap.width * scale), ch = Math.round(bitmap.height * scale);
  canvas.width = cw;
  canvas.height = ch;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, cw, ch);
  ctx.drawImage(bitmap, 0, 0, cw, ch);
  return { scale, cw, ch };
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// Accessible mask editor: a base canvas (the image), an overlay canvas the visitor paints on (a
// translucent indigo tint), and an internal white-on-black mask canvas that is what the model receives.
// Brush strokes and the preset rectangle are drawn to BOTH the overlay (for display) and the internal
// mask (for the model). Pointer Events + setPointerCapture give unified mouse/touch/pen; touch-action is
// none on the overlay so a drag paints rather than scrolls. All co-ordinates are converted from client
// space to canvas pixels via getBoundingClientRect so it works at any responsive size / DPR.
export class MaskEditor {
  constructor({ base, overlay, brushSize = 36 }) {
    this.base = base;
    this.overlay = overlay;
    this.mask = document.createElement("canvas"); // internal white-on-black
    this.brush = brushSize;
    this.mode = "brush"; // "brush" | "erase"
    this.drawing = false;
    this.last = null;
    this.onChange = null;
    this._bind();
  }

  _bind() {
    const ov = this.overlay;
    ov.style.touchAction = "none";
    ov.addEventListener("pointerdown", (e) => {
      if (!this.base.width) return;
      this.drawing = true;
      ov.setPointerCapture(e.pointerId);
      this.last = this._pt(e);
      this._stampAt(this.last);
      e.preventDefault();
    });
    ov.addEventListener("pointermove", (e) => {
      if (!this.drawing) return;
      const p = this._pt(e);
      this._line(this.last, p);
      this.last = p;
      e.preventDefault();
    });
    const end = (e) => {
      if (!this.drawing) return;
      this.drawing = false;
      try {
        ov.releasePointerCapture(e.pointerId);
      } catch {}
      this.onChange?.();
    };
    ov.addEventListener("pointerup", end);
    ov.addEventListener("pointercancel", end);
    ov.addEventListener("pointerleave", end);
  }

  _pt(e) {
    const r = this.overlay.getBoundingClientRect();
    return {
      x: (e.clientX - r.left) * (this.overlay.width / r.width),
      y: (e.clientY - r.top) * (this.overlay.height / r.height),
    };
  }

  setImage(bitmap, maxW = 640) {
    const { cw, ch } = drawBitmapFit(this.base, bitmap, maxW);
    for (const c of [this.overlay, this.mask]) {
      c.width = cw;
      c.height = ch;
    }
    this.clear();
  }

  _stampAt(p) {
    const octx = this.overlay.getContext("2d");
    const mctx = this.mask.getContext("2d");
    if (this.mode === "erase") {
      octx.save();
      mctx.save();
      octx.globalCompositeOperation = "destination-out";
      mctx.globalCompositeOperation = "destination-out";
      octx.beginPath();
      octx.arc(p.x, p.y, this.brush / 2, 0, Math.PI * 2);
      octx.fill();
      mctx.beginPath();
      mctx.arc(p.x, p.y, this.brush / 2, 0, Math.PI * 2);
      mctx.fill();
      octx.restore();
      mctx.restore();
      return;
    }
    octx.fillStyle = "rgba(84, 74, 224, 0.55)";
    octx.beginPath();
    octx.arc(p.x, p.y, this.brush / 2, 0, Math.PI * 2);
    octx.fill();
    mctx.fillStyle = "#fff";
    mctx.beginPath();
    mctx.arc(p.x, p.y, this.brush / 2, 0, Math.PI * 2);
    mctx.fill();
  }

  _line(a, b) {
    // stamp along the segment so fast drags stay continuous
    const dist = Math.hypot(b.x - a.x, b.y - a.y);
    const step = Math.max(2, this.brush / 4);
    const n = Math.max(1, Math.ceil(dist / step));
    for (let i = 1; i <= n; i++) {
      this._stampAt({ x: a.x + (b.x - a.x) * (i / n), y: a.y + (b.y - a.y) * (i / n) });
    }
  }

  /** Fill a centred rectangle covering `frac` of each dimension (preset "remove the middle"). */
  presetRect(frac = 0.4) {
    if (!this.overlay.width) return;
    const w = this.overlay.width, h = this.overlay.height;
    const rw = w * frac, rh = h * frac;
    const x = (w - rw) / 2, y = (h - rh) / 2;
    this.overlay.getContext("2d").fillStyle = "rgba(84, 74, 224, 0.55)";
    this.overlay.getContext("2d").fillRect(x, y, rw, rh);
    const mctx = this.mask.getContext("2d");
    mctx.fillStyle = "#fff";
    mctx.fillRect(x, y, rw, rh);
    this.onChange?.();
  }

  clear() {
    this.overlay.getContext("2d").clearRect(0, 0, this.overlay.width, this.overlay.height);
    const mctx = this.mask.getContext("2d");
    mctx.clearRect(0, 0, this.mask.width, this.mask.height);
    this.onChange?.();
  }

  isEmpty() {
    if (!this.mask.width) return true;
    const d = this.mask.getContext("2d").getImageData(0, 0, this.mask.width, this.mask.height).data;
    for (let i = 3; i < d.length; i += 4) if (d[i] > 20) return false;
    return true;
  }

  maskBitmap() {
    // white-on-black opaque mask for the model (composite the alpha strokes over black)
    const c = document.createElement("canvas");
    c.width = this.mask.width;
    c.height = this.mask.height;
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.drawImage(this.mask, 0, 0);
    return createImageBitmap(c);
  }
}

/** Shared inline styles for the inpainting widgets. Structural only — colours from design-system vars. */
export const INPAINT_CSS = `
.dropzone { border: 2px dashed var(--border-strong); border-radius: var(--radius);
  background: var(--bg-raised); padding: .8rem; text-align: center; cursor: pointer;
  transition: border-color .15s, background .15s; font-size: .85rem; }
.dropzone.drag { border-color: var(--accent); background: var(--bg-secondary); }
.dropzone:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.paint-stage { position: relative; inline-size: 100%; max-inline-size: 640px; border-radius: 6px;
  overflow: hidden; background: var(--bg-raised); border: 1px solid var(--border); }
.paint-stage canvas { display: block; inline-size: 100%; block-size: auto; }
.paint-stage .paint-overlay { position: absolute; inset: 0; inline-size: 100%; block-size: 100%; cursor: crosshair; }
.paint-overlay:focus-visible { outline: 3px solid var(--accent); outline-offset: -3px; }
.fit-canvas { inline-size: 100%; block-size: auto; display: block; max-inline-size: 100%; border-radius: 6px; }
.controls-row { display: flex; flex-wrap: wrap; gap: .6rem; align-items: center; margin: .6rem 0; }
.seg-toggle { display: inline-flex; border: 1px solid var(--border-strong); border-radius: var(--radius); overflow: hidden; }
.seg-toggle button { border: 0; background: var(--bg-raised); padding: .45rem .8rem; cursor: pointer; font: inherit; color: var(--muted); min-block-size: 44px; }
.seg-toggle button[aria-pressed="true"] { background: var(--accent); color: #fff; }
.seg-toggle button:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.param-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 220px), 1fr)); gap: .8rem 1.2rem; margin: .6rem 0; }
.param { display: flex; flex-direction: column; gap: .25rem; min-inline-size: 0; }
.param label { font-size: .8rem; color: var(--muted); }
.two-col { display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start; }
.two-col > * { flex: 1 1 300px; min-inline-size: 0; }
.inside-table { inline-size: 100%; border-collapse: collapse; font-size: .82rem; margin-top: .5rem; }
.inside-table th, .inside-table td { text-align: left; padding: .3rem .5rem; border-bottom: 1px solid var(--border); font-family: var(--font-mono); }
.inside-table th { color: var(--muted); font-weight: 600; }
.privacy-note { font-size: .78rem; color: var(--muted); margin: .35rem 0 0; }
.result-canvas { inline-size: 100%; max-inline-size: 640px; block-size: auto; border-radius: 6px; border: 1px solid var(--border); display: block; }
.fill-canvas { inline-size: 100%; max-inline-size: 300px; block-size: auto; border-radius: 6px; border: 1px solid var(--border); display: block; }
.slow-note { font-size: .78rem; color: var(--muted); margin: .4rem 0 0; }
.fallback { border: 1px solid var(--warn); border-radius: var(--radius); background: var(--bg-raised); padding: .8rem; margin: .5rem 0; font-size: .85rem; }
`;
