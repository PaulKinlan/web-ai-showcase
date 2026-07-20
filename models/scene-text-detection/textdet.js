// Front-end helpers shared by every Scene-Text-Detection page. Keeps pages thin: owns the worker
// handshake (transferring ImageBitmaps so nothing is copied), turns files / samples / camera frames
// into ImageBitmaps, and renders the detected region polygons + the probability heatmap. ALL inference
// and DB post-processing live in worker.js (off the main thread, raw ONNX Runtime Web). Privacy by
// construction: the image, the probability map, and the boxes never leave the device. This LOCATES
// text regions — it does not READ them (that's OCR; see the multi-model demo).

const WORKER_URL = "/web-ai-showcase/models/scene-text-detection/worker.js";

export class TextDetEngine {
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

  /** Detect text regions in an ImageBitmap (transferred → zero-copy). Returns
   * { regions, heat:Uint8ClampedArray, mapW, mapH, imgW, imgH, ms, infMs, device }. */
  detect(bitmap, opts) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, bitmap, opts }, [bitmap]);
    });
  }
}

/** Read a File / Blob into an ImageBitmap for the worker. */
export function toBitmap(source) {
  return createImageBitmap(source);
}

/** Fetch a same-origin sample and decode it to an ImageBitmap. */
export async function urlToBitmap(src) {
  return createImageBitmap(await (await fetch(src)).blob());
}

/** Draw an ImageBitmap into a canvas, sizing the canvas to the image and returning the draw scale so
 * boxes (in image pixels) map to canvas pixels. Fits within maxW while keeping aspect. */
export function drawImageFit(canvas, bitmap, maxW = 640) {
  const scale = Math.min(maxW / bitmap.width, 1);
  const cw = Math.round(bitmap.width * scale), ch = Math.round(bitmap.height * scale);
  canvas.width = cw;
  canvas.height = ch;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, cw, ch);
  ctx.drawImage(bitmap, 0, 0, cw, ch);
  return { scale, cw, ch };
}

/** Draw detected region polygons (in image pixels) onto a canvas already showing the image at `scale`.
 * Oriented polygons hug slanted text; each is numbered so the text list can reference it. */
export function drawRegions(canvas, regions, scale, { showIndex = true } = {}) {
  const ctx = canvas.getContext("2d");
  ctx.lineWidth = 2;
  ctx.font = "600 13px system-ui, sans-serif";
  regions.forEach((r, i) => {
    const pts = r.points;
    ctx.beginPath();
    ctx.moveTo(pts[0].x * scale, pts[0].y * scale);
    for (let k = 1; k < pts.length; k++) ctx.lineTo(pts[k].x * scale, pts[k].y * scale);
    ctx.closePath();
    ctx.strokeStyle = "rgba(75,58,255,0.95)";
    ctx.fillStyle = "rgba(75,58,255,0.14)";
    ctx.fill();
    ctx.stroke();
    if (showIndex) {
      const bx = r.box.x0 * scale, by = r.box.y0 * scale;
      const label = String(i + 1);
      ctx.fillStyle = "rgba(75,58,255,0.95)";
      const tw = ctx.measureText(label).width + 8;
      ctx.fillRect(bx, Math.max(0, by - 16), tw, 16);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, bx + 4, Math.max(11, by - 4));
    }
  });
}

/** Paint the raw probability map (Uint8, mapW×mapH) into a canvas as an indigo heatmap — the model's
 * actual per-pixel "is this text?" belief, before the box decode. */
export function drawHeatmap(canvas, heat, mapW, mapH, displayW = 320) {
  const scale = displayW / mapW;
  canvas.width = Math.round(mapW * scale);
  canvas.height = Math.round(mapH * scale);
  const off = new OffscreenCanvas(mapW, mapH);
  const octx = off.getContext("2d");
  const img = octx.createImageData(mapW, mapH);
  for (let i = 0; i < heat.length; i++) {
    const v = heat[i];
    img.data[i * 4] = 30 + (v * (75 - 30)) / 255; // toward indigo
    img.data[i * 4 + 1] = 20 + (v * (58 - 20)) / 255;
    img.data[i * 4 + 2] = 40 + (v * (255 - 40)) / 255;
    img.data[i * 4 + 3] = 40 + (v * (215)) / 255; // brighter where text-prob is high
  }
  octx.putImageData(img, 0, 0);
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.fillStyle = "#0d0f18";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
}

/** Redact (block out) detected regions on a copy of the image — the practical privacy use case. */
export function drawRedacted(canvas, bitmap, regions, scale, cw, ch) {
  canvas.width = cw;
  canvas.height = ch;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0, cw, ch);
  ctx.fillStyle = "#111";
  for (const r of regions) {
    ctx.beginPath();
    const pts = r.points;
    ctx.moveTo(pts[0].x * scale, pts[0].y * scale);
    for (let k = 1; k < pts.length; k++) ctx.lineTo(pts[k].x * scale, pts[k].y * scale);
    ctx.closePath();
    ctx.fill();
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Shared inline styles for the scene-text-detection widgets. Injected once per page. */
export const TEXTDET_CSS = `
.dropzone { border: 2px dashed var(--border-strong); border-radius: var(--radius);
  background: var(--bg-raised); padding: .8rem; text-align: center; cursor: pointer;
  transition: border-color .15s, background .15s; font-size: .85rem; }
.dropzone.drag { border-color: var(--accent); background: var(--bg-secondary); }
.dropzone:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.sample-strip { display: flex; gap: .5rem; flex-wrap: wrap; margin: .5rem 0; }
.sample-thumb { inline-size: 72px; block-size: 54px; object-fit: cover; border-radius: 8px;
  border: 2px solid transparent; cursor: pointer; padding: 0; background: var(--bg-raised); }
.sample-thumb.active { border-color: var(--accent); }
.sample-thumb:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.canvas-frame { position: relative; inline-size: 100%; background: var(--bg-raised);
  border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; padding: .4rem; }
.det-canvas { inline-size: 100%; block-size: auto; display: block; max-inline-size: 100%; border-radius: 6px; }
.controls-row { display: flex; flex-wrap: wrap; gap: .6rem; align-items: center; margin: .6rem 0; }
.big-verdict { font-size: 1.35rem; font-weight: 600; margin: .3rem 0; display: flex; gap: .5rem; align-items: baseline; flex-wrap: wrap; }
.count-num { font-family: var(--font-mono); font-size: 1.6rem; font-weight: 700; color: var(--accent); }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.region-list { list-style: none; padding: 0; margin: .6rem 0; display: flex; flex-direction: column; gap: .3rem; max-block-size: 260px; overflow: auto; }
.region-list li { display: grid; grid-template-columns: 2.2rem 1fr auto; gap: .5rem; align-items: center;
  font-family: var(--font-mono); font-size: .78rem; padding: .25rem .4rem; border: 1px solid var(--border); border-radius: 6px; }
.region-list .r-idx { background: var(--accent); color: var(--accent-ink, #fff); border-radius: 4px; text-align: center; font-weight: 700; }
.region-list .r-meta { color: var(--muted); }
.param-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 220px), 1fr)); gap: .8rem 1.2rem; margin: .6rem 0; }
.param { display: flex; flex-direction: column; gap: .25rem; min-inline-size: 0; }
.param label { font-size: .8rem; color: var(--muted); }
.two-col { display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start; }
.two-col > * { flex: 1 1 300px; min-inline-size: 0; }
.inside-table { inline-size: 100%; border-collapse: collapse; font-size: .82rem; margin-top: .5rem; }
.inside-table th, .inside-table td { text-align: left; padding: .3rem .5rem; border-bottom: 1px solid var(--border); font-family: var(--font-mono); }
.inside-table th { color: var(--muted); font-weight: 600; }
.privacy-note { font-size: .78rem; color: var(--muted); margin: .35rem 0 0; }
.fallback { border: 1px solid var(--warn); border-radius: var(--radius); background: var(--bg-raised); padding: .8rem; margin: .5rem 0; font-size: .85rem; }
.heat-canvas { inline-size: 100%; max-inline-size: 340px; block-size: auto; border-radius: 6px; border: 1px solid var(--border); display: block; }
`;
