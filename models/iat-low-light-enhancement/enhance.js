// Front-end helpers shared by every IAT low-light page. Keeps pages thin: owns the worker handshake
// (transferring ImageBitmaps so nothing is copied), turns files / gallery images into ImageBitmaps, can
// synthesise an UNDER-EXPOSED input so the demo can show IAT lifting a known-dark image, and renders the
// enhanced result + an accessible before/after reveal slider. ALL inference AND the dense output
// composite live in worker.js (off the main thread, raw ONNX Runtime Web). Privacy by construction: the
// image and every enhanced pixel never leave the device.

const WORKER_URL = "/web-ai-showcase/models/iat-low-light-enhancement/worker.js";

export class EnhanceEngine {
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

  /** Enhance an ImageBitmap (transferred → zero-copy). Returns
   * { enhancedBmp, gainBmp, w, h, imgW, imgH, inLuma, outLuma, inContrast, outContrast, gMin, gMax, ms, infMs, device }. */
  enhance(bitmap, opts) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, bitmap, opts }, [bitmap]);
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

// Synthesise an UNDER-EXPOSED input so the demo can show IAT lifting a known-dark image. `amount` is the
// exposure multiplier's inverse strength: 0 = untouched, 1 = very dark. We multiply pixel values by
// `gain = (1 - 0.85*amount)` and add a touch of gamma so it looks like a real dark photo, not a flat
// dim. Honest by design — the user chooses how dark, and watches IAT recover it; genuinely dark photos
// can be uploaded instead (the bundled "night city" sample is a real low-light scene).
const WORK = 640;
export async function darken(bitmap, { amount = 0.7 } = {}) {
  const long = Math.max(bitmap.width, bitmap.height);
  const s = Math.min(WORK / long, 1);
  const w = Math.max(1, Math.round(bitmap.width * s));
  const h = Math.max(1, Math.round(bitmap.height * s));
  const c = new OffscreenCanvas(w, h);
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, w, h);
  if (amount <= 0) return c.transferToImageBitmap();
  const gain = 1 - 0.85 * amount;
  const gamma = 1 + 0.6 * amount;
  const imgData = ctx.getImageData(0, 0, w, h);
  const d = imgData.data;
  for (let i = 0; i < d.length; i += 4) {
    d[i] = 255 * Math.pow((d[i] / 255) * gain, gamma);
    d[i + 1] = 255 * Math.pow((d[i + 1] / 255) * gain, gamma);
    d[i + 2] = 255 * Math.pow((d[i + 2] / 255) * gain, gamma);
  }
  ctx.putImageData(imgData, 0, 0);
  return c.transferToImageBitmap();
}

// Accessible before/after reveal — two layered canvases clipped by a draggable native range input.
export class RevealCompare {
  constructor(root) {
    this.root = root;
    this.before = root.querySelector(".reveal-before");
    this.after = root.querySelector(".reveal-after");
    this.range = root.querySelector(".reveal-range");
    this.handle = root.querySelector(".reveal-handle");
    this.range.addEventListener("input", () => this._apply());
    this._apply();
  }
  _apply() {
    const pct = +this.range.value;
    this.after.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
    this.handle.style.insetInlineStart = pct + "%";
    this.range.setAttribute("aria-valuetext", `${Math.round(pct)}% enhanced`);
  }
  set(beforeBmp, afterBmp, maxW = 640) {
    drawBitmapFit(this.before, beforeBmp, maxW);
    this.after.width = this.before.width;
    this.after.height = this.before.height;
    this.after.getContext("2d").drawImage(afterBmp, 0, 0, this.after.width, this.after.height);
    this._apply();
  }
}

/** Shared inline styles for the low-light widgets. Structural only — colours from design-system vars. */
export const ENHANCE_CSS = `
.dropzone { border: 2px dashed var(--border-strong); border-radius: var(--radius);
  background: var(--bg-raised); padding: .8rem; text-align: center; cursor: pointer;
  transition: border-color .15s, background .15s; font-size: .85rem; }
.dropzone.drag { border-color: var(--accent); background: var(--bg-secondary); }
.dropzone:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.fit-canvas { inline-size: 100%; block-size: auto; display: block; max-inline-size: 100%; border-radius: 6px; }
.controls-row { display: flex; flex-wrap: wrap; gap: .6rem; align-items: center; margin: .6rem 0; }
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
.gain-canvas { inline-size: 100%; max-inline-size: 300px; block-size: auto; border-radius: 6px; border: 1px solid var(--border); display: block; }
.reveal { position: relative; inline-size: 100%; border-radius: 6px; overflow: hidden; background: var(--bg-raised); touch-action: pan-y; }
.reveal-before, .reveal-after { inline-size: 100%; block-size: auto; display: block; }
.reveal-after { position: absolute; inset: 0; inline-size: 100%; block-size: 100%; }
.reveal-handle { position: absolute; inset-block: 0; inline-size: 2px; background: var(--accent);
  transform: translateX(-1px); pointer-events: none; box-shadow: 0 0 0 1px rgba(0,0,0,.25); }
.reveal-handle::after { content: ""; position: absolute; inset-block-start: 50%; inset-inline-start: 50%;
  inline-size: 26px; block-size: 26px; transform: translate(-50%,-50%); border-radius: 50%;
  background: var(--accent); box-shadow: 0 1px 4px rgba(0,0,0,.4); }
.reveal-range { position: absolute; inset-block-end: .4rem; inset-inline: .4rem; inline-size: calc(100% - .8rem);
  margin: 0; cursor: ew-resize; }
.reveal-range:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
.reveal-labels { display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: .72rem; color: var(--muted); margin-top: .35rem; }
.slow-note { font-size: .78rem; color: var(--muted); margin: .4rem 0 0; }
`;
