// Front-end helpers shared by every UVDoc document-dewarping page. Keeps pages thin: owns the worker
// handshake (transferring ImageBitmaps so nothing is copied), turns files / gallery images into
// ImageBitmaps, can synthesise a CURLED + perspective-warped page so the demo can show UVDoc flattening
// a known distortion, and renders the flattened result + an accessible before/after reveal slider. ALL
// inference AND the dense output composite live in worker.js (off the main thread, raw ONNX Runtime
// Web). Privacy by construction: the document and every corrected pixel never leave the device.

const WORKER_URL = "/web-ai-showcase/models/uvdoc-document-dewarping/worker.js";

export class DewarpEngine {
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

  /** Dewarp an ImageBitmap (transferred → zero-copy). Returns
   * { flatBmp, w, h, imgW, imgH, correction, ms, infMs, device }. */
  dewarp(bitmap, opts) {
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

// Synthesise a CURLED + perspective-warped page so the demo can show UVDoc flattening a known
// distortion. `amount` 0..1 controls the vertical page-curl bow + the horizontal perspective shrink.
// The page is drawn as vertical strips, each bowed and shrunk, over a white margin — like a photo of a
// book page held at an angle. Honest by design — the user chooses the warp and watches UVDoc undo it;
// genuinely warped document photos can be uploaded instead.
const WORK = 512;
export async function warpPage(bitmap, { amount = 0.6 } = {}) {
  const long = Math.max(bitmap.width, bitmap.height);
  const s = Math.min(WORK / long, 1);
  const w = Math.max(1, Math.round(bitmap.width * s));
  const h = Math.max(1, Math.round(bitmap.height * s));
  const c = new OffscreenCanvas(w, h);
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
  if (amount <= 0) {
    ctx.drawImage(bitmap, 0, 0, w, h);
    return c.transferToImageBitmap();
  }
  const src = await createImageBitmap(bitmap);
  const strips = 48;
  const bowMax = h * 0.16 * amount; // vertical bow (page curl)
  const perspMax = 0.28 * amount; // horizontal perspective shrink toward the right edge
  for (let i = 0; i < strips; i++) {
    const sx = i * w / strips;
    const sw = w / strips + 1;
    const t = i / strips;
    const bow = Math.sin(t * Math.PI) * bowMax;
    const persp = 1 - perspMax * t;
    const dh = h * persp;
    const dy = (h - dh) / 2 - bow;
    ctx.drawImage(src, sx, 0, sw, h, sx, dy, sw, dh);
  }
  src.close?.();
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
    this.range.setAttribute("aria-valuetext", `${Math.round(pct)}% flattened`);
  }
  set(beforeBmp, afterBmp, maxW = 640) {
    drawBitmapFit(this.before, beforeBmp, maxW);
    this.after.width = this.before.width;
    this.after.height = this.before.height;
    this.after.getContext("2d").drawImage(afterBmp, 0, 0, this.after.width, this.after.height);
    this._apply();
  }
}

/** Shared inline styles for the dewarping widgets. Structural only — colours from design-system vars. */
export const DEWARP_CSS = `
.dropzone { border: 2px dashed var(--border-strong); border-radius: var(--radius);
  background: var(--bg-raised); padding: .8rem; text-align: center; cursor: pointer;
  transition: border-color .15s, background .15s; font-size: .85rem; }
.dropzone.drag { border-color: var(--accent); background: var(--bg-secondary); }
.dropzone:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.canvas-frame { position: relative; inline-size: 100%; background: var(--bg-raised);
  border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; padding: .4rem; }
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
