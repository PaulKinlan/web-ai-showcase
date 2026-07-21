// Front-end helpers for the QuickDraw sketch page: the worker handshake, canvas → 28x28 grayscale
// preprocessing, and prediction rendering. All inference lives in worker.js (off the main thread).

export const SIZE = 28;

export class SketchEngine {
  constructor() {
    this.worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
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
        for (const w of this._loadWaiters) w.reject(new Error(msg.message));
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
  /** Classify a 28x28 grayscale sketch (Float32Array 784) → { top:[{label,prob}], ms }. */
  classify(gray, topK = 5) {
    const id = ++this._id;
    const g = new Float32Array(gray);
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "classify", id, gray: g, topK }, [g.buffer]);
    });
  }
}

/**
 * Downscale a drawing canvas (white strokes on black) to a 28x28 grayscale Float32Array (0-1). The sketch
 * is first tight-cropped to its ink bounding box and centred, so position/scale don't matter — matching how
 * QuickDraw sketches are normalised. Returns null if the canvas is empty.
 */
export function canvasToGray28(canvas) {
  const w = canvas.width, h = canvas.height;
  const ctx = canvas.getContext("2d");
  const px = ctx.getImageData(0, 0, w, h).data;
  // bounding box of non-black pixels
  let minX = w, minY = h, maxX = -1, maxY = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (px[(y * w + x) * 4] > 20) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null; // empty
  const bw = maxX - minX + 1, bh = maxY - minY + 1;
  const side = Math.max(bw, bh);
  const pad = Math.round(side * 0.18); // QuickDraw sketches sit with a small margin
  const box = side + pad * 2;
  const off = new OffscreenCanvas(SIZE, SIZE);
  const octx = off.getContext("2d");
  octx.fillStyle = "black";
  octx.fillRect(0, 0, SIZE, SIZE);
  octx.imageSmoothingEnabled = true;
  const scale = SIZE / box;
  const dx = (box - bw) / 2, dy = (box - bh) / 2;
  octx.drawImage(canvas, minX, minY, bw, bh, dx * scale, dy * scale, bw * scale, bh * scale);
  const im = octx.getImageData(0, 0, SIZE, SIZE).data;
  const a = new Float32Array(SIZE * SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    a[i] = (im[i * 4] * 0.299 + im[i * 4 + 1] * 0.587 + im[i * 4 + 2] * 0.114) / 255;
  }
  return a;
}

export function escapeHTML(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]),
  );
}

/** Render top-k predictions as labelled probability bars. */
export function renderPredictions(container, top) {
  container.innerHTML = top.map((t, i) => {
    const pct = Math.round(t.prob * 100);
    return `<div class="qd-bar${i === 0 ? " top" : ""}"><span class="qd-name">${
      escapeHTML(t.label)
    }</span>` +
      `<span class="qd-track"><i style="width:${Math.max(2, pct)}%"></i></span>` +
      `<span class="qd-pct">${pct}%</span></div>`;
  }).join("");
}

export const DRAW_CSS = `
  .qd-wrap { display: flex; flex-wrap: wrap; gap: 1.2rem; align-items: flex-start; margin: 0.6rem 0; }
  .qd-pad { flex: none; }
  #pad { width: 280px; height: 280px; max-width: 78vw; background: #0b0f14; border-radius: 12px; touch-action: none; cursor: crosshair; display: block; }
  .qd-tools { display: flex; gap: 0.5rem; margin: 0.5rem 0; }
  .qd-preds { flex: 1; min-width: 15rem; }
  .qd-guess { font-size: 1.3rem; font-weight: 700; margin: 0 0 0.5rem; min-height: 1.6rem; }
  .qd-bars { display: flex; flex-direction: column; gap: 0.25rem; max-width: 24rem; }
  .qd-bar { display: grid; grid-template-columns: 8rem 1fr 2.4rem; align-items: center; gap: 0.4rem; font-size: 0.85rem; }
  .qd-bar.top .qd-name { font-weight: 700; }
  .qd-name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .qd-track { height: 8px; border-radius: 4px; background: #7772; overflow: hidden; }
  .qd-track > i { display: block; height: 100%; background: linear-gradient(90deg, #2bb59a, #4ac6e0); }
  .qd-bar.top .qd-track > i { background: linear-gradient(90deg, #2bb59a, #7fe8d2); }
  .qd-pct { font-family: var(--font-mono, monospace); text-align: right; }
  .qd-hint { font-size: 0.85rem; color: var(--muted, #888); margin: 0.3rem 0; }
  .qd-status { font-family: var(--font-mono, monospace); font-size: 0.9rem; margin: 0.4rem 0; }
`;
