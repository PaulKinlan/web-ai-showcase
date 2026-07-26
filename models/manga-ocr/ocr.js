// Shared front-end helpers for the Manga OCR pages. Keeps each page thin: owns the worker handshake,
// turns files/samples/camera frames into data URLs, streams recognised characters back to the page, and
// renders the crop selector + "see inside" token trace. All inference is in worker.js (raw-ORT sessions),
// off the main thread.

const WORKER_URL = "/web-ai-showcase/models/manga-ocr/worker.js";

export class MangaOcrEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.onProgress = null;
    this.onToken = null;
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
    } else if (msg.type === "token") {
      this._pending.get(msg.id)?.onToken?.(msg);
      this.onToken?.(msg);
    } else if (msg.type === "result" || msg.type === "cancelled") {
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

  /**
   * Recognise Japanese text in an image (data/URL string). Streams characters via
   * onToken({token, prob, alternatives, t, i}); resolves with
   * { text, raw, tokens, ms, encMs, decMs, device, truncated }.
   */
  recognize(imageURL, maxTokens = 128, onToken) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject, onToken });
      this.worker.postMessage({ type: "run", id, image: imageURL, maxTokens });
    });
  }

  /** Ask the worker to stop decoding after the current step (resolves as `cancelled`). */
  cancel() {
    this.worker.postMessage({ type: "cancel" });
  }

  /** Release the model from memory: terminate the worker (frees its WASM heap + both ONNX sessions)
   *  and reject anything in flight. The engine is single-use after this — build a fresh one to reload. */
  dispose() {
    const err = new Error("Model released from memory");
    for (const w of this._loadWaiters) w.reject(err);
    this._loadWaiters = [];
    for (const [, p] of this._pending) p.reject(err);
    this._pending.clear();
    this.ready = false;
    this.worker.terminate();
  }
}

/** Read a File (upload, drop, paste, or a captured camera frame) into a data URL. */
export function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}

/**
 * Crop a normalised region {x,y,w,h} (0..1) of an already-loaded HTMLImageElement to a PNG data URL.
 * Manga OCR reads anything from a single line to a whole speech bubble, so the crop is how you point
 * it at exactly the text you want. Returns the whole image when the region covers (nearly) everything.
 */
export function cropRegion(img, region, maxSize = Infinity) {
  const iw = img.naturalWidth || img.width;
  const ih = img.naturalHeight || img.height;
  let { x = 0, y = 0, w = 1, h = 1 } = region || {};
  x = Math.max(0, Math.min(1, x));
  y = Math.max(0, Math.min(1, y));
  w = Math.max(0.02, Math.min(1 - x, w));
  h = Math.max(0.02, Math.min(1 - y, h));
  const sx = Math.round(x * iw), sy = Math.round(y * ih);
  const sw = Math.round(w * iw), sh = Math.round(h * ih);
  const scale = Math.min(1, maxSize / sw, maxSize / sh);
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(sw * scale));
  canvas.height = Math.max(1, Math.round(sh * scale));
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/png");
}

/**
 * A small self-contained crop selector drawn on a <canvas>. Shows the image, dims everything outside
 * the current selection, and lets the user drag a new rectangle. `region()` returns the normalised
 * {x,y,w,h}. Keeps the demo honest: what you crop is what the model sees.
 */
export class CropCanvas {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.img = new Image();
    this.sel = { x: 0, y: 0, w: 1, h: 1 };
    this._drag = null;
    this.onChange = null;
    this.img.addEventListener("load", () => {
      this.sel = { x: 0, y: 0, w: 1, h: 1 };
      this._resize();
      this.draw();
      this.onChange?.();
    });
    const toNorm = (ev) => {
      const r = canvas.getBoundingClientRect();
      return {
        x: Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width)),
        y: Math.max(0, Math.min(1, (ev.clientY - r.top) / r.height)),
      };
    };
    canvas.addEventListener("pointerdown", (ev) => {
      if (!this.img.naturalWidth) return;
      canvas.setPointerCapture(ev.pointerId);
      this._drag = toNorm(ev);
      this.sel = { x: this._drag.x, y: this._drag.y, w: 0, h: 0 };
      this.draw();
    });
    canvas.addEventListener("pointermove", (ev) => {
      if (!this._drag) return;
      const p = toNorm(ev);
      this.sel = {
        x: Math.min(this._drag.x, p.x),
        y: Math.min(this._drag.y, p.y),
        w: Math.abs(p.x - this._drag.x),
        h: Math.abs(p.y - this._drag.y),
      };
      this.draw();
    });
    const end = () => {
      if (!this._drag) return;
      this._drag = null;
      if (this.sel.w < 0.02 || this.sel.h < 0.02) this.sel = { x: 0, y: 0, w: 1, h: 1 };
      this.draw();
      this.onChange?.();
    };
    canvas.addEventListener("pointerup", end);
    canvas.addEventListener("pointercancel", end);
  }

  setSrc(url) {
    this.img.src = url;
  }
  reset() {
    this.sel = { x: 0, y: 0, w: 1, h: 1 };
    this.draw();
    this.onChange?.();
  }
  get hasImage() {
    return this.img.naturalWidth > 0;
  }
  region() {
    return { ...this.sel };
  }
  crop() {
    return cropRegion(this.img, this.sel);
  }
  cropDataURL(maxSize = Infinity) {
    return cropRegion(this.img, this.sel, maxSize);
  }
  _resize() {
    const maxW = 560;
    const iw = this.img.naturalWidth || 1, ih = this.img.naturalHeight || 1;
    const scale = Math.min(1, maxW / iw);
    this.canvas.width = Math.round(iw * scale);
    this.canvas.height = Math.round(ih * scale);
  }
  draw() {
    const { ctx, canvas } = this;
    if (!this.img.naturalWidth) return;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(this.img, 0, 0, W, H);
    const s = this.sel;
    if (s.w < 0.999 || s.h < 0.999) {
      ctx.fillStyle = "rgba(20,20,30,.45)";
      ctx.fillRect(0, 0, W, H);
      const rx = s.x * W, ry = s.y * H, rw = s.w * W, rh = s.h * H;
      ctx.clearRect(rx, ry, rw, rh);
      ctx.drawImage(
        this.img,
        s.x * this.img.naturalWidth,
        s.y * this.img.naturalHeight,
        s.w * this.img.naturalWidth,
        s.h * this.img.naturalHeight,
        rx,
        ry,
        rw,
        rh,
      );
      ctx.strokeStyle = "#6366f1";
      ctx.lineWidth = 2;
      ctx.strokeRect(rx + 1, ry + 1, rw - 2, rh - 2);
    }
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/**
 * Render the "see inside" token trace: one chip per decoded character with its model probability,
 * plus the top alternatives the model weighed at that step. All values are the real per-step
 * softmax probabilities streamed from the worker.
 */
export function renderTokenChip(container, tok) {
  const chip = document.createElement("span");
  chip.className = "tok";
  chip.title = `step ${tok.i + 1} · ${(tok.prob * 100).toFixed(1)}%` +
    (tok.alternatives?.length
      ? ` · also: ${
        tok.alternatives.map((a) => `${a.token || "␣"} ${(a.p * 100).toFixed(1)}%`).join(", ")
      }`
      : "");
  const label = tok.token || "⟨end⟩";
  chip.innerHTML = `${escapeHTML(label)}<small>${(tok.prob * 100).toFixed(0)}%</small>`;
  // Colour by confidence: accent when the model is sure, muted when it is guessing.
  chip.classList.toggle("tok-unsure", tok.prob < 0.6);
  container.append(chip);
}

/** Shared inline styles for the Manga OCR widgets (dropzone, crop overlay, transcript, token trace). */
export const OCR_CSS = `
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
.sample-strip { display: flex; gap: .5rem; flex-wrap: wrap; margin: .5rem 0; justify-content: center; }
.sample-thumb {
  block-size: 52px; max-inline-size: 160px; object-fit: contain; border-radius: 6px;
  border: 2px solid transparent; cursor: pointer; padding: 2px; background: #fff;
}
.sample-thumb img { max-inline-size: 100%; block-size: 100%; object-fit: contain; }
.sample-thumb.active { border-color: var(--accent); }
.sample-thumb:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.crop-wrap { position: relative; display: inline-block; max-inline-size: 100%; touch-action: none; }
.crop-wrap canvas { max-inline-size: 100%; block-size: auto; border-radius: 8px; display: block;
  background: #fff; border: 1px solid var(--border); cursor: crosshair; }
.crop-hint { font-size: .75rem; color: var(--muted); margin: .3rem 0; }
.text-box {
  font-size: 1.3rem; line-height: 1.5; letter-spacing: .02em;
  padding: .8rem 1rem; border-radius: var(--radius); background: var(--bg-raised);
  border: 1px solid var(--border); min-block-size: 2.6em; margin: .5rem 0; word-break: break-word;
}
.text-box .cursor { display: inline-block; inline-size: .5ch; background: var(--accent);
  animation: ocrblink 1s steps(2) infinite; }
@keyframes ocrblink { 50% { opacity: 0; } }
@media (prefers-reduced-motion: reduce) { .text-box .cursor { animation: none; } }
.readout {
  display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem;
}
.readout b { color: var(--color); font-weight: 600; }
.token-trace { display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .4rem; }
.token-trace .tok {
  font-size: .82rem; padding: .15rem .45rem; border-radius: 4px;
  background: var(--bg-secondary); border: 1px solid var(--border); white-space: pre;
}
.token-trace .tok small { color: var(--muted); margin-inline-start: .3rem; font-size: .68rem; }
.token-trace .tok.tok-unsure { border-color: var(--warn, #b45309); }
.inside-table { inline-size: 100%; border-collapse: collapse; font-size: .82rem; margin-top: .5rem; }
.inside-table th, .inside-table td {
  text-align: left; padding: .3rem .5rem; border-bottom: 1px solid var(--border);
  font-family: var(--font-mono);
}
.inside-table th { color: var(--muted); font-weight: 600; }
.field-row { display: flex; flex-wrap: wrap; gap: .6rem; align-items: end; margin: .6rem 0; }
.field-row label { display: flex; flex-direction: column; gap: .25rem; font-size: .82rem; flex: 1 1 200px; }
.chip {
  font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer;
}
.chip:hover { border-color: var(--accent); }
.chip[aria-pressed="true"] { border-color: var(--accent); background: var(--bg-secondary); }
`;
