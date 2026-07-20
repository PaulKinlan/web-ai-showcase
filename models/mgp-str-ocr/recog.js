// Front-end helpers for the MGP-STR scene-text recognition pages. Owns the worker handshake, turns
// files/samples + a cropped word region into data URLs, and renders the three-granularity + per-char
// "see inside" surface. All inference happens off the main thread in worker.js.

const WORKER_URL = "/web-ai-showcase/models/mgp-str-ocr/worker.js";

export class MgpEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
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

  /** Recognise ONE cropped word. Resolves with
   *  { id, text, fusedScore, granularities:[{name,pred,conf}], steps:[{ch,p}], ms, device }. */
  recognise(imageURL) {
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

/** Crop a normalised region {x,y,w,h} (0..1) of a loaded image to a PNG data URL. MGP-STR is a
 *  single-WORD recogniser, so cropping to one word is how you point it at the text you want. */
export function cropRegion(img, region) {
  const iw = img.naturalWidth || img.width;
  const ih = img.naturalHeight || img.height;
  let { x = 0, y = 0, w = 1, h = 1 } = region || {};
  x = Math.max(0, Math.min(1, x));
  y = Math.max(0, Math.min(1, y));
  w = Math.max(0.02, Math.min(1 - x, w));
  h = Math.max(0.02, Math.min(1 - y, h));
  const sx = Math.round(x * iw), sy = Math.round(y * ih);
  const sw = Math.round(w * iw), sh = Math.round(h * ih);
  const canvas = document.createElement("canvas");
  canvas.width = sw;
  canvas.height = sh;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
  return canvas.toDataURL("image/png");
}

/** A small self-contained crop selector drawn on a <canvas>: shows the image, dims outside the current
 *  selection, and lets the user drag a box around one word (pointer + keyboard-friendly reset). */
export class WordCrop {
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
  region() {
    return { ...this.sel };
  }
  crop() {
    return cropRegion(this.img, this.sel);
  }
  _resize() {
    const maxW = 480;
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

/** Render the three-granularity comparison (char/bpe/wp) into a container: each head's predicted word,
 *  a confidence bar, and a badge on the head the fusion agreed with. Returns accessible text too. */
export function renderGranularities(container, result) {
  const labels = {
    char: "Character",
    bpe: "BPE subword",
    wp: "WordPiece",
  };
  container.replaceChildren();
  for (const g of result.granularities) {
    const row = document.createElement("div");
    row.className = "gran-row";
    const agrees = (g.pred || "").toLowerCase() === (result.text || "").toLowerCase();
    row.innerHTML = `
      <span class="gran-name">${labels[g.name] || g.name}</span>
      <span class="gran-pred${agrees ? " agrees" : ""}">${escapeHTML(g.pred || "—")}${
      agrees ? '<span class="gran-badge" title="the fusion chose this reading">✓ fused</span>' : ""
    }</span>
      <span class="gran-bar" role="img" aria-label="confidence ${
      (g.conf * 100).toFixed(0)
    }%"><i style="inline-size:${(g.conf * 100).toFixed(0)}%"></i></span>
      <span class="gran-conf">${(g.conf * 100).toFixed(0)}%</span>`;
    container.append(row);
  }
}

/** Render the per-character char-head trace: each glyph with a confidence-tinted background. */
export function renderCharTrace(container, steps) {
  container.replaceChildren();
  for (const s of steps) {
    const cell = document.createElement("span");
    cell.className = "char-cell";
    const hue = Math.round(120 * s.p); // red→green with confidence
    cell.style.setProperty("--c", `hsl(${hue} 70% 45%)`);
    cell.innerHTML = `<b>${escapeHTML(s.ch)}</b><small>${(s.p * 100).toFixed(0)}%</small>`;
    container.append(cell);
  }
}

export const MGP_CSS = `
.dropzone {
  border: 2px dashed var(--border-strong); border-radius: var(--radius); background: var(--bg-raised);
  padding: 1rem; text-align: center; cursor: pointer; transition: border-color .15s, background .15s;
}
.dropzone.drag { border-color: var(--accent); background: var(--bg-secondary); }
.dropzone:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.sample-strip { display: flex; gap: .5rem; flex-wrap: wrap; margin: .5rem 0; }
.sample-thumb {
  block-size: 46px; max-inline-size: 150px; object-fit: contain; border-radius: 6px;
  border: 2px solid transparent; cursor: pointer; padding: 2px; background: #fff;
}
.sample-thumb.active { border-color: var(--accent); }
.sample-thumb:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.crop-wrap { position: relative; display: inline-block; max-inline-size: 100%; touch-action: none; }
.crop-wrap canvas { max-inline-size: 100%; block-size: auto; border-radius: 8px; display: block;
  background: #fff; border: 1px solid var(--border); cursor: crosshair; }
.crop-hint { font-size: .75rem; color: var(--muted); margin: .3rem 0; }
.word-out {
  font-family: var(--font-mono); font-size: 1.9rem; font-weight: 600; letter-spacing: .04em;
  padding: .7rem 1rem; border-radius: var(--radius); background: var(--bg-raised);
  border: 1px solid var(--border); min-block-size: 1.4em; margin: .5rem 0; word-break: break-word;
  text-align: center;
}
.word-out.empty { color: var(--muted); font-size: 1rem; font-weight: 400; }
.readout {
  display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem;
}
.readout b { color: var(--color); font-weight: 600; }
.field-row { display: flex; flex-wrap: wrap; gap: .8rem; align-items: start; margin: .6rem 0; }
.field-row > div { min-inline-size: 0; }
.chip {
  font: inherit; font-size: .78rem; padding: .25rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer;
  min-block-size: 34px;
}
.chip:hover { border-color: var(--accent); }
.gran-table { display: flex; flex-direction: column; gap: .45rem; margin: .5rem 0; }
.gran-row {
  display: grid; grid-template-columns: 6.5rem minmax(0, 1fr) 4.5rem 2.6rem; gap: .5rem;
  align-items: center; font-family: var(--font-mono); font-size: .82rem;
}
.gran-name { color: var(--muted); font-size: .72rem; }
.gran-pred { font-weight: 600; word-break: break-word; display: flex; align-items: baseline; gap: .4rem; }
.gran-pred.agrees { color: var(--accent); }
.gran-badge { font-size: .62rem; color: var(--accent); border: 1px solid var(--accent);
  border-radius: 999px; padding: 0 .35rem; font-weight: 600; }
.gran-bar { position: relative; block-size: .55rem; background: var(--bg-secondary);
  border-radius: 999px; overflow: hidden; border: 1px solid var(--border); }
.gran-bar i { position: absolute; inset-block: 0; inset-inline-start: 0; background: var(--accent);
  display: block; }
.gran-conf { text-align: end; color: var(--muted); font-size: .74rem; }
.char-trace { display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .4rem; }
.char-cell {
  display: inline-flex; flex-direction: column; align-items: center; min-inline-size: 1.8rem;
  padding: .25rem .35rem; border-radius: 6px; border: 1px solid var(--border);
  background: var(--bg-raised); border-block-end: 3px solid var(--c, var(--accent));
}
.char-cell b { font-family: var(--font-mono); font-size: 1.05rem; }
.char-cell small { color: var(--muted); font-size: .6rem; }
.inside-table { inline-size: 100%; border-collapse: collapse; font-size: .82rem; margin-top: .5rem;
  table-layout: fixed; }
.inside-table th, .inside-table td {
  text-align: left; padding: .3rem .5rem; border-bottom: 1px solid var(--border);
  font-family: var(--font-mono); vertical-align: top; overflow-wrap: anywhere; word-break: break-word;
}
.inside-table th { color: var(--muted); font-weight: 600; inline-size: 34%; }
.words-out { display: flex; flex-direction: column; gap: .4rem; margin: .5rem 0; }
.word-line { display: flex; gap: .6rem; align-items: baseline; font-family: var(--font-mono);
  font-size: .95rem; padding: .35rem .5rem; border: 1px solid var(--border); border-radius: 6px;
  background: var(--bg-raised); }
.word-line b { color: var(--accent); font-weight: 600; }
.word-line small { color: var(--muted); margin-inline-start: auto; font-size: .72rem; }
`;
