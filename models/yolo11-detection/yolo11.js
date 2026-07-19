// Front-end helpers for the YOLO11 pages. Thin: owns the worker handshake (transferring ImageBitmaps
// so webcam frames are zero-copy), runs the CLIENT-SIDE Non-Max-Suppression that YOLO11 needs (this is
// the whole point of the demo — v10 is NMS-free, v11 is not), draws the box overlay, and renders tables.
// All inference + letterbox + dense-head decode lives in worker.js (off the main thread, raw ORT-Web).

const WORKER_URL = "/web-ai-showcase/models/yolo11-detection/worker.js";

export class Yolo11Engine {
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

  /** Detect on an ImageBitmap (transferred → zero-copy). Returns {rawDetections, ms, device, imgW,
   * imgH}. rawDetections are the pre-NMS candidates above the floor — call nms()/filterAndNms() next. */
  detect(bitmap) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, bitmap }, [bitmap]);
    });
  }
}

/** Intersection-over-union of two {xmin,ymin,xmax,ymax} boxes. */
export function iou(a, b) {
  const ix = Math.max(0, Math.min(a.xmax, b.xmax) - Math.max(a.xmin, b.xmin));
  const iy = Math.max(0, Math.min(a.ymax, b.ymax) - Math.max(a.ymin, b.ymin));
  const inter = ix * iy;
  const areaA = Math.max(0, a.xmax - a.xmin) * Math.max(0, a.ymax - a.ymin);
  const areaB = Math.max(0, b.xmax - b.xmin) * Math.max(0, b.ymax - b.ymin);
  const union = areaA + areaB - inter;
  return union <= 0 ? 0 : inter / union;
}

/**
 * Class-aware greedy Non-Max-Suppression — the step YOLOv10 removed and YOLO11 still needs. Input
 * candidates should be score-sorted (the worker sorts them). Boxes of the SAME class overlapping more
 * than `iouThresh` are suppressed by the higher-scoring one. Returns { kept, suppressed } counts + list.
 */
export function nms(candidates, iouThresh = 0.45) {
  const sorted = [...candidates].sort((p, q) => q.score - p.score);
  const removed = new Array(sorted.length).fill(false);
  const kept = [];
  for (let i = 0; i < sorted.length; i++) {
    if (removed[i]) continue;
    kept.push(sorted[i]);
    for (let j = i + 1; j < sorted.length; j++) {
      if (removed[j]) continue;
      if (
        sorted[j].classId === sorted[i].classId && iou(sorted[i].box, sorted[j].box) > iouThresh
      ) {
        removed[j] = true;
      }
    }
  }
  return { kept, suppressed: sorted.length - kept.length };
}

/** Filter raw candidates by score, then run class-aware NMS. Returns { kept, before, suppressed }. */
export function filterAndNms(rawDetections, scoreThresh, iouThresh) {
  const above = rawDetections.filter((d) => d.score >= scoreThresh);
  const { kept, suppressed } = nms(above, iouThresh);
  return { kept, before: above.length, suppressed };
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// Saturated, all dark enough that white label text passes WCAG AA on top of them.
export const BOX_PALETTE = [
  "#b3261e",
  "#1565c0",
  "#1a7a3a",
  "#6a1b9a",
  "#a15c00",
  "#00695c",
  "#ad1457",
  "#37474f",
];

const _labelColors = new Map();
export function colorForLabel(label) {
  if (!_labelColors.has(label)) {
    _labelColors.set(label, BOX_PALETTE[_labelColors.size % BOX_PALETTE.length]);
  }
  return _labelColors.get(label);
}

/**
 * Draw an image/video frame + its detection boxes onto `canvas` at the source's natural resolution.
 * CSS scales the canvas down responsively, so boxes stay pixel-accurate. `opts.ghost` draws faint
 * boxes underneath (used to show the suppressed duplicates behind the kept ones).
 */
export function drawDetections(canvas, source, detections, opts = {}) {
  const w = source.naturalWidth || source.videoWidth || source.width;
  const h = source.naturalHeight || source.videoHeight || source.height;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(source, 0, 0, w, h);

  const lw = Math.max(2, w / 320);

  // Faint "ghost" boxes for suppressed duplicates, drawn first, underneath.
  if (opts.ghost && opts.ghost.length) {
    ctx.save();
    ctx.globalAlpha = 0.28;
    ctx.setLineDash([Math.max(4, w / 160), Math.max(4, w / 160)]);
    for (const d of opts.ghost) {
      const { xmin, ymin, xmax, ymax } = d.box;
      ctx.lineWidth = Math.max(1, lw * 0.6);
      ctx.strokeStyle = colorForLabel(d.label);
      ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);
    }
    ctx.restore();
  }

  const fontPx = Math.max(13, Math.round(w / 42));
  ctx.font = `600 ${fontPx}px "Avenir Next", "Segoe UI", system-ui, sans-serif`;
  ctx.textBaseline = "middle";
  const highlight = opts.highlightIndex;

  detections.forEach((d, i) => {
    const { xmin, ymin, xmax, ymax } = d.box;
    const color = colorForLabel(d.label);
    ctx.lineWidth = i === highlight ? lw * 2 : lw;
    ctx.strokeStyle = color;
    ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);

    const text = `${d.label} ${(d.score * 100).toFixed(0)}%`;
    const padX = fontPx * 0.35;
    const th = fontPx * 1.35;
    const tw = ctx.measureText(text).width + padX * 2;
    const ly = Math.max(0, ymin - th);
    ctx.fillStyle = color;
    ctx.fillRect(xmin - lw / 2, ly, tw, th);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, xmin - lw / 2 + padX, ly + th / 2);
  });
  return detections;
}

/** Count detections by label → [{label, count}], busiest first. */
export function countByLabel(detections) {
  const m = new Map();
  for (const d of detections) m.set(d.label, (m.get(d.label) ?? 0) + 1);
  return [...m.entries()].map(([label, count]) => ({ label, count }))
    .sort((a, b) => b.count - a.count);
}

export const YOLO11_CSS = `
.dropzone {
  border: 2px dashed var(--border-strong); border-radius: var(--radius); background: var(--bg-raised);
  padding: 1rem; text-align: center; cursor: pointer; transition: border-color .15s, background .15s;
}
.dropzone.drag { border-color: var(--accent); background: var(--bg-secondary); }
.dropzone:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.sample-strip { display: flex; gap: .5rem; flex-wrap: wrap; margin: .5rem 0; }
.sample-thumb {
  inline-size: 76px; block-size: 56px; object-fit: cover; border-radius: 6px;
  border: 2px solid transparent; cursor: pointer; padding: 0;
}
.sample-thumb.active { border-color: var(--accent); }
.sample-thumb:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.canvas-wrap {
  position: relative; display: block; margin-top: .5rem; border-radius: 8px; overflow: hidden;
  background: var(--bg-raised); border: 1px solid var(--border);
}
.stage-canvas { display: block; inline-size: 100%; block-size: auto; max-block-size: 62vh; object-fit: contain; }
.stage-canvas:focus-visible { outline: 3px solid var(--accent); outline-offset: -3px; }
.slider-row { display: flex; align-items: center; gap: .6rem; margin: .6rem 0; flex-wrap: wrap; }
.slider-row input[type=range] { flex: 1 1 180px; accent-color: var(--accent); }
.slider-row output { font-family: var(--font-mono); font-size: .82rem; min-inline-size: 3ch; }
.readout {
  display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem;
}
.readout b { color: var(--color); font-weight: 600; }
.count-chips { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0; }
.count-chip {
  display: inline-flex; align-items: center; gap: .35rem; font-size: .82rem;
  padding: .15rem .55rem; border-radius: 999px; border: 1px solid var(--border); background: var(--bg-raised);
}
.count-chip .swatch { inline-size: .7rem; block-size: .7rem; border-radius: 3px; }
.count-chip b { font-family: var(--font-mono); }
.nms-stat { display: flex; flex-wrap: wrap; gap: .5rem; margin: .5rem 0; }
.nms-pill {
  font-family: var(--font-mono); font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised);
}
.nms-pill b { color: var(--accent); }
.inside-table { inline-size: 100%; border-collapse: collapse; font-size: .82rem; margin-top: .5rem; }
.inside-table th, .inside-table td {
  text-align: left; padding: .3rem .5rem; border-bottom: 1px solid var(--border); font-family: var(--font-mono);
}
.inside-table th { color: var(--muted); font-weight: 600; }
.obj-list { display: flex; flex-direction: column; gap: .3rem; margin: .5rem 0; }
.obj-btn {
  display: flex; align-items: center; gap: .5rem; text-align: left; inline-size: 100%;
  background: var(--bg-raised); color: var(--color); border: 1px solid var(--border);
  border-radius: 8px; padding: .5rem .6rem; font-size: .85rem; cursor: pointer; min-block-size: 44px;
}
.obj-btn:hover, .obj-btn:focus-visible { border-color: var(--accent); }
.obj-btn[aria-pressed=true] { border-color: var(--accent); background: var(--bg-secondary); }
.obj-btn .swatch { inline-size: .8rem; block-size: .8rem; border-radius: 3px; flex: none; }
.obj-btn .sc { margin-inline-start: auto; font-family: var(--font-mono); color: var(--muted); }
.fallback { border: 1px solid var(--warn, #a15c00); border-radius: var(--radius); background: var(--bg-raised); padding: 1rem; }
.fallback code { background: var(--bg-secondary); padding: .05rem .3rem; border-radius: 4px; }
.crop-out { display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start; margin-top: .6rem; }
.crop-out canvas { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised); max-inline-size: 220px; block-size: auto; }
.pred-row2 { display:grid; grid-template-columns: 1fr 3.4rem; gap:.5rem; align-items:center; margin:.2rem 0; }
.pred-row2 .w { font-family:var(--font-mono); font-size:.82rem; }
.pred-row2 .barw { grid-column:1/3; block-size:.6rem; border-radius:999px; background:var(--bg-secondary); border:1px solid var(--border); overflow:hidden; }
.pred-row2 .barf { display:block; block-size:100%; background:var(--accent); }
`;
