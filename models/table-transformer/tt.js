// Shared front-end helpers for the Table Transformer pages. Keeps each page thin: it owns the worker
// handshakes (one worker per stage), image/crop helpers, the canvas overlays for detection and
// structure, grid reconstruction from row/column boxes, and the injected CSS. ALL inference lives in the
// workers (off the main thread). Two models compose a two-stage pipeline:
//   stage 1  worker.js            — WHERE are the tables            (detection)
//   stage 2  structure-worker.js  — rows / columns / header / cells (structure recognition)

const BASE = "/web-ai-showcase/models/table-transformer";

/** Generic client for our detection-style workers (both stages share one message protocol). */
export class DetectionClient {
  constructor(workerUrl) {
    this.worker = new Worker(workerUrl, { type: "module" });
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

  _run(image) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, image });
    });
  }
}

/** Stage 1 — table detection. detect(url) → { detections:[{label,score,box}], ms, device } */
export class TableDetector extends DetectionClient {
  constructor() {
    super(`${BASE}/worker.js`);
  }
  detect(imageURL) {
    return this._run(imageURL);
  }
}

/** Stage 2 — structure recognition on a cropped table. recognize(url) → { cells:[…], ms, device } */
export class TableStructure extends DetectionClient {
  constructor() {
    super(`${BASE}/structure-worker.js`);
  }
  recognize(imageURL) {
    return this._run(imageURL);
  }
}

/** Read a File (upload/drop) into a data URL usable by a worker. */
export function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}

/** Decode a URL into an HTMLImageElement for drawing. */
export function decodeImage(url) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error("Could not decode image"));
    im.src = url;
  });
}

/**
 * Crop a detected table box out of the source image, with padding (the structure model was trained on
 * table crops that include a little margin). Returns { url, sx, sy, sw, sh } — the crop's data URL plus
 * where it sits in the source, so overlays can be mapped back onto the whole document.
 */
export function cropTable(img, box, pad = 0.06) {
  const iw = img.naturalWidth || img.width;
  const ih = img.naturalHeight || img.height;
  const bw = box.xmax - box.xmin, bh = box.ymax - box.ymin;
  const sx = Math.max(0, Math.round(box.xmin - bw * pad));
  const sy = Math.max(0, Math.round(box.ymin - bh * pad));
  const sw = Math.min(iw - sx, Math.round(bw * (1 + 2 * pad)));
  const sh = Math.min(ih - sy, Math.round(bh * (1 + 2 * pad)));
  const cv = document.createElement("canvas");
  cv.width = sw;
  cv.height = sh;
  cv.getContext("2d").drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
  return { url: cv.toDataURL("image/png"), sx, sy, sw, sh };
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// Distinct, WCAG-friendly colours per structure class.
export const STRUCTURE_COLORS = {
  "table": "#37474f",
  "table row": "#1565c0",
  "table column": "#1a7a3a",
  "table column header": "#ad1457",
  "table projected row header": "#6a1b9a",
  "table spanning cell": "#a15c00",
};
export function structureColor(label) {
  return STRUCTURE_COLORS[label] || "#b3261e";
}

/** Draw an image + labelled detection boxes at natural resolution (CSS scales it down). */
export function drawDetections(canvas, img, detections, opts = {}) {
  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0, w, h);
  const lw = Math.max(3, w / 260);
  const fontPx = Math.max(15, Math.round(w / 40));
  ctx.font = `600 ${fontPx}px "Avenir Next","Segoe UI",system-ui,sans-serif`;
  ctx.textBaseline = "middle";
  detections.forEach((d, i) => {
    const { xmin, ymin, xmax, ymax } = d.box;
    const color = d.label === "table rotated" ? "#a15c00" : "#1565c0";
    ctx.lineWidth = i === opts.highlightIndex ? lw * 1.8 : lw;
    ctx.strokeStyle = color;
    ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);
    const text = `${d.label} ${(d.score * 100).toFixed(0)}%`;
    const padX = fontPx * 0.35, th = fontPx * 1.35;
    const tw = ctx.measureText(text).width + padX * 2;
    const ly = Math.max(0, ymin - th);
    ctx.fillStyle = color;
    ctx.fillRect(xmin - lw / 2, ly, tw, th);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, xmin - lw / 2 + padX, ly + th / 2);
  });
}

/**
 * Draw the structure overlay on a cropped-table canvas. `cells` are in the crop's pixel coords.
 * `show` is a Set of labels to render (lets the page toggle rows / columns / headers). Rows and columns
 * are drawn as translucent bands; header/spanning cells as filled highlights.
 */
export function drawStructure(canvas, img, cells, show, opts = {}) {
  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0, w, h);
  const lw = Math.max(2, w / 400);
  // Draw bands first (rows/cols), then emphasis boxes on top.
  const order = ["table row", "table column", "table column header", "table projected row header", "table spanning cell"];
  for (const label of order) {
    if (!show.has(label)) continue;
    for (const c of cells) {
      if (c.label !== label) continue;
      const { xmin, ymin, xmax, ymax } = c.box;
      const color = structureColor(label);
      if (label === "table column header" || label === "table spanning cell" || label === "table projected row header") {
        ctx.fillStyle = color + "33";
        ctx.fillRect(xmin, ymin, xmax - xmin, ymax - ymin);
      }
      ctx.lineWidth = lw;
      ctx.strokeStyle = color;
      ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);
    }
  }
  // Optionally draw reconstructed cell grid dots at row×col intersections.
  if (opts.gridCells) {
    ctx.fillStyle = "#b3261e";
    for (const g of opts.gridCells) {
      ctx.beginPath();
      ctx.arc((g.xmin + g.xmax) / 2, (g.ymin + g.ymax) / 2, Math.max(3, w / 260), 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

/** Count structure cells by label → [{label, count}]. */
export function countCells(cells) {
  const m = new Map();
  for (const c of cells) m.set(c.label, (m.get(c.label) ?? 0) + 1);
  return [...m.entries()].map(([label, count]) => ({ label, count }));
}

/**
 * Reconstruct a grid of CELLS from the row and column boxes. Rows and columns are 1-D partitions; a cell
 * is the rectangle where a row band meets a column band. Returns { rows, cols, cells:[{r,c,box}] } sorted
 * top-to-bottom, left-to-right. This is the classic Table Transformer post-processing step.
 */
export function reconstructGrid(cells) {
  const rows = cells.filter((c) => c.label === "table row")
    .sort((a, b) => a.box.ymin - b.box.ymin);
  const cols = cells.filter((c) => c.label === "table column")
    .sort((a, b) => a.box.xmin - b.box.xmin);
  const grid = [];
  rows.forEach((row, r) => {
    cols.forEach((col, c) => {
      grid.push({
        r,
        c,
        box: {
          xmin: col.box.xmin,
          xmax: col.box.xmax,
          ymin: row.box.ymin,
          ymax: row.box.ymax,
        },
      });
    });
  });
  return { rows, cols, cells: grid };
}

export const TT_CSS = `
.dropzone { border: 2px dashed var(--border-strong); border-radius: var(--radius); background: var(--bg-raised);
  padding: 1rem; text-align: center; cursor: pointer; transition: border-color .15s, background .15s; }
.dropzone.drag { border-color: var(--accent); background: var(--bg-secondary); }
.dropzone:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.sample-strip { display: flex; gap: .5rem; flex-wrap: wrap; margin: .5rem 0; }
.sample-thumb { inline-size: 84px; block-size: 60px; object-fit: cover; object-position: top; border-radius: 6px;
  border: 2px solid transparent; cursor: pointer; padding: 0; background: #fff; }
.sample-thumb.active { border-color: var(--accent); }
.sample-thumb:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.canvas-wrap { position: relative; display: block; margin-top: .5rem; border-radius: 8px; overflow: hidden;
  background: var(--bg-raised); border: 1px solid var(--border); }
.stage-canvas { display: block; inline-size: 100%; block-size: auto; max-block-size: 68vh; object-fit: contain; }
.stage-canvas:focus-visible { outline: 3px solid var(--accent); outline-offset: -3px; }
.slider-row { display: flex; align-items: center; gap: .6rem; margin: .6rem 0; flex-wrap: wrap; }
.slider-row input[type=range] { flex: 1 1 180px; accent-color: var(--accent); }
.slider-row output { font-family: var(--font-mono); font-size: .82rem; min-inline-size: 3ch; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .78rem;
  color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.count-chips { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0; }
.count-chip { display: inline-flex; align-items: center; gap: .35rem; font-size: .82rem; padding: .15rem .55rem;
  border-radius: 999px; border: 1px solid var(--border); background: var(--bg-raised); }
.count-chip .swatch { inline-size: .7rem; block-size: .7rem; border-radius: 3px; }
.count-chip b { font-family: var(--font-mono); }
.toggle-row { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0; }
.toggle-chip { display: inline-flex; align-items: center; gap: .35rem; font: inherit; font-size: .8rem;
  padding: .2rem .6rem; border-radius: 999px; border: 1px solid var(--border); background: var(--bg-raised);
  color: var(--color); cursor: pointer; }
.toggle-chip[aria-pressed=true] { border-color: var(--accent); background: var(--bg-secondary); }
.toggle-chip:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.toggle-chip .swatch { inline-size: .7rem; block-size: .7rem; border-radius: 3px; }
.inside-table { inline-size: 100%; border-collapse: collapse; font-size: .82rem; margin-top: .5rem; }
.inside-table th, .inside-table td { text-align: left; padding: .3rem .5rem; border-bottom: 1px solid var(--border);
  font-family: var(--font-mono); }
.inside-table th { color: var(--muted); font-weight: 600; }
.grid-tbl { border-collapse: collapse; margin-top: .6rem; font-size: .82rem; }
.grid-tbl td { border: 1px solid var(--border-strong); padding: .3rem .5rem; min-inline-size: 3rem; }
.grid-tbl tr:first-child td { background: var(--bg-secondary); font-weight: 600; }
.obj-list { display: flex; flex-direction: column; gap: .3rem; margin: .5rem 0; }
.obj-btn { display: flex; align-items: center; gap: .5rem; text-align: left; inline-size: 100%;
  background: var(--bg-raised); color: var(--color); border: 1px solid var(--border); border-radius: 8px;
  padding: .35rem .6rem; font-size: .85rem; cursor: pointer; }
.obj-btn:hover, .obj-btn:focus-visible { border-color: var(--accent); }
.obj-btn[aria-pressed=true] { border-color: var(--accent); background: var(--bg-secondary); }
.obj-btn .sc { margin-inline-start: auto; font-family: var(--font-mono); color: var(--muted); }
.grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; align-items: start; }
.legend { display: flex; flex-wrap: wrap; gap: .8rem; font-size: .76rem; color: var(--muted);
  font-family: var(--font-mono); margin-top: .5rem; }
.legend .swatch { display: inline-block; inline-size: .8rem; block-size: .8rem; border-radius: 3px;
  margin-inline-end: .3rem; vertical-align: -1px; }
.cell-out { font-family: var(--font-mono); font-size: .9rem; padding: .5rem .7rem; border: 1px solid var(--border);
  border-radius: 8px; background: var(--bg-raised); margin-top: .5rem; min-block-size: 1.5rem; }
`;
