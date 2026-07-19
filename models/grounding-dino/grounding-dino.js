// Shared front-end helpers for the Grounding DINO pages. Keeps each page thin: it owns the worker
// handshake, turns files/samples into data URLs, draws the box overlay (coloured per phrase), parses
// the caption/prompt into phrases, and summarises per-phrase scores. All inference is in worker.js.
//
// Grounding DINO is PHRASE-GROUNDING: you give it a natural-language caption — "a cat. a remote
// control. a laptop." — and it grounds each phrase to boxes in the image. So the prompt here is a
// sentence split on periods into phrases, not a fixed comma list of labels (that's OWLv2's framing).

const WORKER_URL = "/web-ai-showcase/models/grounding-dino/worker.js";

export class GroundingEngine {
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

  detect(imageURL, queries) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, image: imageURL, queries });
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

export function decodeImage(url) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.crossOrigin = "anonymous";
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error("Could not decode image"));
    im.src = url;
  });
}

/**
 * Parse a Grounding DINO caption/prompt into clean, de-duped phrases. Phrases are separated by
 * PERIODS (the model's own convention) — we also accept newlines/commas for convenience. Each phrase
 * is lowercased and stripped of a trailing period; the worker re-adds the "." the model expects, and
 * the returned box label is the phrase in this same lowercased, period-free form.
 */
export function parseQueries(text) {
  return [
    ...new Set(
      text.split(/[.,\n]/).map((s) => s.trim().toLowerCase().replace(/\.+$/, "")).filter(Boolean),
    ),
  ];
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

const _queryColors = new Map();
/** Deterministic colour per text QUERY so the same prompt keeps its colour across runs. */
export function colorForQuery(q) {
  if (!_queryColors.has(q)) {
    _queryColors.set(q, BOX_PALETTE[_queryColors.size % BOX_PALETTE.length]);
  }
  return _queryColors.get(q);
}

/** Draw an image + its detection boxes onto `canvas` at the image's natural resolution. */
export function drawDetections(canvas, imgEl, detections, opts = {}) {
  const w = imgEl.naturalWidth || imgEl.videoWidth || imgEl.width;
  const h = imgEl.naturalHeight || imgEl.videoHeight || imgEl.height;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(imgEl, 0, 0, w, h);

  const lw = Math.max(2, w / 320);
  const fontPx = Math.max(13, Math.round(w / 42));
  ctx.font = `600 ${fontPx}px "Avenir Next", "Segoe UI", system-ui, sans-serif`;
  ctx.textBaseline = "middle";
  const highlight = opts.highlightIndex;

  detections.forEach((d, i) => {
    const { xmin, ymin, xmax, ymax } = d.box;
    const color = colorForQuery(d.label);
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

/** Summarise detections by query → [{query, count, top}], best-scoring query first. */
export function summariseByQuery(queries, detections) {
  const m = new Map(queries.map((q) => [q, { query: q, count: 0, top: 0 }]));
  for (const d of detections) {
    const row = m.get(d.label) || { query: d.label, count: 0, top: 0 };
    row.count++;
    row.top = Math.max(row.top, d.score);
    m.set(d.label, row);
  }
  return [...m.values()].sort((a, b) => b.top - a.top);
}

/** Shared inline styles for the OWLv2 widgets (canvas stage, dropzone, query chips, box table). */
export const GDINO_CSS = `
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
.sample-strip { display: flex; gap: .5rem; flex-wrap: wrap; margin: .5rem 0; }
.sample-thumb {
  inline-size: 76px; block-size: 56px; object-fit: cover; border-radius: 6px;
  border: 2px solid transparent; cursor: pointer; padding: 0;
}
.sample-thumb.active { border-color: var(--accent); }
.sample-thumb:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.query-field { display: flex; flex-direction: column; gap: .25rem; font-size: .82rem; margin: .4rem 0; }
.query-field textarea { inline-size: 100%; }
.preset-row { display: flex; flex-wrap: wrap; gap: .4rem; margin: .3rem 0; }
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
.chip {
  font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer;
}
.chip:hover { border-color: var(--accent); }
.count-chips { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0; }
.count-chip {
  display: inline-flex; align-items: center; gap: .35rem; font-size: .82rem;
  padding: .15rem .55rem; border-radius: 999px; border: 1px solid var(--border);
  background: var(--bg-raised);
}
.count-chip .swatch { inline-size: .7rem; block-size: .7rem; border-radius: 3px; }
.count-chip b { font-family: var(--font-mono); }
.inside-table { inline-size: 100%; border-collapse: collapse; font-size: .82rem; margin-top: .5rem; }
.inside-table th, .inside-table td {
  text-align: left; padding: .3rem .5rem; border-bottom: 1px solid var(--border);
  font-family: var(--font-mono);
}
.inside-table th { color: var(--muted); font-weight: 600; }
.fallback { border: 1px solid var(--warn); border-radius: var(--radius); background: var(--bg-raised); padding: 1rem; }
`;
