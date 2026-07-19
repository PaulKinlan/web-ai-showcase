// Front-end helpers for the YOLO-World open-vocabulary pages. Thin: owns the worker handshake
// (transferring ImageBitmaps zero-copy + the current class list), draws the box overlay, renders the
// per-box table, and provides an accessible class-name CHIP editor (the open-vocab control that makes
// this model distinct — you type the classes). All inference lives in worker.js, off the main thread.

const WORKER_URL = "/web-ai-showcase/models/yolo-world/worker.js";

export class YoloWorldEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.onProgress = null;
    this.onStage = null;
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
    } else if (msg.type === "stage") {
      this.onStage?.(msg.stage);
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

  load(onProgress, onStage) {
    if (onProgress) this.onProgress = onProgress;
    if (onStage) this.onStage = onStage;
    if (this.ready) return Promise.resolve(this.device);
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load" });
    });
  }

  /** Detect on an ImageBitmap (transferred → zero-copy) for the given class-name list. */
  detect(bitmap, classes) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, bitmap, classes }, [bitmap]);
    });
  }
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

/** Normalise a raw class string into a clean, deduped, lowercased list. */
export function parseClasses(raw) {
  const seen = new Set();
  const out = [];
  for (const part of String(raw).split(/[,\n]+/)) {
    const c = part.trim().toLowerCase().replace(/\s+/g, " ");
    if (c && !seen.has(c)) {
      seen.add(c);
      out.push(c);
    }
  }
  return out;
}

/**
 * Draw an image/video frame + its detection boxes onto `canvas` at the source's natural resolution.
 * Boxes are in the source's pixel coordinates; CSS scales the canvas down responsively so boxes stay
 * pixel-accurate. `highlightIndex` thickens one box (used by the multi-model crop picker).
 */
export function drawDetections(canvas, source, detections, opts = {}) {
  const w = source.naturalWidth || source.videoWidth || source.width;
  const h = source.naturalHeight || source.videoHeight || source.height;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(source, 0, 0, w, h);

  const lw = Math.max(2, w / 320);
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

/**
 * An accessible class-name chip editor. Renders the current classes as removable chips plus a text
 * input; Enter or comma commits a chip, Backspace on an empty input removes the last chip. Calls
 * `onChange(classes)` whenever the list changes. Returns { get, set }.
 */
export function createClassEditor(mount, initial, onChange) {
  let classes = [...initial];
  mount.classList.add("chip-editor");
  const list = document.createElement("div");
  list.className = "chip-list";
  const input = document.createElement("input");
  input.type = "text";
  input.className = "chip-input";
  input.setAttribute("aria-label", "Add a class name to detect");
  input.placeholder = "type a class + Enter…";
  mount.replaceChildren(list, input);

  function render() {
    list.replaceChildren(...classes.map((c, i) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.style.setProperty("--chip", colorForLabel(c));
      const label = document.createElement("span");
      label.className = "chip-label";
      label.textContent = c;
      const rm = document.createElement("button");
      rm.type = "button";
      rm.className = "chip-x";
      rm.setAttribute("aria-label", `Remove class ${c}`);
      rm.textContent = "×";
      rm.addEventListener("click", () => {
        classes.splice(i, 1);
        render();
        onChange(classes);
      });
      chip.append(label, rm);
      return chip;
    }));
  }
  function commit() {
    for (const c of parseClasses(input.value)) {
      if (!classes.includes(c)) classes.push(c);
    }
    input.value = "";
    render();
    onChange(classes);
  }
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      commit();
    } else if (e.key === "Backspace" && input.value === "" && classes.length) {
      classes.pop();
      render();
      onChange(classes);
    }
  });
  input.addEventListener("blur", () => {
    if (input.value.trim()) commit();
  });
  render();
  return {
    get: () => [...classes],
    set: (next) => {
      classes = [...next];
      render();
      onChange(classes);
    },
  };
}

export const YW_CSS = `
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
.slider-row input[type=range] { flex: 1 1 180px; accent-color: var(--accent); min-inline-size: 0; }
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
.inside-table { inline-size: 100%; border-collapse: collapse; font-size: .82rem; margin-top: .5rem; }
.inside-table th, .inside-table td {
  text-align: left; padding: .3rem .5rem; border-bottom: 1px solid var(--border); font-family: var(--font-mono);
}
.inside-table th { color: var(--muted); font-weight: 600; }
.chip-editor {
  display: flex; flex-wrap: wrap; gap: .4rem; align-items: center; padding: .45rem .5rem;
  border: 1px solid var(--border-strong); border-radius: var(--radius); background: var(--bg-raised);
}
.chip-editor:focus-within { border-color: var(--accent); }
.chip-list { display: flex; flex-wrap: wrap; gap: .35rem; }
.chip {
  display: inline-flex; align-items: center; gap: .3rem; font-size: .84rem; line-height: 1.4;
  padding: .1rem .2rem .1rem .5rem; border-radius: 999px; color: #fff; background: var(--chip, #37474f);
}
.chip-label { white-space: nowrap; }
.chip-x {
  display: inline-flex; align-items: center; justify-content: center;
  inline-size: 1.35rem; block-size: 1.35rem; border: 0; border-radius: 999px; cursor: pointer;
  background: rgba(255,255,255,.22); color: #fff; font-size: 1rem; line-height: 1; padding: 0;
}
.chip-x:hover, .chip-x:focus-visible { background: rgba(255,255,255,.4); outline: 2px solid #fff; outline-offset: -2px; }
.chip-input {
  flex: 1 1 8rem; min-inline-size: 6rem; border: 0; background: transparent; color: var(--color);
  font: inherit; padding: .25rem; outline: none;
}
.preset-row { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0; }
.preset-btn {
  font-size: .8rem; padding: .25rem .6rem; border-radius: 999px; cursor: pointer;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color);
}
.preset-btn:hover, .preset-btn:focus-visible { border-color: var(--accent); }
.obj-list { display: flex; flex-direction: column; gap: .3rem; margin: .5rem 0; }
.obj-btn {
  display: flex; align-items: center; gap: .5rem; text-align: left; inline-size: 100%;
  background: var(--bg-raised); color: var(--color); border: 1px solid var(--border);
  border-radius: 8px; padding: .35rem .6rem; font-size: .85rem; cursor: pointer;
}
.obj-btn:hover, .obj-btn:focus-visible { border-color: var(--accent); }
.obj-btn[aria-pressed=true] { border-color: var(--accent); background: var(--bg-secondary); }
.obj-btn .swatch { inline-size: .8rem; block-size: .8rem; border-radius: 3px; flex: none; }
.obj-btn .sc { margin-inline-start: auto; font-family: var(--font-mono); color: var(--muted); }
.fallback { border: 1px solid var(--warn, #a15c00); border-radius: var(--radius); background: var(--bg-raised); padding: 1rem; }
.fallback code { background: var(--bg-secondary); padding: .05rem .3rem; border-radius: 4px; }
.crop-out { display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start; margin-top: .6rem; }
.crop-out canvas { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised); max-inline-size: 220px; block-size: auto; }
`;
