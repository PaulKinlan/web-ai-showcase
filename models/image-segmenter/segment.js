// Front-end helpers for the MediaPipe Image Segmenter pages. Mirrors the face.js/hand.js/pose.js
// pattern: one shared module the overview + ladder routes import; real inference runs in
// models/image-segmenter/worker.js (off the main thread, lib/worker-protocol.js); the main thread
// only colorizes the returned category mask and paints it onto a canvas.
//
// Model: Google's official MediaPipe image segmenter — DeepLab v3+ trained on PASCAL VOC 2012
// (21 classes), shipped by Google as deeplab_v3.tflite on mediapipe-models.storage.googleapis.com.
// Verified 2026-07-27 (probe): @mediapipe/tasks-vision@0.10.18 + this .tflite returns a real
// 21-label category mask + 21 confidence masks in headless Chrome on the WASM/CPU delegate.

import { TASKS_VISION_VERSION } from "/web-ai-showcase/lib/mediapipe.js";
import { SupersededError, WorkerClient } from "/web-ai-showcase/lib/worker-protocol.js";

export const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite";
export const MODEL_SIZE_MB = 2.8;
export const TASK_GUIDE_URL =
  "https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter";

const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}`;

// PASCAL VOC 2012 class order used by Google's deeplab_v3.tflite label map — verified live via
// segmenter.getLabels() in the probe. Palette: distinct, colour-blind-aware-ish hues.
export const VOC_COLORS = [
  ["background", "#1b1b1f"],
  ["aeroplane", "#e6194b"],
  ["bicycle", "#3cb44b"],
  ["bird", "#ffe119"],
  ["boat", "#4363d8"],
  ["bottle", "#f58231"],
  ["bus", "#911eb4"],
  ["car", "#42d4f4"],
  ["cat", "#f032e6"],
  ["chair", "#bfef45"],
  ["cow", "#469990"],
  ["dining table", "#9a6324"],
  ["dog", "#800000"],
  ["horse", "#aaffc3"],
  ["motorbike", "#808000"],
  ["person", "#3949ab"],
  ["potted plant", "#2e8b57"],
  ["sheep", "#ffd8b1"],
  ["sofa", "#e6beff"],
  ["train", "#000075"],
  ["tv", "#a9a9a9"],
];
export const className = (i) => VOC_COLORS[i]?.[0] || `class ${i}`;
export const classColor = (i) => VOC_COLORS[i]?.[1] || "#888888";

/** Wraps the worker-hosted ImageSegmenter. Same client shape as FaceWorkerClient. */
export class SegmenterWorkerClient {
  constructor(client, delegate, labels) {
    this.client = client;
    this._delegate = delegate;
    this.labels = labels;
  }

  /** Boot the worker, download+create the segmenter, and return a ready handle. */
  static async create({ modelUrl = MODEL_URL, options = {}, onProgress, onState } = {}) {
    onProgress?.({ status: "initiate", file: modelUrl });
    const client = new WorkerClient({
      url: new URL("./worker.js", import.meta.url), // resolved relative to segment.js
      name: "image-segmenter",
      maxInFlight: 1,
      maxQueue: 1,
      onState,
      // CLASSIC worker: MediaPipe's WASM loader (FilesetResolver) calls importScripts, which a
      // module worker rejects — the same measured exception as the 5 built MediaPipe demos.
      module: false,
    });
    await client.ready;
    const { result } = await client.request("configure", { modelUrl, options });
    onProgress?.({ status: "ready" });
    return new SegmenterWorkerClient(client, result?.delegate || "CPU", result?.labels || []);
  }

  get delegate() {
    return this._delegate;
  }

  /**
   * Segment a still image. Returns { width, height, mask: Uint8Array (category index per pixel),
   * hist: Uint32Array (pixels per class), conf: number[] (mean confidence per class), labels }.
   * The input bitmap is TRANSFERRED to the worker; the ~256KB category mask is cloned back.
   */
  async segmentImage(source) {
    const bitmap = await createImageBitmap(source);
    try {
      const { result } = await this.client.request(
        "segment",
        { bitmap },
        { transfer: [bitmap] },
      );
      return result;
    } catch (err) {
      if (err instanceof SupersededError || err?.name === "AbortError") return null;
      throw err;
    }
  }

  terminate() {
    return this.client.terminate();
  }
}

/**
 * Per-class summary of a result, sorted by pixel coverage (largest first). Includes only classes
 * actually present (>=1 px) plus their mean confidence from the model's confidence masks.
 */
export function summarize(result) {
  const { hist, conf, mask } = result;
  const total = mask.length;
  const out = [];
  for (let i = 0; i < hist.length; i++) {
    if (!hist[i]) continue;
    out.push({
      index: i,
      name: className(i),
      color: classColor(i),
      pct: (hist[i] / total) * 100,
      conf: conf?.[i] ?? null,
    });
  }
  out.sort((a, b) => b.pct - a.pct);
  return out;
}

/**
 * Paint `source` + the category mask onto `canvas` at the source's natural size.
 * mode: "overlay" (source + class colours at `opacity`) | "mask" (flat colour map on dark ground)
 *       | "cutout" (foreground classes keep the photo; background pixels become transparent —
 *         the page should put a checkerboard/solid behind the canvas)
 * isolate: a class index to spotlight (only that class is coloured/kept; others dim to 12%).
 * Per-pixel work happens at MASK resolution (≤640² on bundled samples, a few ms) in one pass.
 */
export function drawSegmentation(canvas, source, result, opts = {}) {
  const w = source.naturalWidth || source.videoWidth || source.width;
  const h = source.naturalHeight || source.videoHeight || source.height;
  const { mask, width: mw, height: mh } = result;
  const mode = opts.mode || "overlay";
  const opacity = opts.opacity ?? 0.55;
  const isolate = Number.isInteger(opts.isolate) ? opts.isolate : null;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);

  // Colorize at mask resolution.
  const rgba = new Uint8ClampedArray(mw * mh * 4);
  for (let p = 0, q = 0; p < mask.length; p++, q += 4) {
    const cls = mask[p];
    const hex = classColor(cls);
    rgba[q] = parseInt(hex.slice(1, 3), 16);
    rgba[q + 1] = parseInt(hex.slice(3, 5), 16);
    rgba[q + 2] = parseInt(hex.slice(5, 7), 16);
    if (mode === "cutout") {
      const keep = isolate != null ? cls === isolate : cls !== 0;
      rgba[q + 3] = keep ? 255 : 0;
    } else if (mode === "spotlight") {
      // Black dimming layer: transparent over the spotlit class, near-opaque elsewhere.
      rgba[q] = 0;
      rgba[q + 1] = 0;
      rgba[q + 2] = 0;
      rgba[q + 3] = isolate != null && cls === isolate ? 0 : 225;
    } else {
      rgba[q + 3] = isolate != null && cls !== isolate ? 30 : 255;
    }
  }
  const off = document.createElement("canvas");
  off.width = mw;
  off.height = mh;
  off.getContext("2d").putImageData(new ImageData(rgba, mw, mh), 0, 0);

  if (mode === "mask") {
    ctx.fillStyle = "#1b1b1f";
    ctx.fillRect(0, 0, w, h);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, 0, 0, w, h);
    return;
  }
  ctx.drawImage(source, 0, 0, w, h);
  if (mode === "cutout") {
    // Keep only the wanted class pixels of the photo itself.
    ctx.globalCompositeOperation = "destination-in";
    ctx.drawImage(off, 0, 0, w, h);
    ctx.globalCompositeOperation = "source-over";
    return;
  }
  if (mode === "spotlight") {
    ctx.drawImage(off, 0, 0, w, h);
    return;
  }
  // overlay
  ctx.globalAlpha = opacity;
  ctx.drawImage(off, 0, 0, w, h);
  ctx.globalAlpha = 1;
}

/**
 * Render the class legend into `el`: one chip per present class (colour swatch + name + coverage %
 * + mean confidence). If onSelect is given, chips are toggle-buttons (aria-pressed) for spotlight.
 */
export function renderLegend(el, summary, { onSelect = null, selected = null } = {}) {
  el.replaceChildren();
  for (const s of summary) {
    const chip = document.createElement(onSelect ? "button" : "span");
    chip.className = "count-chip";
    if (onSelect) {
      chip.type = "button";
      chip.setAttribute("aria-pressed", String(selected === s.index));
      chip.addEventListener("click", () => onSelect(selected === s.index ? null : s.index));
    }
    const sw = document.createElement("span");
    sw.className = "swatch";
    sw.style.background = s.color;
    chip.append(
      sw,
      `${s.name} · ${s.pct.toFixed(1)}%` +
        (s.conf != null ? ` · conf ${s.conf.toFixed(2)}` : ""),
    );
    el.append(chip);
  }
}

// Shared widget styles (design-system variables only) — same base as the other MediaPipe demos.
export const SEGMENT_CSS = `
.dropzone { border:2px dashed var(--border-strong); border-radius:var(--radius); background:var(--bg-raised);
  padding:1rem; text-align:center; cursor:pointer; transition:border-color .15s, background .15s; }
.dropzone.drag { border-color:var(--accent); background:var(--bg-secondary); }
.dropzone:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.sample-strip { display:flex; gap:.5rem; flex-wrap:wrap; margin:.5rem 0; }
.sample-thumb { inline-size:76px; block-size:56px; object-fit:cover; border-radius:6px; border:2px solid transparent;
  cursor:pointer; padding:0; background:var(--bg-raised); }
.sample-thumb.active { border-color:var(--accent); }
.sample-thumb:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.canvas-wrap { position:relative; display:block; margin-top:.5rem; border-radius:8px; overflow:hidden;
  border:1px solid var(--border);
  background:repeating-conic-gradient(#c8c8d0 0% 25%, #ececf0 0% 50%) 0 0/18px 18px; }
.stage-canvas { display:block; inline-size:100%; block-size:auto; max-block-size:64vh; object-fit:contain; }
.stage-canvas:focus-visible { outline:3px solid var(--accent); outline-offset:-3px; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem;
  color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.field-row { display:flex; flex-wrap:wrap; gap:.6rem; align-items:center; margin:.6rem 0; }
.count-chips { display:flex; flex-wrap:wrap; gap:.4rem; margin:.5rem 0; }
.count-chip { display:inline-flex; align-items:center; gap:.35rem; font-size:.82rem; padding:.35rem .55rem;
  border-radius:999px; border:1px solid var(--border); background:var(--bg-raised); color:var(--color); }
button.count-chip { cursor:pointer; }
button.count-chip[aria-pressed="true"] { border-color:var(--accent); background:color-mix(in srgb, var(--accent) 14%, transparent); }
button.count-chip:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.count-chip .swatch { inline-size:.7rem; block-size:.7rem; border-radius:3px; flex:none; }
.inside-table { inline-size:100%; border-collapse:collapse; font-size:.82rem; margin-top:.5rem; }
.inside-table th, .inside-table td { text-align:left; padding:.3rem .5rem; border-bottom:1px solid var(--border);
  font-family:var(--font-mono); }
.inside-table th { color:var(--muted); font-weight:600; }
.seg-view { display:inline-flex; border:1px solid var(--border); border-radius:999px; overflow:hidden; }
.seg-view button { border:0; background:var(--bg-raised); color:var(--color); padding:.4rem .8rem;
  font-size:.85rem; cursor:pointer; }
.seg-view button[aria-pressed="true"] { background:var(--accent); color:var(--on-accent,#fff); }
.seg-view button:focus-visible { outline:2px solid var(--accent); outline-offset:-2px; }
.seg-slider { display:flex; align-items:center; gap:.5rem; font-size:.85rem; color:var(--muted); }
.seg-slider input[type="range"] { accent-color:var(--accent); min-inline-size:8rem; }
/* Long MediaPipe model URLs in the shared loader detail must wrap at narrow viewports
   (same family-local fix as models/headline-generation). */
.model-loader .loader-detail { overflow-wrap:anywhere; }
`;
