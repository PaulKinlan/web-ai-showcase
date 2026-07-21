// RAFT optical-flow engine + rendering helpers. The engine talks to worker.js (raw ORT-web); the helpers
// build a two-frame pair with a KNOWN motion from one source image (so the demo is self-contained and the
// recovered flow can be checked against the applied motion) and render the flow field as a Middlebury-style
// colour wheel (hue = direction, colourfulness = speed). No fake output — every pixel comes from the model.

export const IN_W = 480, IN_H = 360; // the RAFT Sintel export's fixed input size

export class FlowEngine {
  constructor() {
    this.worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    this.onProgress = null;
    this.pending = new Map();
    this.seq = 0;
    this.worker.addEventListener("message", (e) => {
      const m = e.data;
      if (m.type === "progress") this.onProgress?.(m.p);
      else if (m.type === "ready") this._ready?.();
      else if (m.type === "flow") this.pending.get(m.id)?.resolve(m);
      else if (m.type === "error") {
        if (m.id != null && this.pending.has(m.id)) {
          this.pending.get(m.id).reject(new Error(m.message));
        } else this._readyReject?.(new Error(m.message));
      }
    });
  }
  load(onProgress) {
    if (onProgress) this.onProgress = onProgress;
    return new Promise((resolve, reject) => {
      this._ready = resolve;
      this._readyReject = reject;
      this.worker.postMessage({ type: "load" });
    });
  }
  /** Compute dense flow between two RGBA ImageData frames (each IN_W×IN_H). */
  computeFlow(frame1, frame2) {
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      // copy the pixel buffers so the transfer doesn't neuter the caller's ImageData
      const f1 = new Uint8ClampedArray(frame1.data);
      const f2 = new Uint8ClampedArray(frame2.data);
      this.worker.postMessage({ type: "flow", id, frame1: f1, frame2: f2 }, [f1.buffer, f2.buffer]);
    }).finally(() => this.pending.delete(id));
  }
  dispose() {
    this.worker.terminate();
  }
}

/**
 * Build a two-frame pair from ONE source bitmap with a known translation (dx,dy in output pixels). Uses a
 * cover-crop so the moved frame shows real content (no black borders) for small pans — the recovered flow
 * should then match (dx,dy). Returns { frame1, frame2, appliedDx, appliedDy } as IN_W×IN_H ImageData.
 */
export function buildFrames(bitmap, { dx = 12, dy = 0, zoom = 0 } = {}) {
  const sw = bitmap.width, sh = bitmap.height;
  // Zoom in ~18% past the cover scale so the crop is smaller than the source on BOTH axes — this leaves
  // real content in the margins to pan into (otherwise a portrait source cover-cropped to landscape has
  // no horizontal room and frame 2 would clamp to frame 1 → a false zero flow). The extra headroom also
  // lets a `zoom` motion sample a smaller (magnified) source window for frame 2 → radial outward flow.
  const scale = Math.max(IN_W / sw, IN_H / sh) / 0.8;
  const cropW = IN_W / scale, cropH = IN_H / scale;
  const cx = (sw - cropW) / 2, cy = (sh - cropH) / 2;
  // grab a source window: offset by (offx,offy) dest px (pan) and shrunk by `z` fraction (zoom-in).
  const grab = (offx, offy, z) => {
    const c = new OffscreenCanvas(IN_W, IN_H);
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.imageSmoothingQuality = "high";
    const zw = cropW * (1 - z), zh = cropH * (1 - z); // magnified window
    const sx = Math.max(0, Math.min(sw - zw, cx + (cropW - zw) / 2 - offx / scale));
    const sy = Math.max(0, Math.min(sh - zh, cy + (cropH - zh) / 2 - offy / scale));
    ctx.drawImage(bitmap, sx, sy, zw, zh, 0, 0, IN_W, IN_H);
    return ctx.getImageData(0, 0, IN_W, IN_H);
  };
  return {
    frame1: grab(0, 0, 0),
    frame2: grab(dx, dy, zoom),
    appliedDx: dx,
    appliedDy: dy,
    appliedZoom: zoom,
  };
}

/** HSV→RGB (h in [0,360), s,v in [0,1]). */
function hsv(h, s, v) {
  const c = v * s, x = c * (1 - Math.abs(((h / 60) % 2) - 1)), m = v - c;
  let r = 0, g = 0, b = 0;
  if (h < 60) [r, g, b] = [c, x, 0];
  else if (h < 120) [r, g, b] = [x, c, 0];
  else if (h < 180) [r, g, b] = [0, c, x];
  else if (h < 240) [r, g, b] = [0, x, c];
  else if (h < 300) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  return [(r + m) * 255, (g + m) * 255, (b + m) * 255];
}

/** Render a flow field (u,v planes, w×h) as a colour image: hue = direction, saturation = speed/maxMag. */
export function flowToImageData(u, v, w, h, maxMag) {
  const out = new ImageData(w, h);
  const scale = maxMag > 1e-6 ? maxMag : 1;
  for (let i = 0; i < w * h; i++) {
    const mag = Math.hypot(u[i], v[i]);
    const ang = (Math.atan2(v[i], u[i]) * 180) / Math.PI; // -180..180
    const [r, g, b] = hsv((ang + 360) % 360, Math.min(1, mag / scale), 1);
    out.data[i * 4] = r;
    out.data[i * 4 + 1] = g;
    out.data[i * 4 + 2] = b;
    out.data[i * 4 + 3] = 255;
  }
  return out;
}

/** A small legend colour wheel (direction → hue) for the UI. */
export function drawColorWheel(canvas, r = 48) {
  const ctx = canvas.getContext("2d");
  const d = r * 2;
  canvas.width = d;
  canvas.height = d;
  const img = ctx.createImageData(d, d);
  for (let y = 0; y < d; y++) {
    for (let x = 0; x < d; x++) {
      const dx = x - r, dy = y - r, mag = Math.hypot(dx, dy) / r;
      const i = (y * d + x) * 4;
      if (mag > 1) {
        img.data[i + 3] = 0;
        continue;
      }
      const ang = (Math.atan2(dy, dx) * 180) / Math.PI;
      const [rr, gg, bb] = hsv((ang + 360) % 360, mag, 1);
      img.data[i] = rr;
      img.data[i + 1] = gg;
      img.data[i + 2] = bb;
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

export function putImageDataFit(canvas, imageData) {
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  canvas.getContext("2d").putImageData(imageData, 0, 0);
}

export async function urlToBitmap(src) {
  const res = await fetch(src);
  return createImageBitmap(await res.blob());
}

export const FLOW_CSS = `
  .flow-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.8rem; margin: 0.8rem 0; }
  .flow-cell { text-align: center; }
  .flow-cell canvas { width: 100%; max-width: 480px; height: auto; border-radius: 8px; background: #7772; aspect-ratio: 4/3; }
  .flow-cell figcaption { font-size: 0.78rem; opacity: 0.8; margin-top: 0.25rem; }
  .flow-controls { display: flex; flex-wrap: wrap; gap: 0.6rem 1rem; align-items: center; margin: 0.6rem 0; font-size: 0.9rem; }
  .flow-legend { display: inline-flex; align-items: center; gap: 0.5rem; font-size: 0.78rem; opacity: 0.85; }
  .flow-readout { font-family: var(--font-mono, monospace); font-size: 0.82rem; margin: 0.4rem 0; }
  .flow-samples { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.4rem 0; }
  .flow-samples img { width: 76px; height: 57px; object-fit: cover; border-radius: 6px; cursor: pointer; border: 2px solid transparent; }
  .flow-samples img.active { border-color: var(--accent, #38f); }
`;
