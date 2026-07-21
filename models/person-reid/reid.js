// Person re-ID engine + helpers. The engine talks to worker.js (raw ORT-web); the helpers crop a person
// box from a source photo to the model's 128×256 input and compute cosine similarity between L2-normalised
// embeddings. No fake output — every number is the model's.

export const IN_W = 128, IN_H = 256; // the model's fixed person-crop input size

export class ReidEngine {
  constructor() {
    this.worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    this.onProgress = null;
    this.pending = new Map();
    this.seq = 0;
    this.worker.addEventListener("message", (e) => {
      const m = e.data;
      if (m.type === "progress") this.onProgress?.(m.p);
      else if (m.type === "ready") this._ready?.();
      else if (m.type === "embedding") this.pending.get(m.id)?.resolve(m);
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
  /** Embed a person crop (RGBA ImageData at IN_W×IN_H) → { vec:Float32Array, dim, ms }. */
  embed(crop) {
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      const c = new Uint8ClampedArray(crop.data);
      this.worker.postMessage({ type: "embed", id, crop: c }, [c.buffer]);
    }).finally(() => this.pending.delete(id));
  }
  dispose() {
    this.worker.terminate();
  }
}

/** Cosine similarity of two already-L2-normalised vectors. */
export function cosine(a, b) {
  let d = 0;
  for (let i = 0; i < a.length; i++) d += a[i] * b[i];
  return d;
}

/**
 * Crop a person box [x,y,w,h] (source pixels) from a bitmap → IN_W×IN_H ImageData. `jitter` (fraction)
 * shifts + brightens the crop a little to simulate a different-camera view of the SAME person.
 */
export function cropPerson(bitmap, [x, y, w, h], { jitter = 0 } = {}) {
  const c = new OffscreenCanvas(IN_W, IN_H);
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  const jx = jitter ? (w * jitter * 0.5) : 0, jy = jitter ? (h * jitter * 0.3) : 0;
  ctx.drawImage(
    bitmap,
    x + jx,
    y + jy,
    w * (1 - jitter * 0.4),
    h * (1 - jitter * 0.2),
    0,
    0,
    IN_W,
    IN_H,
  );
  const img = ctx.getImageData(0, 0, IN_W, IN_H);
  if (jitter) {
    const b = 22;
    for (let i = 0; i < img.data.length; i += 4) {
      img.data[i] = Math.min(255, img.data[i] + b);
      img.data[i + 1] = Math.min(255, img.data[i + 1] + b);
      img.data[i + 2] = Math.min(255, img.data[i + 2] + b);
    }
  }
  return img;
}

export function putImageData(canvas, imageData) {
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  canvas.getContext("2d").putImageData(imageData, 0, 0);
}

export async function urlToBitmap(src) {
  const res = await fetch(src);
  return createImageBitmap(await res.blob());
}

export const REID_CSS = `
  .reid-gallery { display: flex; flex-wrap: wrap; gap: 0.6rem; margin: 0.8rem 0; }
  .reid-person { text-align: center; cursor: pointer; border: 2px solid transparent; border-radius: 8px; padding: 3px; }
  .reid-person.sel { border-color: var(--accent, #38f); }
  .reid-person canvas { width: 72px; height: 144px; object-fit: cover; border-radius: 6px; background: #7772; display: block; }
  .reid-person figcaption { font-size: 0.72rem; opacity: 0.8; margin-top: 0.2rem; max-width: 78px; }
  .reid-result { font-family: var(--font-mono, monospace); font-size: 0.9rem; margin: 0.6rem 0; }
  .reid-bar { height: 12px; border-radius: 6px; background: #7772; overflow: hidden; max-width: 320px; margin: 0.3rem 0; }
  .reid-bar > i { display: block; height: 100%; background: var(--accent, #38f); }
  .reid-verdict { font-weight: 600; }
`;
