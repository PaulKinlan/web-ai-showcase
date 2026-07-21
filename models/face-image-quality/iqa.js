// Face image-quality engine + helpers. The engine talks to worker.js (raw ORT-web); the helpers crop a
// face to the model's 112×112 input and can DEGRADE it (blur / darken) so the quality score's response to
// capture problems is visible. No fake output — every score is the model's.

export const IN = 112; // the model's fixed square input

export class IqaEngine {
  constructor() {
    this.worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    this.onProgress = null;
    this.pending = new Map();
    this.seq = 0;
    this.worker.addEventListener("message", (e) => {
      const m = e.data;
      if (m.type === "progress") this.onProgress?.(m.p);
      else if (m.type === "ready") this._ready?.();
      else if (m.type === "quality") this.pending.get(m.id)?.resolve(m);
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
  /** Score a face crop (RGBA ImageData at IN×IN) → { score, ms }. */
  assess(crop) {
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      const c = new Uint8ClampedArray(crop.data);
      this.worker.postMessage({ type: "assess", id, crop: c }, [c.buffer]);
    }).finally(() => this.pending.delete(id));
  }
  dispose() {
    this.worker.terminate();
  }
}

/**
 * Center-crop a face bitmap to IN×IN, with optional degradation: `blur` px (Gaussian) and `dark` in
 * [0,1] (fraction darker) — to show how capture problems drop the quality score.
 */
export function faceCrop(bitmap, { blur = 0, dark = 0 } = {}) {
  const c = new OffscreenCanvas(IN, IN);
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  // cover-crop to square, centred
  const s = Math.min(bitmap.width, bitmap.height);
  const sx = (bitmap.width - s) / 2, sy = (bitmap.height - s) / 2;
  if (blur) ctx.filter = `blur(${blur}px)`;
  ctx.drawImage(bitmap, sx, sy, s, s, 0, 0, IN, IN);
  ctx.filter = "none";
  const img = ctx.getImageData(0, 0, IN, IN);
  if (dark) {
    const k = 1 - Math.max(0, Math.min(1, dark));
    for (let i = 0; i < img.data.length; i += 4) {
      img.data[i] *= k;
      img.data[i + 1] *= k;
      img.data[i + 2] *= k;
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

export const IQA_CSS = `
  .iqa-gallery { display: flex; flex-wrap: wrap; gap: 0.7rem; margin: 0.8rem 0; }
  .iqa-face { text-align: center; }
  .iqa-face canvas { width: 96px; height: 96px; object-fit: cover; border-radius: 8px; background: #7772; display: block; }
  .iqa-face figcaption { font-size: 0.74rem; margin-top: 0.25rem; }
  .iqa-face .q { font-family: var(--font-mono, monospace); font-weight: 600; }
  .iqa-bar { height: 8px; border-radius: 4px; background: #7772; overflow: hidden; width: 96px; margin: 0.2rem auto 0; }
  .iqa-bar > i { display: block; height: 100%; background: linear-gradient(90deg,#e55,#fb3,#3c6); }
  .iqa-degrade { display: flex; flex-wrap: wrap; gap: 0.8rem 1.4rem; align-items: center; margin: 0.6rem 0; font-size: 0.9rem; }
  .iqa-result { font-family: var(--font-mono, monospace); font-size: 0.9rem; margin: 0.5rem 0; }
`;
