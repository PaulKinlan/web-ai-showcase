// media-pipeline.image-worker.js — the OffscreenCanvas half of ImagePreprocessor (lib/media-pipeline.js).
//
// It receives a TRANSFERRED ImageBitmap (already decoded + resized on the main thread via
// createImageBitmap), draws it into an OffscreenCanvas, reads the pixels HERE (getImageData off the
// main thread — the whole point), normalizes them into a Float32 tensor, and transfers the tensor
// buffer back. Self-contained (no imports) so it also runs where module-worker import maps differ.

/** Normalize RGBA bytes into a Float32 tensor (mirrors rgbaToTensor in media-pipeline.js). */
function rgbaToTensor(rgba, width, height, { mean, std, layout, normalize }) {
  const n = width * height;
  const out = new Float32Array(n * 3);
  const m = mean || [0, 0, 0];
  const s = std || [1, 1, 1];
  const scale = normalize ? 1 / 255 : 1;
  for (let i = 0; i < n; i++) {
    const r = rgba[i * 4] * scale;
    const g = rgba[i * 4 + 1] * scale;
    const b = rgba[i * 4 + 2] * scale;
    const rv = (r - m[0]) / s[0];
    const gv = (g - m[1]) / s[1];
    const bv = (b - m[2]) / s[2];
    if (layout === "CHW") {
      out[i] = rv;
      out[n + i] = gv;
      out[2 * n + i] = bv;
    } else {
      out[i * 3] = rv;
      out[i * 3 + 1] = gv;
      out[i * 3 + 2] = bv;
    }
  }
  return out;
}

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === "init") {
    // Report whether THIS worker realm actually has OffscreenCanvas — the client falls back if not.
    self.postMessage({ type: "ready", offscreen: typeof OffscreenCanvas !== "undefined" });
    return;
  }
  if (msg.type === "preprocess") {
    const { id, bitmap, width, height, mean, std, layout, normalize } = msg;
    try {
      const t0 = performance.now();
      const canvas = new OffscreenCanvas(width, height);
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(bitmap, 0, 0, width, height);
      try {
        bitmap.close();
      } catch { /* noop */ }
      const { data } = ctx.getImageData(0, 0, width, height); // off the main thread
      const tensor = rgbaToTensor(data, width, height, { mean, std, layout, normalize });
      const ms = performance.now() - t0;
      self.postMessage({ type: "result", id, tensor, width, height, ms }, [tensor.buffer]);
    } catch (err) {
      try {
        msg.bitmap?.close();
      } catch { /* noop */ }
      self.postMessage({ type: "error", id, message: err?.message || String(err) });
    }
  }
};
