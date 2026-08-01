// Pure mask analysis/compositing shared by the MediaPipe worker and deterministic tests.
// Dense per-pixel work stays off the main thread in production.

const ACCENT = [57, 73, 171];

export function clamp01(value) {
  return Math.min(1, Math.max(0, Number(value)));
}

export function summarizeConfidence(confidence, threshold, point, width, height) {
  const cut = clamp01(threshold);
  let selected = 0;
  let sum = 0;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  for (let i = 0; i < confidence.length; i++) {
    const p = clamp01(confidence[i]);
    if (p < cut) continue;
    selected++;
    sum += p;
    const x = i % width;
    const y = Math.floor(i / width);
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }
  const px = Math.min(width - 1, Math.max(0, Math.round(clamp01(point.x) * (width - 1))));
  const py = Math.min(height - 1, Math.max(0, Math.round(clamp01(point.y) * (height - 1))));
  return {
    selectedPixels: selected,
    coverage: confidence.length ? selected / confidence.length : 0,
    meanSelectedConfidence: selected ? sum / selected : 0,
    targetConfidence: clamp01(confidence[py * width + px] ?? 0),
    bbox: selected ? { minX, minY, maxX, maxY } : null,
  };
}

export function composeMask(source, confidence, options = {}) {
  if (source.length !== confidence.length * 4) {
    throw new Error("source RGBA and confidence mask dimensions do not match");
  }
  const mode = options.mode || "overlay";
  const opacity = clamp01(options.opacity ?? 0.58);
  const threshold = clamp01(options.threshold ?? 0.5);
  const output = new Uint8ClampedArray(source.length);
  for (let i = 0, q = 0; i < confidence.length; i++, q += 4) {
    const p = clamp01(confidence[i]);
    const selected = p >= threshold;
    const r = source[q];
    const g = source[q + 1];
    const b = source[q + 2];
    if (mode === "cutout") {
      output[q] = r;
      output[q + 1] = g;
      output[q + 2] = b;
      output[q + 3] = selected ? Math.round(255 * p) : 0;
    } else if (mode === "mask") {
      output[q] = Math.round(ACCENT[0] * p + 27 * (1 - p));
      output[q + 1] = Math.round(ACCENT[1] * p + 27 * (1 - p));
      output[q + 2] = Math.round(ACCENT[2] * p + 31 * (1 - p));
      output[q + 3] = 255;
    } else if (mode === "spotlight") {
      if (selected) {
        output[q] = r;
        output[q + 1] = g;
        output[q + 2] = b;
      } else {
        const gray = Math.round((r * 0.2126 + g * 0.7152 + b * 0.0722) * 0.38);
        output[q] = gray;
        output[q + 1] = gray;
        output[q + 2] = gray;
      }
      output[q + 3] = 255;
    } else {
      const mix = opacity * p;
      output[q] = Math.round(r * (1 - mix) + ACCENT[0] * mix);
      output[q + 1] = Math.round(g * (1 - mix) + ACCENT[1] * mix);
      output[q + 2] = Math.round(b * (1 - mix) + ACCENT[2] * mix);
      output[q + 3] = 255;
    }
  }
  return output;
}
