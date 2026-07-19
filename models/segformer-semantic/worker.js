// SegFormer semantic-segmentation worker — runs ALL inference off the main thread. One forward pass
// returns a dense per-pixel label map over the 150 ADE20K classes; here we turn the pipeline's
// per-class masks into (a) a colour-coded RGBA overlay (the "map"), (b) a stable colour per class,
// and (c) per-class pixel coverage — everything the page's map, legend, and "see inside" need.
//
// Model: Xenova/segformer-b0-finetuned-ade-512-512 (task: image-segmentation, subtask semantic).
// WASM backend, q8. We use the documented `image-segmentation` pipeline — the real API, not invented.
// The pipeline returns [{ label, mask: RawImage }, …], one entry per class present in the image.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let device = "wasm";

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Deterministic, well-separated colour per class INDEX (golden-angle hue walk). Same index → same
// colour every run, so the legend and overlay always agree.
function colorForIndex(i) {
  const h = (i * 137.508) % 360;
  const s = i % 2 ? 0.52 : 0.68;
  const l = i % 3 === 0 ? 0.58 : 0.48;
  return hslToRgb(h / 360, s, l);
}
function hslToRgb(h, s, l) {
  if (s === 0) {
    const v = Math.round(l * 255);
    return [v, v, v];
  }
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const hk = (t) => {
    let x = t;
    if (x < 0) x += 1;
    if (x > 1) x -= 1;
    if (x < 1 / 6) return p + (q - p) * 6 * x;
    if (x < 1 / 2) return q;
    if (x < 2 / 3) return p + (q - p) * (2 / 3 - x) * 6;
    return p;
  };
  return [
    Math.round(hk(h + 1 / 3) * 255),
    Math.round(hk(h) * 255),
    Math.round(hk(h - 1 / 3) * 255),
  ];
}

async function ensureLoaded() {
  if (pipe) return;
  const loaded = await loadPipeline({
    task: "image-segmentation",
    model: "Xenova/segformer-b0-finetuned-ade-512-512",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

// Pull a single-channel 0..255 mask array out of a RawImage regardless of channel count.
function maskToAlpha(mask) {
  const w = mask.width, h = mask.height;
  const ch = mask.channels ?? (mask.data.length / (w * h)) | 0;
  if (ch === 1) {
    return mask.data instanceof Uint8Array ? mask.data : Uint8Array.from(mask.data);
  }
  const a = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) a[i] = mask.data[i * ch];
  return a;
}

async function run(id, imageURL) {
  await ensureLoaded();
  const t0 = performance.now();
  const output = await pipe(imageURL); // [{ score, label, mask: RawImage }, …]
  if (!Array.isArray(output) || output.length === 0) {
    throw new Error("segmentation returned no classes");
  }

  const w = output[0].mask.width, h = output[0].mask.height;
  const n = w * h;
  const overlay = new Uint8ClampedArray(n * 4);
  // Track, per pixel, the winning class (largest mask value) so overlapping/soft masks resolve to a
  // single argmax label — true semantic segmentation is one class per pixel.
  const bestVal = new Uint8Array(n);
  const classes = [];

  output.forEach((seg, ci) => {
    const alpha = maskToAlpha(seg.mask);
    const [r, g, b] = colorForIndex(ci);
    let pixels = 0;
    for (let i = 0; i < n; i++) {
      const v = alpha[i];
      if (v >= 128) pixels++;
      if (v > bestVal[i]) {
        bestVal[i] = v;
        const o = i * 4;
        overlay[o] = r;
        overlay[o + 1] = g;
        overlay[o + 2] = b;
        overlay[o + 3] = 255;
      }
    }
    classes.push({ label: seg.label, r, g, b, pixels, coverage: pixels / n });
  });

  // Any pixel no mask claimed (rare) stays transparent so the photo shows through there.
  classes.sort((a, b) => b.pixels - a.pixels);

  const ms = Math.round(performance.now() - t0);
  const buf = overlay.buffer;
  post(
    {
      type: "result",
      id,
      width: w,
      height: h,
      overlay: buf,
      classes,
      classCount: classes.length,
      ms,
      device,
    },
    [buf],
  );
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await run(e.data.id, e.data.image);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
