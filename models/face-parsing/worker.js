// Face-parsing worker — runs ALL inference off the main thread. One SegFormer forward pass returns a
// dense per-pixel label map over 19 FACE-PART classes (skin, nose, eyes, brows, ears, lips, hair,
// hat, neck, clothing…). Here we turn the pipeline's per-class masks into (a) a colour-coded RGBA
// overlay composited into an ImageBitmap off-thread, (b) a STABLE colour per face-part class, and
// (c) per-class pixel coverage — everything the page's map, legend, and "see inside" need.
//
// Model: Xenova/face-parsing (task: image-segmentation, subtask semantic; SegFormer backbone finetuned
// on CelebAMask-HQ). WASM backend, q8. We use the documented `image-segmentation` pipeline — the real
// API. The pipeline returns [{ label, mask: RawImage }, …], one entry per face-part class present.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let device = "wasm";

// The 19 CelebAMask-HQ face-part classes, in the model's own id2label order. Colouring by this FIXED
// index (not by detection order) means "hair" is the same colour in every image — vital for a legend
// that stays legible across faces.
const FACE_CLASSES = [
  "background",
  "skin",
  "nose",
  "eye_g",
  "l_eye",
  "r_eye",
  "l_brow",
  "r_brow",
  "l_ear",
  "r_ear",
  "mouth",
  "u_lip",
  "l_lip",
  "hair",
  "hat",
  "ear_r",
  "neck_l",
  "neck",
  "cloth",
];
// Human-friendly names for the legend / readouts.
const PRETTY = {
  background: "background",
  skin: "skin",
  nose: "nose",
  eye_g: "glasses",
  l_eye: "left eye",
  r_eye: "right eye",
  l_brow: "left brow",
  r_brow: "right brow",
  l_ear: "left ear",
  r_ear: "right ear",
  mouth: "mouth",
  u_lip: "upper lip",
  l_lip: "lower lip",
  hair: "hair",
  hat: "hat",
  ear_r: "earring",
  neck_l: "necklace",
  neck: "neck",
  cloth: "clothing",
};
const INDEX_OF = new Map(FACE_CLASSES.map((l, i) => [l, i]));

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Deterministic, well-separated colour per class INDEX (golden-angle hue walk). Same face part → same
// colour every run, so legend and overlay always agree. background (index 0) stays transparent.
function colorForIndex(i) {
  const h = (i * 137.508 + 20) % 360;
  const s = i % 2 ? 0.55 : 0.72;
  const l = i % 3 === 0 ? 0.6 : 0.5;
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
    model: "Xenova/face-parsing",
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
    throw new Error("face parsing returned no classes");
  }

  const w = output[0].mask.width, h = output[0].mask.height;
  const n = w * h;
  const overlay = new Uint8ClampedArray(n * 4);
  // Per-pixel argmax over the soft masks — semantic segmentation is one class per pixel.
  const bestVal = new Uint8Array(n);
  const classes = [];

  output.forEach((seg) => {
    const idx = INDEX_OF.has(seg.label) ? INDEX_OF.get(seg.label) : 0;
    const alpha = maskToAlpha(seg.mask);
    const isBg = seg.label === "background";
    const [r, g, b] = colorForIndex(idx);
    let pixels = 0;
    for (let i = 0; i < n; i++) {
      const v = alpha[i];
      if (v >= 128) pixels++;
      if (v > bestVal[i]) {
        bestVal[i] = v;
        const o = i * 4;
        if (isBg) {
          overlay[o + 3] = 0; // background stays transparent so the photo shows through
        } else {
          overlay[o] = r;
          overlay[o + 1] = g;
          overlay[o + 2] = b;
          overlay[o + 3] = 255;
        }
      }
    }
    classes.push({
      label: seg.label,
      pretty: PRETTY[seg.label] || seg.label,
      classIndex: idx,
      r,
      g,
      b,
      isBackground: isBg,
      pixels,
      coverage: pixels / n,
    });
  });

  classes.sort((a, b) => b.pixels - a.pixels);

  // Composite the colour-coded overlay into an ImageBitmap HERE (off the main thread) so the page only
  // has to drawImage() it — no full-res putImageData on the main thread on every render/opacity drag
  // (invariant 15: dense-output composite in the worker, transfer an ImageBitmap back). We still send
  // the raw RGBA buffer too for the practical/multi-model pages' per-class hit-testing.
  const { overlayBitmap, compositeMs } = buildOverlayBitmap(overlay, w, h);

  const ms = Math.round(performance.now() - t0);
  const buf = overlay.buffer;
  const transfer = [buf];
  if (overlayBitmap) transfer.push(overlayBitmap);
  post(
    {
      type: "result",
      id,
      width: w,
      height: h,
      overlay: buf,
      overlayBitmap,
      classes,
      classCount: classes.length,
      ms,
      compositeMs,
      device,
    },
    transfer,
  );
}

function buildOverlayBitmap(overlay, w, h) {
  if (typeof OffscreenCanvas === "undefined") return { overlayBitmap: null, compositeMs: 0 };
  try {
    const t = performance.now();
    const oc = new OffscreenCanvas(w, h);
    oc.getContext("2d").putImageData(new ImageData(overlay, w, h), 0, 0);
    const overlayBitmap = oc.transferToImageBitmap();
    return { overlayBitmap, compositeMs: +(performance.now() - t).toFixed(2) };
  } catch {
    return { overlayBitmap: null, compositeMs: 0 };
  }
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
