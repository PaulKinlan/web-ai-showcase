// DETR panoptic-segmentation worker — runs ALL inference off the main thread so the control UI stays
// responsive (modern-web-guidance: break-up-long-tasks). One forward pass returns a PANOPTIC parse of
// the scene: a set of NON-overlapping segments, each with a class label, a confidence score, and its
// own pixel mask. Panoptic = things (countable instances: two separate cats become two segments) PLUS
// stuff (amorphous regions: sky, grass, wall). This is distinct from SegFormer's dense per-class
// semantic map (no instances) and SAM's single point-prompted mask.
//
// Model: Xenova/detr-resnet-50-panoptic (task: image-segmentation), WASM backend, q8.
// The default `image-segmentation` call on this model IS the panoptic parse — we call it with no
// subtask (transformers.js 3.7.5 throws on an explicit subtask arg for this pipeline). The pipeline
// returns [{ score, label, mask: RawImage }, …]; we composite the per-segment masks into ONE colour-
// coded RGBA overlay (argmax by mask strength ⇒ one segment per pixel) and hand back per-segment
// metadata (label, score, area, colour, thing-vs-stuff). No invented API.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let device = "wasm";

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// The 80 COCO "thing" classes (countable objects). Everything else the panoptic model labels is
// "stuff" (amorphous background regions: sky, grass, road, wall, …). Panoptic splits the two.
const COCO_THINGS = new Set([
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "dining table",
  "toilet",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush",
]);

function isThing(label) {
  return COCO_THINGS.has(label);
}

// Deterministic, well-separated colour per SEGMENT index (golden-angle hue walk). Same index → same
// colour every run so the legend and overlay always agree. All colours are dark enough that white
// label text on top passes WCAG AA (l capped ≤ .5).
function colorForIndex(i) {
  const h = (i * 137.508) % 360;
  const s = i % 2 ? 0.55 : 0.7;
  const l = i % 3 === 0 ? 0.46 : 0.4;
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
    model: "Xenova/detr-resnet-50-panoptic",
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
  // Default call = the panoptic parse for this model. No subtask arg (3.7.5 throws on it here).
  const output = await pipe(imageURL); // [{ score, label, mask: RawImage }, …]
  if (!Array.isArray(output) || output.length === 0) {
    throw new Error("panoptic segmentation returned no segments");
  }

  const w = output[0].mask.width, h = output[0].mask.height;
  const n = w * h;
  const overlay = new Uint8ClampedArray(n * 4);
  // Per pixel, keep the winning segment (strongest mask value) so the composite is a clean panoptic
  // parse: exactly one segment owns each pixel.
  const bestVal = new Uint8Array(n);
  const bestSeg = new Int16Array(n).fill(-1);
  const segments = [];

  output.forEach((seg, si) => {
    const alpha = maskToAlpha(seg.mask);
    for (let i = 0; i < n; i++) {
      const v = alpha[i];
      if (v > bestVal[i]) {
        bestVal[i] = v;
        bestSeg[i] = si;
      }
    }
  });

  // Colour + area from the RESOLVED ownership (post argmax), so areas sum to ≤ 100% with no double
  // counting. Paint the overlay from the winner map.
  const areaPx = new Array(output.length).fill(0);
  for (let i = 0; i < n; i++) {
    const si = bestSeg[i];
    if (si < 0) continue;
    areaPx[si]++;
  }
  output.forEach((seg, si) => {
    const [r, g, b] = colorForIndex(si);
    segments.push({
      index: si,
      label: seg.label ?? "region",
      score: typeof seg.score === "number" ? seg.score : null,
      thing: isThing(seg.label),
      r,
      g,
      b,
      areaPx: areaPx[si],
      coverage: areaPx[si] / n,
    });
  });
  for (let i = 0; i < n; i++) {
    const si = bestSeg[i];
    if (si < 0) continue; // unclaimed pixel stays transparent → photo shows through
    const [r, g, b] = colorForIndex(si);
    const o = i * 4;
    overlay[o] = r;
    overlay[o + 1] = g;
    overlay[o + 2] = b;
    overlay[o + 3] = 255;
  }

  // Sort segments biggest-first for the legend/table; keep index for colour stability.
  segments.sort((a, b) => b.areaPx - a.areaPx);

  // Composite the colour-coded panoptic overlay into an ImageBitmap HERE (off the main thread) so the
  // page only drawImage()s it — no full-res putImageData on the main thread per render/opacity drag
  // (invariant 15). The raw RGBA buffer is still sent: the index isolate toggle + wild/multi-model
  // extract read its pixels per segment colour.
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
      segments,
      segmentCount: segments.length,
      thingCount: segments.filter((s) => s.thing).length,
      stuffCount: segments.filter((s) => !s.thing).length,
      ms,
      compositeMs,
      device,
    },
    transfer,
  );
}

// Render the RGBA overlay into an OffscreenCanvas and hand back a transferable ImageBitmap. Guarded:
// if a worker realm lacks OffscreenCanvas we return null and the page falls back to its measured
// main-thread putImageData path (overlayToDrawable in panoptic.js).
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
