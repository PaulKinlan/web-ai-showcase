// Swin2SR super-resolution worker — runs ALL inference off the main thread so the control UI stays
// responsive. The model is fully-convolutional but memory grows with input area, so we TILE the input
// into overlapping patches, upscale each 2×, and stitch the non-overlap cores back together. Each tile
// is timed so the page can show honest per-tile progress.
//
// Model: Xenova/swin2SR-classical-sr-x2-64 (task: image-to-image), WASM backend, q8. Output is a
// RawImage at 2× the input resolution. We import the SHARED loader from lib/webai.js — no invented API.

import { loadPipeline, TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let device = "wasm";
let RawImage = null;

const SCALE = 2; // this checkpoint is the ×2 classical SR model
const OVERLAP = 8; // input-px border shared between neighbouring tiles, cropped away after upscale

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

async function ensureLoaded() {
  if (pipe) return;
  const mod = await import(TRANSFORMERS_URL);
  RawImage = mod.RawImage;
  const loaded = await loadPipeline({
    task: "image-to-image",
    model: "Xenova/swin2SR-classical-sr-x2-64",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

// Copy an (sx,sy,sw,sh) RGBA sub-rectangle out of a flat RGBA buffer.
function extractTile(rgba, W, sx, sy, sw, sh) {
  const out = new Uint8ClampedArray(sw * sh * 4);
  for (let y = 0; y < sh; y++) {
    const srcStart = ((sy + y) * W + sx) * 4;
    out.set(rgba.subarray(srcStart, srcStart + sw * 4), y * sw * 4);
  }
  return out;
}

// Blit a RawImage tile (RGB or RGBA) into the destination RGBA buffer, taking only its valid core
// (dropping the upscaled overlap borders) and placing it at (dx,dy) in the output image.
function blitCore(out, OW, tileImg, cropL, cropT, validW, validH, dx, dy) {
  const tw = tileImg.width;
  const ch = tileImg.channels;
  const src = tileImg.data;
  for (let y = 0; y < validH; y++) {
    const sy = cropT + y;
    for (let x = 0; x < validW; x++) {
      const sx = cropL + x;
      const s = (sy * tw + sx) * ch;
      const d = ((dy + y) * OW + (dx + x)) * 4;
      out[d] = src[s];
      out[d + 1] = ch >= 2 ? src[s + 1] : src[s];
      out[d + 2] = ch >= 3 ? src[s + 2] : src[s];
      out[d + 3] = ch === 4 ? src[s + 3] : 255;
    }
  }
}

async function run(id, rgba, W, H, tile) {
  await ensureLoaded();
  const TILE = tile || 128;
  const OW = W * SCALE, OH = H * SCALE;
  const out = new Uint8ClampedArray(OW * OH * 4);

  const xs = [];
  for (let x = 0; x < W; x += TILE) xs.push(x);
  const ys = [];
  for (let y = 0; y < H; y += TILE) ys.push(y);
  const total = xs.length * ys.length;

  const tiles = [];
  const t0 = performance.now();
  let done = 0;
  for (const ty of ys) {
    for (const tx of xs) {
      const sx = Math.max(0, tx - OVERLAP);
      const sy = Math.max(0, ty - OVERLAP);
      const ex = Math.min(W, tx + TILE + OVERLAP);
      const ey = Math.min(H, ty + TILE + OVERLAP);
      const tw = ex - sx, th = ey - sy;

      const tileData = extractTile(rgba, W, sx, sy, tw, th);
      const raw = new RawImage(tileData, tw, th, 4);
      const tt = performance.now();
      const up = await pipe(raw); // → RawImage at SCALE× (tw,th)
      const tileMs = Math.round(performance.now() - tt);

      const cropL = (tx - sx) * SCALE;
      const cropT = (ty - sy) * SCALE;
      const validW = Math.min(TILE, W - tx) * SCALE;
      const validH = Math.min(TILE, H - ty) * SCALE;
      blitCore(out, OW, up, cropL, cropT, validW, validH, tx * SCALE, ty * SCALE);

      done++;
      tiles.push({ ms: tileMs, w: tw, h: th });
      post({ type: "tile", id, done, total, tileMs });
    }
  }
  const ms = Math.round(performance.now() - t0);
  post(
    { type: "result", id, rgba: out, width: OW, height: OH, tiles, ms, device, scale: SCALE },
    [out.buffer],
  );
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded();
    } else if (type === "run") {
      await run(e.data.id, e.data.rgba, e.data.width, e.data.height, e.data.tile);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
