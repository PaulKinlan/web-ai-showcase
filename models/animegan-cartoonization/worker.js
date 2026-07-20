// AnimeGANv2 cartoonization worker — ALL inference AND the dense output composite off the main thread
// via raw ONNX Runtime Web.
//
// Why raw ORT and not transformers.js: transformers.js registers no AnimeGAN model class and there is no
// photo-stylization / cartoonization pipeline task. The ONNX export is a clean image→image graph, so we
// run it directly with onnxruntime-web and hand-write the pre/post a pipeline would own: (1) resize the
// photo (long side ≤ maxSide, rounded to a multiple of 32 for the generator's down/upsampling) and pack
// it as **NHWC** [1,H,W,3] float32 in [-1,1] (AnimeGAN is a TensorFlow export — channels-last, tanh
// range), and (2) map the tanh output back from [-1,1] to an RGBA image. This is the isolated per-worker
// ORT-web escape hatch (like models/scunet-image-denoising/worker.js) — onnxruntime-web is pinned HERE
// only, never in shared libs.
//
// Models: vumichien/AnimeGANv2_{Hayao,Shinkai,Paprika} (one ~8.4 MB ONNX per style, Apache-2.0).
// AnimeGANv2 (Chen et al.) — a GAN that repaints a photo in a learned anime style (Miyazaki/Hayao,
// Makoto Shinkai, Satoshi Kon/Paprika). Input "generator_input:0" [1,H,W,3] float32 in [-1,1] (dynamic
// H,W). Output "generator/G_MODEL/out_layer/Tanh:0" [1,H,W,3] float32 in [-1,1]. Styles load lazily
// (only the ones you pick are downloaded). Everything stays on-device: the image never leaves the tab.
//
// Runnability was proven FIRST in headless Chrome: on real photos (dog, street) each style produced a
// genuine anime repaint — dynamic-size verified (256², 384×256, 512² all ran), normalized RMSE(out,in)
// ~0.10–0.12 (a real style change, not identity, not a colour curve). Inference ~1.3 s @256² to ~5 s
// @512² on single-thread WASM.

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const STYLES = {
  hayao: { repo: "vumichien/AnimeGANv2_Hayao", file: "AnimeGANv2_Hayao.onnx" },
  shinkai: { repo: "vumichien/AnimeGANv2_Shinkai", file: "AnimeGANv2_Shinkai.onnx" },
  paprika: { repo: "vumichien/AnimeGANv2_Paprika", file: "AnimeGANv2_Paprika.onnx" },
};
const DEFAULT_STYLE = "hayao";
const CACHE_NAME = "animegan-onnx-cache";

let ort = null;
let device = "wasm";
const sessions = new Map(); // style -> { session, inputName, outputName }

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Fetch weights THROUGH Cache Storage so lib/model-cache.js (which scans caches for
// "/vumichien/AnimeGANv2_Hayao/") sees them → auto-init on a returning visit, honest Download on first
// visit, and the per-model "clear cache" control all work. Streams download progress.
async function fetchModelBytes(url, onChunk) {
  const cache = await caches.open(CACHE_NAME);
  const hit = await cache.match(url);
  if (hit) return new Uint8Array(await hit.arrayBuffer());
  const net = await fetch(url);
  if (!net.ok || !net.body) throw new Error(`model fetch failed (${net.status})`);
  const total = Number(net.headers.get("content-length")) || 0;
  const reader = net.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total) onChunk?.((received / total) * 100);
  }
  const buf = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) {
    buf.set(c, off);
    off += c.length;
  }
  await cache.put(
    url,
    new Response(buf, {
      headers: { "content-length": String(received), "content-type": "application/octet-stream" },
    }),
  );
  return buf;
}

async function ensureStyle(style) {
  if (sessions.has(style)) return sessions.get(style);
  if (!ort) {
    ort = await import(ORT_URL);
    ort.env.wasm.wasmPaths = ORT_WASM;
    ort.env.wasm.numThreads = 1; // no cross-origin isolation on GitHub Pages → single-threaded WASM.
  }
  const s = STYLES[style] || STYLES[DEFAULT_STYLE];
  const url = `https://huggingface.co/${s.repo}/resolve/main/${s.file}`;
  const bytes = await fetchModelBytes(
    url,
    (p) => post({ type: "progress", p: { status: "progress", progress: p } }),
  );
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  const rec = { session, inputName: session.inputNames[0], outputName: session.outputNames[0] };
  sessions.set(style, rec);
  return rec;
}

async function stylize(id, bitmap, opts) {
  const style = opts?.style || DEFAULT_STYLE;
  const rec = await ensureStyle(style);
  const iw = bitmap.width, ih = bitmap.height;
  const maxSide = opts?.maxSide ?? 512;
  const scale = Math.min(maxSide / Math.max(iw, ih), 1);
  const w = Math.max(32, Math.round(iw * scale / 32) * 32);
  const h = Math.max(32, Math.round(ih * scale / 32) * 32);
  const t0 = performance.now();

  const c = new OffscreenCanvas(w, h);
  const cctx = c.getContext("2d", { willReadFrequently: true });
  cctx.drawImage(bitmap, 0, 0, iw, ih, 0, 0, w, h);
  bitmap.close?.();
  const src = cctx.getImageData(0, 0, w, h).data;
  const N = w * h;
  // NHWC, [-1,1]
  const feed = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    feed[i * 3] = src[i * 4] / 127.5 - 1;
    feed[i * 3 + 1] = src[i * 4 + 1] / 127.5 - 1;
    feed[i * 3 + 2] = src[i * 4 + 2] / 127.5 - 1;
  }

  const feeds = {};
  feeds[rec.inputName] = new ort.Tensor("float32", feed, [1, h, w, 3]);
  const results = await rec.session.run(feeds);
  const out = results[rec.outputName].data; // NHWC [1,h,w,3] in [-1,1]
  const infMs = Math.round(performance.now() - t0);

  // Map tanh [-1,1] → RGBA, and measure how much the image changed (mean abs per-pixel delta in [0,1]).
  const rgba = new Uint8ClampedArray(N * 4);
  let deltaSum = 0;
  for (let i = 0; i < N; i++) {
    const r = (out[i * 3] + 1) * 0.5;
    const g = (out[i * 3 + 1] + 1) * 0.5;
    const b = (out[i * 3 + 2] + 1) * 0.5;
    rgba[i * 4] = Math.max(0, Math.min(1, r)) * 255;
    rgba[i * 4 + 1] = Math.max(0, Math.min(1, g)) * 255;
    rgba[i * 4 + 2] = Math.max(0, Math.min(1, b)) * 255;
    rgba[i * 4 + 3] = 255;
    const ir = (feed[i * 3] + 1) * 0.5,
      ig = (feed[i * 3 + 1] + 1) * 0.5,
      ib = (feed[i * 3 + 2] + 1) * 0.5;
    deltaSum += (Math.abs(r - ir) + Math.abs(g - ig) + Math.abs(b - ib)) / 3;
  }
  const styleDelta = deltaSum / N;

  const outC = new OffscreenCanvas(w, h);
  outC.getContext("2d").putImageData(new ImageData(rgba, w, h), 0, 0);
  const styledBmp = outC.transferToImageBitmap();

  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    styledBmp,
    w,
    h,
    imgW: iw,
    imgH: ih,
    style,
    styleDelta,
    ms,
    infMs,
    device,
  }, [styledBmp]);
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureStyle(e.data.style || DEFAULT_STYLE);
      post({ type: "ready", device });
    } else if (type === "run") {
      await stylize(e.data.id, e.data.bitmap, e.data.opts);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
