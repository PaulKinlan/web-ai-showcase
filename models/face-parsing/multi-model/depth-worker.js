// Depth worker for the Face-Parsing multi-model demo — runs Depth Anything v2 (small) OFF the main
// thread and returns the normalized per-pixel depth as a transferable buffer. Kept separate from the
// face-parsing worker so each model owns its own worker + cache (invariant 4/15). Real transformers.js
// depth-estimation pipeline output: { depth: RawImage (0–255 grayscale), predicted_depth: Tensor }.
import { loadPipeline, pickDevice } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let device = "wasm";

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

async function ensureLoaded() {
  if (pipe) return;
  // Mirror the verified built depth-anything worker: fp16 on the WebGPU fast path, q8
  // (model_quantized.onnx) on the WASM/CPU fallback — fp16 compute is a WebGPU-only kernel path, so
  // q8 keeps depth actually runnable on the CPU. Probe for a REAL adapter first.
  device = await pickDevice("webgpu");
  const dtype = device === "webgpu" ? "fp16" : "q8";
  const loaded = await loadPipeline({
    task: "depth-estimation",
    model: "onnx-community/depth-anything-v2-small",
    backend: device,
    dtype,
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

async function run(id, imageURL) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await pipe(imageURL);
  const depth = out.depth; // RawImage, single channel 0..255
  const w = depth.width, h = depth.height;
  const ch = depth.channels ?? (depth.data.length / (w * h)) | 0;
  const gray = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) gray[i] = depth.data[i * ch];
  const ms = Math.round(performance.now() - t0);
  const buf = gray.buffer;
  post({ type: "result", id, width: w, height: h, depth: buf, ms, device }, [buf]);
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
