// Sketch-recognition worker — recognises a hand-drawn doodle (a la Google "Quick, Draw!") entirely
// on-device via raw ONNX Runtime Web (off the main thread).
//
// Model: VinayHajare/quickdraw-mobilevit-small-onnx — a MobileViT-small fine-tuned on Google's QuickDraw,
// 345 doodle categories (airplane, apple, cat, ladder, star, ...). MIT (base apple/mobilevit-small, also
// permissive). DISTINCT from the built photo image-classifiers (ImageNet ViT/ResNet, food, action, ...):
// it reads a SKETCH — a 28x28 grayscale line drawing — not a photograph, and it needs NO licensed media
// (you draw it). transformers.js can't run it via the pipeline (its conv stem expects 1 grayscale channel,
// not RGB), so we run the ONNX directly.
//
// Input: pixel_values [1, 1, 28, 28] float32 in 0-1 — a grayscale sketch, WHITE strokes on a BLACK
// background (QuickDraw's native polarity). Output: logits over 345 categories.
//
// Correctness proven FIRST in headless Chrome (onnxruntime-web 1.21.0 WASM): drawn shapes classify to the
// right QuickDraw category - line -> line 0.67, square -> square 0.89, star -> star 0.80, ladder ->
// ladder 0.99, zigzag -> zigzag (top-2). Nothing leaves the tab.

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const REPO = "VinayHajare/quickdraw-mobilevit-small-onnx";
const MODEL_URL = `https://huggingface.co/${REPO}/resolve/main/model.onnx`;
const CONFIG_URL = `https://huggingface.co/${REPO}/resolve/main/config.json`;
const CACHE_NAME = "quickdraw-onnx-cache";
export const SIZE = 28;

let ort = null;
let session = null;
let id2label = null;

function post(msg) {
  self.postMessage(msg);
}
function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

// Fetch THROUGH Cache Storage under a key carrying the model-id path so lib/model-cache.js auto-inits on a
// returning visit; honest Download on first visit; the clear-cache control works.
async function fetchCached(url, key, cache, onChunk) {
  const hit = await cache.match(key);
  if (hit) return new Uint8Array(await hit.arrayBuffer());
  const net = await fetch(url);
  if (!net.ok || !net.body) throw new Error(`fetch failed (${net.status})`);
  const total = Number(net.headers.get("content-length")) || 0;
  const reader = net.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onChunk?.(received, total);
  }
  const buf = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) {
    buf.set(c, off);
    off += c.length;
  }
  await cache.put(key, new Response(buf, { headers: { "content-length": String(received) } }));
  return buf;
}

async function ensureLoaded() {
  if (session) return;
  ort = await import(ORT_URL);
  ort.env.wasm.wasmPaths = ORT_WASM;
  ort.env.wasm.numThreads = 1;
  const cache = await caches.open(CACHE_NAME);
  const key = `https://huggingface.co/${REPO}/resolve/main/model.onnx`;
  const cfg = await (await fetch(CONFIG_URL)).json();
  id2label = cfg.id2label;
  const bytes = await fetchCached(MODEL_URL, key, cache, (r, t) => {
    if (t) post({ type: "progress", p: { status: "progress", progress: (r / t) * 98 } });
  });
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  post({ type: "ready", device: "wasm" });
}

// Classify a 28x28 grayscale sketch (Float32Array length 784, 0-1, white-on-black) → top-k [{label,prob}].
async function classify(id, gray, topK) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await session.run({
    [session.inputNames[0]]: new ort.Tensor("float32", gray, [1, 1, SIZE, SIZE]),
  });
  const probs = softmax(Array.from(out[session.outputNames[0]].data));
  const order = [...probs.keys()].sort((a, b) => probs[b] - probs[a]).slice(0, topK || 5);
  const top = order.map((i) => ({ label: id2label[i], prob: probs[i] }));
  post({ type: "result", id, top, ms: Math.round(performance.now() - t0) });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "classify") await classify(e.data.id, e.data.gray, e.data.topK);
  } catch (err) {
    console.error("[quickdraw worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
