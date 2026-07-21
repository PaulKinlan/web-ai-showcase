// Speech separation worker — the "cocktail party" problem: split a recording of TWO people talking at once
// into two clean single-speaker tracks, entirely on-device via raw ONNX Runtime Web (off the main thread).
//
// Why raw ORT and not transformers.js: transformers.js has no speech-separation task, so we run the ONNX
// graph directly with onnxruntime-web (a per-worker pin, like the other raw-ORT demos).
//
// Model: ConvTasNet trained on Libri2Mix (clean 2-speaker separation, 16 kHz) — a lightweight time-domain
// network (a learned encoder → masking → decoder, no STFT). Input "mixture" [1, samples] float32 mono at
// 16 kHz -> output "sources" [1, 2, samples]: the two separated speaker waveforms. DISTINCT from the built
// gtcrn speech-ENHANCEMENT demo (which removes noise from ONE speaker): this pulls TWO overlapping speakers
// apart. License: CC BY-SA 4.0 (welcomyou ONNX export; base JorisCos/ConvTasNet_Libri2Mix_sepclean_16k;
// derived from Libri2Mix / LibriSpeech) — a free license used here unmodified with attribution. Nothing
// leaves the tab.
//
// Correctness proven FIRST in headless Chrome (onnxruntime-web 1.21.0 WASM): a 3 s mix of two distinct
// speakers separated cleanly — output A correlated 0.80 with speaker 1 and 0.11 with speaker 2, output B
// correlated 0.96 with speaker 2 and 0.12 with speaker 1 (a clear one-speaker-per-output split), in ~0.9 s.

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const REPO = "welcomyou/convtasnet-libri2mix-16k-onnx";
const MODEL_URL = `https://huggingface.co/${REPO}/resolve/main/convtasnet_16k.onnx`;
const CACHE_NAME = "speech-separation-onnx-cache";
export const SR = 16000;

let ort = null;
let session = null;
let inName = null;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Fetch THROUGH Cache Storage under a key carrying the model-id path so lib/model-cache.js auto-inits on a
// returning visit; honest Download on first visit; the clear-cache control works.
async function fetchCached(url, cache, onChunk) {
  const key = `https://huggingface.co/${REPO}/resolve/main/convtasnet_16k.onnx`;
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
  ort.env.wasm.numThreads = 1; // no cross-origin isolation on GitHub Pages → single-threaded WASM.
  const cache = await caches.open(CACHE_NAME);
  const bytes = await fetchCached(MODEL_URL, cache, (r, t) => {
    if (t) post({ type: "progress", p: { status: "progress", progress: (r / t) * 98 } });
  });
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  inName = session.inputNames[0];
  post({ type: "ready", device: "wasm" });
}

// Separate a mono 16 kHz mixture (Float32Array) → two source Float32Arrays.
async function separate(id, mix) {
  await ensureLoaded();
  const t0 = performance.now();
  const N = mix.length;
  const out = await session.run({ [inName]: new ort.Tensor("float32", mix, [1, N]) });
  const o = out[session.outputNames[0]]; // [1, 2, N]
  const d = o.data;
  const half = Math.floor(d.length / 2);
  const s1 = new Float32Array(d.subarray(0, half));
  const s2 = new Float32Array(d.subarray(half, 2 * half));
  post({ type: "result", id, s1, s2, sr: SR, ms: Math.round(performance.now() - t0) }, [
    s1.buffer,
    s2.buffer,
  ]);
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "separate") await separate(e.data.id, e.data.mix);
  } catch (err) {
    console.error("[speech-separation worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
