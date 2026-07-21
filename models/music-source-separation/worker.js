// Music source separation worker — Demucs off the main thread.
// Model: MrCitron/demucs-v4-onnx / htdemucs.onnx (cc-by-nc-4.0) — Hybrid Transformer Demucs (htdemucs).
// It takes a STEREO waveform segment at 44.1 kHz [1, 2, 343980] (~7.8 s) and returns FOUR source stems
// [1, 4, 2, 343980] in the order [drums, bass, other, vocals]. The STFT is inside the graph, so we feed raw
// waveform and get waveform stems back — no manual spectrogram maths. onnxruntime-web runs the ONNX directly
// (a per-worker pin, like the other raw-ORT demos). DISTINCT from the built speech-separation (ConvTasNet
// separates two SPEAKERS): this splits MUSIC into instrument/vocal stems. Nothing leaves the tab.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const REPO = "MrCitron/demucs-v4-onnx";
const MODEL_URL = `https://huggingface.co/${REPO}/resolve/main/htdemucs.onnx`;
const CACHE_NAME = "demucs-onnx-cache";
export const SEG = 343980; // model's fixed segment length (samples per channel) = 7.8 s @ 44.1 kHz
export const SR = 44100;
export const STEMS = ["drums", "bass", "other", "vocals"];

let ort = null;
let session = null;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Fetch the 303 MB ONNX THROUGH Cache Storage under a key carrying the model-id path so lib/model-cache.js
// auto-inits on a returning visit with no Download click.
async function fetchCached(url, cache, onChunk) {
  const key = MODEL_URL;
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
  await import(TRANSFORMERS_URL); // shared runtime warm-up (keeps lib pin consistent)
  ort = await import("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.min.mjs");
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
  ort.env.wasm.numThreads = 1;
  const cache = await caches.open(CACHE_NAME);
  const bytes = await fetchCached(MODEL_URL, cache, (r, t) => {
    if (t) post({ type: "progress", p: { status: "progress", progress: (r / t) * 98 } });
  });
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  post({ type: "ready", device: "wasm" });
}

// Separate one segment. `ch0`/`ch1` are 44.1 kHz Float32 channels (length ≤ SEG); we zero-pad to SEG, run
// htdemucs, and return the 4 stems (each stereo, trimmed back to the input length).
async function separate(id, ch0, ch1, len) {
  await ensureLoaded();
  const n = Math.min(len, SEG);
  const buf = new Float32Array(2 * SEG); // [ch0(SEG), ch1(SEG)]
  buf.set(ch0.subarray(0, n), 0);
  buf.set(ch1.subarray(0, n), SEG);
  const input = new ort.Tensor("float32", buf, [1, 2, SEG]);
  const t0 = performance.now();
  const out = await session.run({ input });
  const y = out.output; // [1,4,2,SEG]
  const data = y.data;
  const transfer = [];
  const stems = STEMS.map((name, s) => {
    const base = s * 2 * SEG;
    const l = data.slice(base, base + n); // channel 0, trimmed to n
    const r = data.slice(base + SEG, base + SEG + n); // channel 1
    transfer.push(l.buffer, r.buffer);
    return { name, l, r };
  });
  post({
    type: "result",
    id,
    stems,
    len: n,
    ms: Math.round(performance.now() - t0),
    device: "wasm",
  }, transfer);
}

self.addEventListener("message", async (e) => {
  const d = e.data;
  try {
    if (d.type === "load") await ensureLoaded();
    else if (d.type === "separate") await separate(d.id, d.ch0, d.ch1, d.len);
  } catch (err) {
    post({ type: "error", id: d?.id, message: String(err?.message ?? err) });
  }
});
