// Learned sparse retrieval (SPLADE) worker — encodes text into a SPARSE bag of weighted vocabulary terms,
// entirely on-device. Unlike a dense embedding (a fixed vector of opaque numbers), SPLADE's output is a
// weight per WORD in the vocabulary — most zero — and it INCLUDES expansion terms the input never used
// (e.g. "pasta" -> also weights "noodles", "italy", "food"). That is what powers neural lexical search:
// documents and queries match on meaning-expanded terms, while staying an interpretable, invertible index.
//
// Model: Splade_PP_en_v1 (devve1/Splade_PP_en_v2_onnx — a faithful ONNX export). The weights are Apache-2.0
// (prithivida/Splade_PP_en_v1); Apache-2.0 permits redistribution, so they stay Apache-2.0 in the ONNX
// export. transformers.js has no SPLADE task, so we run the masked-LM ONNX directly via onnxruntime-web
// (a per-worker pin, like the other raw-ORT demos) and tokenize with a transformers.js AutoTokenizer.
//
// SPLADE aggregation (standard): given MLM logits [1, seq, vocab], the weight of term v is
//   w[v] = max over sequence positions i of log(1 + relu(logits[i][v]))
// masked by attention. Non-zero terms (typically a few dozen) are the sparse representation.
//
// Correctness proven FIRST in headless Chrome (onnxruntime-web 1.21.0 WASM): "A man is eating pasta at an
// Italian restaurant." -> pasta 2.69, italian 1.98, restaurant 1.45 + expansions noodles/italy/food;
// "The spacecraft entered orbit around Mars." -> mars 2.57, spacecraft 1.83, orbit 1.58 + expansions
// martian/nasa/space/orbital. Nothing leaves the tab.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const REPO = "devve1/Splade_PP_en_v2_onnx";
const MODEL_URL = `https://huggingface.co/${REPO}/resolve/main/model.onnx`;
const CACHE_NAME = "splade-onnx-cache";
const MIN_WEIGHT = 0.1; // drop near-zero terms

let ort = null;
let session = null;
let tokenizer = null;
let specialIds = null;

function post(msg) {
  self.postMessage(msg);
}

// Fetch the model THROUGH Cache Storage under a key carrying the model-id path so lib/model-cache.js
// auto-inits on a returning visit (the tokenizer files are cached by transformers.js under the same repo).
async function fetchCached(url, cache, onChunk) {
  const key = `https://huggingface.co/${REPO}/resolve/main/model.onnx`;
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
  const mod = await import(TRANSFORMERS_URL);
  ort = await import("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs");
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
  ort.env.wasm.numThreads = 1;
  tokenizer = await mod.AutoTokenizer.from_pretrained(REPO, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  specialIds = new Set(tokenizer.all_special_ids ?? []);
  const cache = await caches.open(CACHE_NAME);
  const bytes = await fetchCached(MODEL_URL, cache, (r, t) => {
    if (t) post({ type: "progress", p: { status: "progress", progress: (r / t) * 98 } });
  });
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  post({ type: "ready", device: "wasm" });
}

const bi = (a) => BigInt64Array.from(a, BigInt);

// Encode text → { terms:[{term,weight,expansion}], sparse:{id:weight}, ms }.
async function encode(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const enc = await tokenizer(text);
  const ids = Array.from(enc.input_ids.data, Number);
  const L = ids.length;
  const inputSet = new Set(ids.filter((v) => !specialIds.has(v)));
  const out = await session.run({
    input_ids: new ort.Tensor("int64", bi(ids), [1, L]),
    attention_mask: new ort.Tensor("int64", bi(ids.map(() => 1)), [1, L]),
    token_type_ids: new ort.Tensor("int64", bi(ids.map(() => 0)), [1, L]),
  });
  const logits = out.logits;
  const [, S, V] = logits.dims;
  const data = logits.data;
  // SPLADE pool: w[v] = max_i log(1 + relu(logits[i][v]))
  const w = new Float32Array(V);
  for (let i = 0; i < S; i++) {
    for (let v = 0; v < V; v++) {
      const x = data[i * V + v];
      if (x > 0) {
        const t = Math.log(1 + x);
        if (t > w[v]) w[v] = t;
      }
    }
  }
  const sparse = {};
  const kept = [];
  for (let v = 0; v < V; v++) {
    if (w[v] > MIN_WEIGHT) {
      sparse[v] = w[v];
      kept.push(v);
    }
  }
  kept.sort((a, b) => w[b] - w[a]);
  const terms = kept.slice(0, 40).map((v) => ({
    term: tokenizer.decode([v]).trim(),
    weight: +w[v].toFixed(3),
    expansion: !inputSet.has(v),
  })).filter((t) => t.term.length > 0);
  post({
    type: "result",
    id,
    terms,
    sparse,
    nonZero: kept.length,
    ms: Math.round(performance.now() - t0),
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "encode") await encode(e.data.id, e.data.text);
  } catch (err) {
    console.error("[splade worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
