// Model2Vec (potion-base-8M) static-embedding worker — all inference off the main thread.
//
// Model2Vec is NOT a transformer. It is a distilled LOOKUP TABLE: every token has one fixed vector,
// precomputed once (by distilling a sentence-transformer, then PCA + weighting). To embed a sentence
// you simply GATHER each token's vector and AVERAGE them — no attention, no layers, no forward pass
// through a network. That is why it is microseconds-fast on a plain CPU.
//
// Faithful reproduction (verified): the official ONNX export (minishlab/potion-base-8M, onnx/model.onnx)
// is a torch EmbeddingBag(mode=mean) + PCA + L2-normalize. Transformers.js can tokenize the model but
// its feature-extraction pipeline does not supply the ONNX's `offsets` input, so we run the ONNX
// DIRECTLY with onnxruntime-web and tokenize with the transformers.js AutoTokenizer. Inputs:
//   input_ids : int64 [total_tokens]   — every token id of every sentence, concatenated
//   offsets   : int64 [batch]          — start index of each sentence in input_ids (EmbeddingBag)
// Output: embeddings float32 [batch, 256], already L2-normalized (cosine == dot product).
//
// We tokenize with add_special_tokens:false to match Model2Vec's own encode() semantics (it bags the
// content sub-word vectors, no CLS/SEP). The ONNX is fetched THROUGH the Cache API so the shared
// auto-init loader detects it on-device, it works offline, and "clear cached model" removes it.

const MODEL_ID = "minishlab/potion-base-8M";
const ONNX_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/onnx/model.onnx`;
const MODEL_CACHE = "web-ai-showcase-model2vec";
const ORT_VER = "1.20.1";
const ORT_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/ort.webgpu.min.mjs`;
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5";

let ort = null;
let tokenizer = null;
let session = null;
const device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

// Fetch the ONNX via the Cache API (with real download progress) so it lands in Cache Storage under
// the model id — the shared cache layer then sees it as on-device (auto-init + offline + clear-cache).
async function fetchOnnx(onProgress) {
  const cached = await caches.match(ONNX_URL);
  if (cached) return new Uint8Array(await cached.arrayBuffer());
  const resp = await fetch(ONNX_URL);
  if (!resp.ok) throw new Error(`Model download failed (HTTP ${resp.status})`);
  const total = Number(resp.headers.get("content-length")) || 0;
  const reader = resp.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    if (total) onProgress?.({ status: "progress", progress: (loaded / total) * 100 });
  }
  const bytes = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    bytes.set(c, off);
    off += c.length;
  }
  try {
    const cache = await caches.open(MODEL_CACHE);
    await cache.put(
      ONNX_URL,
      new Response(bytes, { headers: { "content-type": "application/octet-stream" } }),
    );
  } catch { /* cache put best-effort; inference still proceeds */ }
  return bytes;
}

async function ensureLoaded(onProgress) {
  if (session) return;
  ort = await import(ORT_URL);
  ort.env.logLevel = "error"; // silence the ONNX graph-optimizer W-level notices; keep the console clean
  ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;
  ort.env.wasm.numThreads = 1;
  const { AutoTokenizer, env } = await import(TRANSFORMERS_URL);
  env.allowLocalModels = false;
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
  const bytes = await fetchOnnx(onProgress);
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  post({ type: "ready", device });
}

function tokenIds(text) {
  const enc = tokenizer(text, { add_special_tokens: false });
  const ids = Array.from(enc.input_ids.data, (x) => Number(x));
  return ids.length ? ids : [0]; // guard: never send an empty bag
}

// Embed a batch of texts → { embeddings:number[][] (256-d, unit), dim, tokenCounts, tokens?, ms }.
async function embed(id, texts, withTokens) {
  await ensureLoaded();
  const t0 = performance.now();
  const idLists = texts.map(tokenIds);
  const flat = [];
  const offsets = [];
  for (const ids of idLists) {
    offsets.push(flat.length);
    for (const t of ids) flat.push(BigInt(t));
  }
  const feeds = {
    input_ids: new ort.Tensor("int64", BigInt64Array.from(flat), [flat.length]),
    offsets: new ort.Tensor("int64", BigInt64Array.from(offsets.map(BigInt)), [offsets.length]),
  };
  const out = await session.run(feeds);
  const outT = out[session.outputNames[0]];
  const dim = outT.dims[outT.dims.length - 1];
  const data = Array.from(outT.data, Number);
  const embeddings = texts.map((_, i) => data.slice(i * dim, (i + 1) * dim));
  const tokenCounts = idLists.map((l) => l.length);
  let tokens = null;
  if (withTokens) {
    tokens = idLists.map((ids) =>
      ids.map((t) => tokenizer.decode([t], { skip_special_tokens: false }))
    );
  }
  const ms = performance.now() - t0;
  post({ type: "result", id, texts, embeddings, dim, tokenCounts, tokens, ms, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded((p) => post({ type: "progress", p }));
    } else if (type === "run") {
      await embed(e.data.id, e.data.texts, e.data.withTokens);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
