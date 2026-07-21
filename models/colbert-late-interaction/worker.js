// Late-interaction retrieval (ColBERT) worker — scores how well a document answers a query by matching
// them TOKEN BY TOKEN, entirely on-device via raw ONNX Runtime Web (off the main thread).
//
// Two neural-retrieval paradigms are already in the catalogue: DENSE embeddings (one vector per text) and
// SPARSE SPLADE (a weight per vocabulary word). ColBERT is the third: LATE INTERACTION. It keeps a small
// vector for EVERY token, and scores a query against a document with MaxSim — for each query token, take
// its best match among the document's tokens, and sum. That makes the match INTERPRETABLE: you can see
// which document word each query word latched onto ("Hamlet" -> "Hamlet", "wrote" -> "wrote").
//
// Model: answerdotai/answerai-colbert-small-v1 (vespa_colbert.onnx export) — a compact BERT-based ColBERT
// that projects each token to a 96-d, L2-normalised vector. Apache-2.0. transformers.js has no ColBERT
// task, so we run the ONNX directly (a per-worker onnxruntime-web pin) and tokenize with a transformers.js
// tokenizer. Queries are prefixed with the [Q] marker (token 1), documents with [D] (token 2), per ColBERT.
//
// Correctness proven FIRST in headless Chrome (onnxruntime-web 1.21.0 WASM): for "who wrote the play
// Hamlet", the Shakespeare/Hamlet documents scored highest (7.6 / 7.5) and unrelated documents (a recipe,
// photosynthesis) lower (~6.6-6.7) - correct ranking. This is a simplified query encoding (no [MASK]
// query augmentation); the per-token alignment it shows is authentic. Nothing leaves the tab.

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const REPO = "answerdotai/answerai-colbert-small-v1";
const MODEL_URL = `https://huggingface.co/${REPO}/resolve/main/vespa_colbert.onnx`;
const CACHE_NAME = "colbert-onnx-cache";
const Q_MARKER = 1, D_MARKER = 2; // [unused0]=query, [unused1]=document

let ort = null;
let session = null;
let tokenizer = null;
let specialIds = null;

function post(msg) {
  self.postMessage(msg);
}

async function fetchCached(url, cache, onChunk) {
  const key = `https://huggingface.co/${REPO}/resolve/main/vespa_colbert.onnx`;
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
  const mod = await import("/web-ai-showcase/lib/webai.js");
  const tj = await import(mod.TRANSFORMERS_URL);
  ort = await import(ORT_URL);
  ort.env.wasm.wasmPaths = ORT_WASM;
  ort.env.wasm.numThreads = 1;
  tokenizer = await tj.AutoTokenizer.from_pretrained(REPO, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  specialIds = new Set(tokenizer.all_special_ids ?? [101, 102, 0]);
  const cache = await caches.open(CACHE_NAME);
  const bytes = await fetchCached(MODEL_URL, cache, (r, t) => {
    if (t) post({ type: "progress", p: { status: "progress", progress: (r / t) * 98 } });
  });
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  post({ type: "ready", device: "wasm" });
}

const bi = (a) => BigInt64Array.from(a, BigInt);

// Encode text (query or document) → { tokens:[{str, emb, content}] } with L2-normalised 96-d per-token embeddings.
async function encode(text, marker) {
  const enc = await tokenizer(text);
  let ids = Array.from(enc.input_ids.data, Number);
  ids = [ids[0], marker, ...ids.slice(1)]; // [CLS] [marker] ... [SEP]
  const L = ids.length;
  const out = await session.run({
    input_ids: new ort.Tensor("int64", bi(ids), [1, L]),
    attention_mask: new ort.Tensor("int64", bi(ids.map(() => 1)), [1, L]),
  });
  const t = out[session.outputNames[0]];
  const dim = t.dims[2];
  const data = t.data;
  const tokens = [];
  for (let i = 0; i < L; i++) {
    let s = 0;
    const v = new Float32Array(dim);
    for (let k = 0; k < dim; k++) {
      v[k] = data[i * dim + k];
      s += v[k] * v[k];
    }
    s = Math.sqrt(s) || 1;
    for (let k = 0; k < dim; k++) v[k] /= s;
    const isMarker = i === 1;
    const content = !specialIds.has(ids[i]) && !isMarker;
    tokens.push({ str: content ? tokenizer.decode([ids[i]]).trim() : "", emb: v, content });
  }
  return { tokens, dim };
}

// Score a query against a document with MaxSim over CONTENT tokens; also return the per-query-token alignment.
async function score(id, query, document) {
  await ensureLoaded();
  const t0 = performance.now();
  const Q = await encode(query, Q_MARKER);
  const D = await encode(document, D_MARKER);
  const qTok = Q.tokens.filter((t) => t.content);
  const dTok = D.tokens.filter((t) => t.content);
  const align = [];
  const docMax = new Array(dTok.length).fill(-1e9); // per-doc-token best sim to any query token
  let total = 0;
  for (const q of qTok) {
    let best = -1e9, bestJ = -1;
    for (let j = 0; j < dTok.length; j++) {
      let dot = 0;
      const d = dTok[j].emb;
      for (let k = 0; k < Q.dim; k++) dot += q.emb[k] * d[k];
      if (dot > best) {
        best = dot;
        bestJ = j;
      }
      if (dot > docMax[j]) docMax[j] = dot;
    }
    align.push({ q: q.str, d: dTok[bestJ]?.str ?? "", sim: best });
    total += best;
  }
  post({
    type: "result",
    id,
    docTokens: dTok.map((t, j) => ({ str: t.str, match: docMax[j] })),
    align,
    score: total,
    ms: Math.round(performance.now() - t0),
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "score") await score(e.data.id, e.data.query, e.data.document);
  } catch (err) {
    console.error("[colbert worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
