// Canonical Turkish BERT NER, run directly with ORT-Web because this repository ships a root
// model.onnx rather than the onnx/model_*.onnx layout expected by Transformers.js pipelines.
// Exact lineage: akdeniz27/bert-base-turkish-cased-ner @
// 99995f7d2be4b3a28c74f0d36ee97f8c04ee0571, fp32 model.onnx (440,394,743 bytes),
// SHA-256 a8f8a685d1a3dbf4a22a0c3ec9810f12a7035062fd61d79cadb759c24ace4482, MIT.
// Tokenisation, ONNX inference, WordPiece reconstruction and BIO span aggregation all stay here,
// off the main thread. The visitor's text never leaves the browser.

import {
  alignPieces,
  assertArtifactIntegrity,
  entitySpans,
  mergeWords,
} from "./ner-core.js";

const MODEL_ID = "akdeniz27/bert-base-turkish-cased-ner";
const REVISION = "99995f7d2be4b3a28c74f0d36ee97f8c04ee0571";
const MODEL_URL = `https://huggingface.co/${MODEL_ID}/resolve/${REVISION}/model.onnx`;
const MODEL_BYTES = 440394743;
const MODEL_SHA256 = "a8f8a685d1a3dbf4a22a0c3ec9810f12a7035062fd61d79cadb759c24ace4482";
const LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"];

let tokenizer = null;
let session = null;
let ort = null;

const post = (message) => self.postMessage(message);

async function readWithProgress(response) {
  const reader = response.body?.getReader();
  if (!reader) return response.arrayBuffer();
  const chunks = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    post({
      type: "progress",
      p: {
        status: "progress",
        file: "model.onnx",
        loaded,
        total: MODEL_BYTES,
        progress: loaded / MODEL_BYTES * 100,
      },
    });
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes.buffer;
}

async function verifiedModelBytes(response, source) {
  const bytes = await response.arrayBuffer();
  await assertArtifactIntegrity(bytes, {
    expectedBytes: MODEL_BYTES,
    expectedSha256: MODEL_SHA256,
    label: `${source} model.onnx`,
  });
  return bytes;
}

async function modelBytes() {
  const cache = await caches.open("transformers-cache");
  let response = await cache.match(MODEL_URL);
  if (response) {
    post({
      type: "progress",
      p: { status: "initiate", file: "model.onnx", name: "Validating cached model" },
    });
    try {
      return await verifiedModelBytes(response, "Cached");
    } catch (error) {
      // Fail closed and remove poisoned/stale bytes. The shared loader exposes Retry; this call must
      // not silently turn a returning user's auto-init into another 420 MiB network transfer.
      await cache.delete(MODEL_URL);
      throw error;
    }
  }
  post({
    type: "progress",
    p: { status: "initiate", file: "model.onnx", name: "Canonical fp32 model" },
  });
  response = await fetch(MODEL_URL, { cache: "no-store" });
  if (!response.ok) throw new Error(`Model download failed (${response.status})`);
  const headers = new Headers(response.headers);
  const bytes = await readWithProgress(new Response(response.body, { status: 200, headers }));
  await assertArtifactIntegrity(bytes, {
    expectedBytes: MODEL_BYTES,
    expectedSha256: MODEL_SHA256,
    label: "Downloaded model.onnx",
  });
  // Only verified bytes cross the persistent-cache boundary.
  await cache.put(MODEL_URL, new Response(bytes, { status: 200, headers }));
  return bytes;
}

async function ensureLoaded() {
  if (session && tokenizer) return;
  const [{ AutoTokenizer }, ortModule] = await Promise.all([
    import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5"),
    import("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.min.mjs"),
  ]);
  ort = ortModule;
  ort.env.wasm.numThreads = 1;
  tokenizer = await AutoTokenizer.from_pretrained("akdeniz27/bert-base-turkish-cased-ner", {
    revision: REVISION,
  });
  const bytes = await modelBytes();
  session = await ort.InferenceSession.create(bytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  post({
    type: "ready",
    device: "wasm",
    revision: REVISION,
    modelBytes: MODEL_BYTES,
    modelSha256: MODEL_SHA256,
  });
}

function softmaxBest(data, offset, width) {
  let max = -Infinity;
  for (let i = 0; i < width; i++) max = Math.max(max, data[offset + i]);
  let sum = 0;
  const exps = new Float64Array(width);
  for (let i = 0; i < width; i++) {
    exps[i] = Math.exp(data[offset + i] - max);
    sum += exps[i];
  }
  let best = 0;
  for (let i = 1; i < width; i++) if (exps[i] > exps[best]) best = i;
  return { label: LABELS[best], score: exps[best] / sum };
}

// Transformers.js currently returns zero offset_mapping values for this slow-compatible BERT
// tokenizer. Pure helpers align exact cased WordPieces and repair/aggregate BIO spans.

async function analyse(text) {
  const encoded = await tokenizer(text, { truncation: true, max_length: 256 });
  const feeds = {};
  for (const name of session.inputNames) {
    const tensor = encoded[name];
    if (!tensor) throw new Error(`Tokenizer did not provide required ONNX input ${name}`);
    feeds[name] = new ort.Tensor("int64", BigInt64Array.from(tensor.data, BigInt), tensor.dims);
  }
  const output = await session.run(feeds);
  const logits = output[session.outputNames[0]];
  const count = logits.dims[1];
  const width = logits.dims[2];
  const ids = Array.from(encoded.input_ids.data, Number);
  const pieces = ids.map((id) =>
    tokenizer.decode([id], { skip_special_tokens: false, clean_up_tokenization_spaces: false })
  );
  const aligned = alignPieces(pieces, text);
  const tokens = pieces.map((word, index) => {
    const { label: entity, score } = softmaxBest(logits.data, index * width, width);
    return { index, id: ids[index], word, entity, score, ...aligned[index] };
  });
  const words = mergeWords(tokens, text);
  return { text, tokens, words, entities: entitySpans(words, text) };
}

const cancelled = new Set();
let activeJob = null;
let queuedJob = null; // bounded latest-wins queue: one running + at most one waiting

async function execute(job) {
  const { id, type } = job;
  await ensureLoaded();
  if (cancelled.delete(id)) return;
  const started = performance.now();
  if (type === "tag") {
    const result = await analyse(job.text);
    if (!cancelled.delete(id)) {
      post({
        type: "tag",
        id,
        ...result,
        ms: Math.round(performance.now() - started),
        device: "wasm",
      });
    }
    return;
  }
  const results = [];
  for (const text of job.texts) {
    if (cancelled.has(id)) break;
    results.push(await analyse(text));
  }
  if (!cancelled.delete(id)) {
    post({
      type: "tagMany",
      id,
      results,
      ms: Math.round(performance.now() - started),
      device: "wasm",
    });
  }
}

async function drain() {
  if (activeJob || !queuedJob) return;
  activeJob = queuedJob;
  queuedJob = null;
  try {
    await execute(activeJob);
  } catch (error) {
    if (!cancelled.delete(activeJob.id)) {
      post({ type: "error", id: activeJob.id, message: String(error?.message || error) });
    }
  } finally {
    activeJob = null;
    void drain();
  }
}

function enqueue(job) {
  if (queuedJob) cancelled.delete(queuedJob.id);
  queuedJob = job;
  void drain();
}

self.addEventListener("message", async ({ data }) => {
  try {
    if (data.type === "load") await ensureLoaded();
    else if (data.type === "cancel") cancelled.add(data.id);
    else if (data.type === "tag" || data.type === "tagMany") enqueue(data);
  } catch (error) {
    post({ type: "error", id: data?.id, message: String(error?.message || error) });
  }
});
