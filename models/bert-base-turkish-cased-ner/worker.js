// Canonical Turkish BERT NER, run directly with ORT-Web because this repository ships a root
// model.onnx rather than the onnx/model_*.onnx layout expected by Transformers.js pipelines.
// Exact lineage: akdeniz27/bert-base-turkish-cased-ner @
// 99995f7d2be4b3a28c74f0d36ee97f8c04ee0571, fp32 model.onnx (440,394,743 bytes), MIT.
// Tokenisation, ONNX inference, WordPiece reconstruction and BIO span aggregation all stay here,
// off the main thread. The visitor's text never leaves the browser.

const MODEL_ID = "akdeniz27/bert-base-turkish-cased-ner";
const REVISION = "99995f7d2be4b3a28c74f0d36ee97f8c04ee0571";
const MODEL_URL = `https://huggingface.co/${MODEL_ID}/resolve/${REVISION}/model.onnx`;
const MODEL_BYTES = 440394743;
const LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"];
const SPECIAL = /^\[(?:CLS|SEP|PAD|MASK)\]$/;

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

async function modelBytes() {
  const cache = await caches.open("transformers-cache");
  let response = await cache.match(MODEL_URL);
  if (response) {
    post({
      type: "progress",
      p: { status: "initiate", file: "model.onnx", name: "Validated cached model" },
    });
    return response.arrayBuffer();
  }
  post({
    type: "progress",
    p: { status: "initiate", file: "model.onnx", name: "Canonical fp32 model" },
  });
  response = await fetch(MODEL_URL, { cache: "no-store" });
  if (!response.ok) throw new Error(`Model download failed (${response.status})`);
  const [forRuntime, forCache] = response.body.tee();
  const headers = new Headers(response.headers);
  const cacheWrite = cache.put(MODEL_URL, new Response(forCache, { status: 200, headers }));
  const bytes = await readWithProgress(new Response(forRuntime, { status: 200, headers }));
  await cacheWrite;
  if (bytes.byteLength !== MODEL_BYTES) {
    throw new Error(
      `Model integrity check failed: expected ${MODEL_BYTES} bytes, received ${bytes.byteLength}`,
    );
  }
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
  post({ type: "ready", device: "wasm", revision: REVISION, modelBytes: MODEL_BYTES });
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
// tokenizer. Align the exact cased WordPieces monotonically against the original Turkish text.
function alignPieces(pieces, text) {
  let cursor = 0;
  return pieces.map((piece) => {
    if (SPECIAL.test(piece)) return { start: null, end: null, special: true };
    while (/\s/u.test(text[cursor] || "")) cursor++;
    let surface = piece.replace(/^##/, "");
    if (piece === "[UNK]") {
      const match = text.slice(cursor).match(/^[^\s.,!?;:'"()]+/u);
      surface = match?.[0] || text[cursor] || "";
    }
    const start = text.indexOf(surface, cursor);
    if (start < 0) {
      throw new Error(`Could not align tokenizer piece “${piece}” after character ${cursor}`);
    }
    const end = start + surface.length;
    cursor = end;
    return { start, end, special: false };
  });
}

function mergeWords(tokens, text) {
  const words = [];
  for (const token of tokens) {
    if (token.special) continue;
    const continuation = token.word.startsWith("##") && words.length > 0;
    if (continuation) {
      const word = words.at(-1);
      word.end = token.end;
      word.surface = text.slice(word.start, word.end);
      word.pieces.push(token);
      // WordPiece aggregation uses the first piece's BIO label, matching the token-classification
      // "first" strategy and preventing an occasional noisy continuation from splitting a word.
      word.score = word.pieces.reduce((sum, item) => sum + item.score, 0) / word.pieces.length;
    } else {
      const type = token.entity === "O" ? null : token.entity.slice(2);
      words.push({
        surface: text.slice(token.start, token.end),
        entity: token.entity,
        type,
        score: token.score,
        start: token.start,
        end: token.end,
        pieces: [token],
      });
    }
  }
  return words.map(({ pieces: _pieces, ...word }) => word);
}

function entitySpans(words, text) {
  const spans = [];
  let open = null;
  for (const word of words) {
    if (!word.type) {
      open = null;
      continue;
    }
    const bio = word.entity.slice(0, 1);
    if (bio === "B" || !open || open.type !== word.type) {
      open = { type: word.type, start: word.start, end: word.end, scores: [word.score] };
      spans.push(open);
    } else {
      open.end = word.end;
      open.scores.push(word.score);
    }
  }
  return spans.map((span) => ({
    type: span.type,
    text: text.slice(span.start, span.end),
    start: span.start,
    end: span.end,
    score: span.scores.reduce((a, b) => a + b, 0) / span.scores.length,
  }));
}

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

async function runOne(id, text) {
  await ensureLoaded();
  const started = performance.now();
  const result = await analyse(text);
  post({ type: "tag", id, ...result, ms: Math.round(performance.now() - started), device: "wasm" });
}

async function runMany(id, texts) {
  await ensureLoaded();
  const started = performance.now();
  const results = [];
  for (const text of texts) results.push(await analyse(text));
  post({
    type: "tagMany",
    id,
    results,
    ms: Math.round(performance.now() - started),
    device: "wasm",
  });
}

async function dispose() {
  await session?.release?.();
  session = null;
  tokenizer = null;
  post({ type: "disposed" });
}

self.addEventListener("message", async ({ data }) => {
  try {
    if (data.type === "load") await ensureLoaded();
    else if (data.type === "tag") await runOne(data.id, data.text);
    else if (data.type === "tagMany") await runMany(data.id, data.texts);
    else if (data.type === "dispose") await dispose();
  } catch (error) {
    post({ type: "error", id: data?.id, message: String(error?.message || error) });
  }
});
