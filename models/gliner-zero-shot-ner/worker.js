// Zero-shot NER worker — extracts entities of ANY types you name, entirely on-device.
//
// Model: GLiNER small v2.1 (onnx-community/gliner_small-v2.1) — a single microsoft/deberta-v3-small encoder
// that scores candidate word-spans against entity-type prompts. The weights are Apache-2.0 (base
// urchade/gliner_small-v2.1); Apache-2.0 permits redistribution, so they stay Apache-2.0 in the
// onnx-community conversion despite its blank license field. transformers.js has no GLiNER pipeline, so we
// run the ONNX graph directly via onnxruntime-web (a per-worker pin, like the other raw-ORT demos) and
// tokenize with a transformers.js AutoTokenizer.
//
// The preprocessing + span decoding below faithfully reimplement the MIT-licensed `gliner` JS library
// (github: urchade/GLiNER, npm `gliner`) — prompt is "<<ENT>> type1 <<ENT>> type2 ... <<SEP>> words",
// words_mask marks the first sub-token of each real word, span_idx enumerates all spans up to max_width,
// and logits[startWord, width, entityType] -> sigmoid -> greedy non-overlapping selection. It was proven
// correct FIRST in headless Chrome against known sentences: "Barack Obama was born in Honolulu Hawaii"
// [person, location] -> person: Barack Obama (0.98), location: Honolulu (0.87), location: Hawaii (0.76);
// "Apple was founded by Steve Jobs in California in 1976" -> company/person/location/date all correct.
// Nothing leaves the tab.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const REPO = "onnx-community/gliner_small-v2.1";
const MODEL_URL = `https://huggingface.co/${REPO}/resolve/main/onnx/model_quantized.onnx`;
const CACHE_NAME = "gliner-onnx-cache";
const MAX_WIDTH = 12; // from gliner_config.json
// Split into word tokens with char offsets (GLiNER WhitespaceTokenSplitter).
const WORD_RE = /\w+(?:[-_]\w+)*|\S/g;

let ort = null;
let session = null;
let tokenizer = null;

function post(msg) {
  self.postMessage(msg);
}
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const hasOverlap = (a, b) => {
  if (a[0] === b[0] && a[1] === b[1]) return true;
  if (a[0] > b[1] || b[0] > a[1]) return false;
  return true;
};

// Fetch the model THROUGH Cache Storage under a key carrying the model-id path so lib/model-cache.js
// auto-inits on a returning visit (the tokenizer files are cached by transformers.js under the same repo).
async function fetchCached(url, cache, onChunk) {
  const key = `https://huggingface.co/${REPO}/resolve/main/model_quantized.onnx`;
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
  const cache = await caches.open(CACHE_NAME);
  const bytes = await fetchCached(MODEL_URL, cache, (r, t) => {
    if (t) post({ type: "progress", p: { status: "progress", progress: (r / t) * 98 } });
  });
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  post({ type: "ready", device: "wasm" });
}

const bi = (a) => BigInt64Array.from(a, BigInt);

// The verified GLiNER span pipeline: text + entity types -> [{ text, start, end, type, score }].
async function extract(text, entities, threshold) {
  await ensureLoaded();
  const t0 = performance.now();
  // 1) whitespace tokenize with char offsets
  const words = [], ws = [], we = [];
  let m;
  WORD_RE.lastIndex = 0;
  while ((m = WORD_RE.exec(text)) !== null) {
    words.push(m[0]);
    ws.push(m.index);
    we.push(WORD_RE.lastIndex);
  }
  const textLength = words.length;
  if (!textLength || !entities.length) {
    post({ type: "result", spans: [], ms: 0 });
    return;
  }
  // 2) prompt: <<ENT>> type .. <<SEP>> words
  const prompt = [];
  for (const e of entities) prompt.push("<<ENT>>", e);
  prompt.push("<<SEP>>");
  const promptLength = prompt.length;
  const seq = prompt.concat(words);
  // 3) encode: hardcoded CLS(1) start, sep_token_id end; words_mask marks first sub-token of each real word
  const wordsMask = [0], inputIds = [1], attn = [1];
  let c = 1;
  seq.forEach((word, wordId) => {
    const sub = tokenizer.encode(word).slice(1, -1);
    sub.forEach((t, ti) => {
      attn.push(1);
      if (wordId < promptLength) wordsMask.push(0);
      else if (ti === 0) wordsMask.push(c++);
      else wordsMask.push(0);
      inputIds.push(t);
    });
  });
  wordsMask.push(0);
  inputIds.push(tokenizer.sep_token_id);
  attn.push(1);
  const L = inputIds.length;
  // 4) spans up to MAX_WIDTH
  const spanIdx = [], spanMask = [];
  for (let i = 0; i < textLength; i++) {
    for (let j = 0; j < MAX_WIDTH; j++) {
      const e = Math.min(i + j, textLength - 1);
      spanIdx.push(i, e);
      spanMask.push(i + j < textLength ? 1 : 0);
    }
  }
  const numSpans = textLength * MAX_WIDTH;
  const feeds = {
    input_ids: new ort.Tensor("int64", bi(inputIds), [1, L]),
    attention_mask: new ort.Tensor("int64", bi(attn), [1, L]),
    words_mask: new ort.Tensor("int64", bi(wordsMask), [1, L]),
    text_lengths: new ort.Tensor("int64", bi([textLength]), [1, 1]),
    span_idx: new ort.Tensor("int64", bi(spanIdx), [1, numSpans, 2]),
    span_mask: new ort.Tensor("bool", Uint8Array.from(spanMask), [1, numSpans]),
  };
  const out = await session.run(feeds);
  const logits = out.logits.data;
  // 5) decode: logits index = [startWord, width, entityType]; sigmoid -> candidates
  const numEntities = entities.length;
  const startPad = MAX_WIDTH * numEntities, entPad = numEntities;
  const cand = [];
  for (let id = 0; id < logits.length; id++) {
    const st = Math.floor(id / startPad) % textLength;
    const et = st + (Math.floor(id / entPad) % MAX_WIDTH);
    const en = id % numEntities;
    const p = sigmoid(logits[id]);
    if (p >= threshold && st < textLength && et < textLength) {
      cand.push({
        text: text.slice(ws[st], we[et]),
        start: ws[st],
        end: we[et],
        type: entities[en],
        score: p,
      });
    }
  }
  // 6) greedy flat NER: highest score first, drop overlaps
  cand.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const s of cand) {
    if (!keep.some((k) => hasOverlap([s.start, s.end], [k.start, k.end]))) keep.push(s);
  }
  keep.sort((a, b) => a.start - b.start);
  post({ type: "result", spans: keep, ms: Math.round(performance.now() - t0) });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "extract") {
      await extract(e.data.text, e.data.entities, e.data.threshold ?? 0.5);
    }
  } catch (err) {
    console.error("[gliner worker] error", err);
    post({ type: "error", message: String(err?.message ?? err) });
  }
});
