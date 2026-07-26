// Manga OCR (Japanese text recognition) worker — all inference off the main thread so the control UI
// stays responsive. REAL local inference, verified against the author's own test suite:
//   kha-white/manga_ocr tests/data/expected_results.json — 6/12 byte-exact, 3 half/full-width
//   punctuation diffs, 3 genuine character errors on adversarial synthetic bubbles (uint8 encoder +
//   q8 decoder, greedy). The q8 decoder output is BYTE-IDENTICAL to the fp32 decoder on all 12.
//
// Model: onnx-community/manga-ocr-base-ONNX (Apache-2.0 export of kha-white/manga-ocr-base) — a
// VisionEncoderDecoder: DeiT-base ViT encoder reads a 224×224 crop, a 2-layer Japanese char-BERT
// decoder writes the text one character per token. Runs via onnxruntime-web directly (raw-ORT
// pattern, like the built SPLADE/ColBERT demos) for two reasons, both verified by inspection:
//   1. The export ships NO decoder_model_merged_*.onnx, which transformers.js 3.7.5 hard-requires
//      for Vision2Seq, and
//   2. transformers.js's generate() derails this model's first token (it suppresses the natural
//      leading [CLS] the model emits — Python keeps it — and the beam never recovers).
// So we run the encoder + decoder sessions ourselves with a GREEDY cacheless decode loop (the
// decoder graph has no KV-cache inputs — each step re-decodes the growing sequence; fine for a
// 2-layer decoder and text-line lengths). Special-token ids (start [CLS]=2, eos [SEP]=3, pad=0)
// come from the repo's generation_config.json. The detokenizer is decode-only char lookup from the
// base repo's vocab.txt (input is pixels, so the MeCab word-tokeniser is never needed) — faithful:
// it reproduces the author's expected outputs exactly where the model is correct.
//
// Downloads go through Cache Storage keyed by the real resolve URLs so lib/model-cache.js can
// auto-init on revisit; progress events use the shared per-file vocabulary (real bytes).

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/ort.all.min.mjs";
const ORT_WASM_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/";
const REPO = "onnx-community/manga-ocr-base-ONNX";
const ENCODER_URL = `https://huggingface.co/${REPO}/resolve/main/onnx/encoder_model_uint8.onnx`; // 87.0 MB
const DECODER_URL = `https://huggingface.co/${REPO}/resolve/main/onnx/decoder_model_quantized.onnx`; // 29.6 MB
const VOCAB_URL = "https://huggingface.co/kha-white/manga-ocr-base/resolve/main/vocab.txt"; // ~50 KB
const CACHE_NAME = "manga-ocr-onnx-cache";

// From generation_config.json (repo): decoder_start_token_id=2 ([CLS]), eos_token_id=3 ([SEP]),
// pad_token_id=0 ([PAD]), max_length=300.
const START_ID = 2;
const EOS_ID = 3;
const HARD_MAX_TOKENS = 300;

let ort = null;
let tjs = null;
let processor = null;
let vocab = null;
let encSession = null;
let decSession = null;
let cancelled = false;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

const SKIP = new Set(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]);
function decodeIds(ids) {
  let out = "";
  for (const id of ids) {
    const tok = vocab[id];
    if (tok == null || SKIP.has(tok) || tok.startsWith("<unused")) continue;
    out += tok.startsWith("##") ? tok.slice(2) : tok;
  }
  return out;
}

// The canonical post-processing from the reference implementation (manga_ocr.ocr.post_process):
// strip whitespace, normalise ellipsis runs, halfwidth ASCII/digits → fullwidth (jaconv.h2z).
function postProcess(text) {
  let t = text.replace(/\s+/g, "");
  t = t.replace(/…/g, "...");
  t = t.replace(/[・.]{2,}/g, (m) => ".".repeat(m.length));
  let out = "";
  for (const ch of t) {
    const c = ch.codePointAt(0);
    // h2z(ascii=True, digit=True): ASCII printables map to their fullwidth forms.
    out += c >= 0x21 && c <= 0x7e ? String.fromCodePoint(c + 0xfee0) : ch;
  }
  return out;
}

function softmaxTop(row, k) {
  let mx = -Infinity;
  for (let i = 0; i < row.length; i++) if (row[i] > mx) mx = row[i];
  let sum = 0;
  for (let i = 0; i < row.length; i++) sum += Math.exp(row[i] - mx);
  const top = [];
  for (let i = 0; i < row.length; i++) {
    const p = Math.exp(row[i] - mx) / sum;
    if (top.length < k || p > top[top.length - 1].p) {
      top.push({ id: i, p });
      top.sort((a, b) => b.p - a.p);
      if (top.length > k) top.pop();
    }
  }
  return top;
}

// Fetch a model file through Cache Storage (keyed by its real resolve URL → model-cache auto-init
// sees it), emitting honest per-file byte progress in the shared vocabulary.
async function fetchCached(url, file, cache) {
  const hit = await cache.match(url);
  if (hit) {
    post({ type: "progress", p: { status: "done", file, name: file } });
    return new Uint8Array(await hit.arrayBuffer());
  }
  post({ type: "progress", p: { status: "initiate", file, name: file } });
  const net = await fetch(url);
  if (!net.ok || !net.body) throw new Error(`fetch failed (${net.status}) for ${file}`);
  const total = Number(net.headers.get("content-length")) || 0;
  post({ type: "progress", p: { status: "download", file, name: file } });
  const reader = net.body.getReader();
  const chunks = [];
  let received = 0;
  let lastPost = 0;
  // Stall watchdog: a CDN stream that goes silent for 45s is dead — fail honestly so the loader
  // can offer Retry instead of spinning "Downloading…" forever.
  const STALL_MS = 45000;
  for (;;) {
    let stall;
    const chunk = await Promise.race([
      reader.read(),
      new Promise((_, reject) => { stall = setTimeout(() => reject(new Error(`Download stalled for ${STALL_MS / 1000}s — connection to the model host went quiet. Retry resumes with a fresh request.`)), STALL_MS); }),
    ]).finally(() => clearTimeout(stall));
    const { done, value } = chunk;
    if (done) break;
    chunks.push(value);
    received += value.length;
    // Throttle: one progress event per ~128 KB so the main thread is never flooded with posts.
    if (received - lastPost >= 128 * 1024) {
      lastPost = received;
      post({ type: "progress", p: { status: "progress", file, name: file, loaded: received, total } });
    }
  }
  post({ type: "progress", p: { status: "progress", file, name: file, loaded: received, total } });
  const buf = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) {
    buf.set(c, off);
    off += c.length;
  }
  try {
    await cache.put(url, new Response(buf, { headers: { "content-length": String(received) } }));
  } catch (err) {
    // Cache Storage can refuse large entries (starved quota, private mode, constrained embeds).
    // The model still runs from memory this session — honestly re-offered as Download next visit.
    console.warn(`[manga-ocr] could not persist ${file} to Cache Storage (${err?.message ?? err}); running from memory`);
  }
  post({ type: "progress", p: { status: "done", file, name: file } });
  return buf;
}

async function ensureLoaded() {
  if (encSession && decSession) return;
  tjs = await import(TRANSFORMERS_URL);
  tjs.env.allowLocalModels = false; // the library owns its Cache Storage; don't fight the SW
  ort = await import(ORT_URL);
  ort.env.wasm.wasmPaths = ORT_WASM_BASE;
  ort.env.wasm.numThreads = 1; // GitHub Pages is not cross-origin-isolated — honest single-thread WASM

  const cache = await caches.open(CACHE_NAME);
  // The detokeniser table (decode-only; ~50 KB) + the image preprocessor config (transformers.js
  // caches it under the same repo id), then the two ONNX graphs.
  const vocabRes = await cache.match(VOCAB_URL).then((h) => h ?? fetch(VOCAB_URL));
  if (vocabRes.ok) await cache.put(VOCAB_URL, vocabRes.clone());
  vocab = (await vocabRes.text()).split("\n");
  processor = await tjs.AutoImageProcessor.from_pretrained(REPO, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  const [encBytes, decBytes] = [
    await fetchCached(ENCODER_URL, "onnx/encoder_model_uint8.onnx", cache),
    await fetchCached(DECODER_URL, "onnx/decoder_model_quantized.onnx", cache),
  ];
  encSession = await ort.InferenceSession.create(encBytes, { executionProviders: ["wasm"] });
  decSession = await ort.InferenceSession.create(decBytes, { executionProviders: ["wasm"] });
  post({ type: "ready", device: "wasm" });
}

async function run(id, imageURL, maxTokens) {
  await ensureLoaded();
  cancelled = false;
  const t0 = performance.now();

  // Preprocess exactly like the reference pipeline: grayscale → RGB, ViTImageProcessor
  // (224×224 resize, rescale 1/255, normalise mean/std 0.5).
  let image = await tjs.RawImage.fromURL(imageURL);
  try {
    image = image.grayscale().rgb();
  } catch { /* older transformers.js without grayscale() — ViT still sees 3 channels */ }
  const inputs = await processor(image);
  const pixel = new ort.Tensor("float32", inputs.pixel_values.data, inputs.pixel_values.dims);

  const tEnc = performance.now();
  const encOut = await encSession.run({ pixel_values: pixel });
  const hidden = encOut.last_hidden_state;
  const encMs = Math.round(performance.now() - tEnc);

  // Greedy cacheless decode: start from [CLS], append the argmax char each step until [SEP].
  const maxNew = Math.max(1, Math.min(HARD_MAX_TOKENS, maxTokens || 128));
  const ids = [START_ID];
  const tDec = performance.now();
  let steps = 0;
  for (; steps < maxNew; steps++) {
    if (cancelled) {
      post({ type: "cancelled", id, text: postProcess(decodeIds(ids.slice(1))), raw: decodeIds(ids.slice(1)), ms: Math.round(performance.now() - t0) });
      return;
    }
    const inputIds = new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
    const out = await decSession.run({ input_ids: inputIds, encoder_hidden_states: hidden });
    const V = out.logits.dims[2];
    const row = out.logits.data.slice((ids.length - 1) * V, ids.length * V);
    const top = softmaxTop(row, 3); // real per-step probabilities for the "see inside" surface
    const next = top[0].id;
    const t = Math.round(performance.now() - t0);
    post({
      type: "token",
      id,
      i: steps,
      t,
      token: next === EOS_ID ? "" : decodeIds([next]),
      prob: top[0].p,
      alternatives: top.slice(1).map((c) => ({ token: decodeIds([c.id]), p: c.p })),
    });
    if (next === EOS_ID) break;
    ids.push(next);
  }
  const raw = decodeIds(ids.slice(1));
  const text = postProcess(raw);
  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    text,
    raw,
    tokens: steps,
    ms,
    encMs,
    decMs: Math.round(performance.now() - tDec),
    device: "wasm",
    truncated: steps >= maxNew,
  });
}

self.addEventListener("message", async (e) => {
  const d = e.data;
  try {
    if (d.type === "load") await ensureLoaded();
    else if (d.type === "run") await run(d.id, d.image, d.maxTokens);
    else if (d.type === "cancel") cancelled = true;
  } catch (err) {
    post({ type: "error", id: d?.id, message: String(err?.message ?? err) });
  }
});
