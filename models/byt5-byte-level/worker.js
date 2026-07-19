// ByT5 (byte-level seq2seq) worker — inference off the main thread so the control UI stays smooth.
// Model: onnx-community/byt5-small-ONNX (google/byt5-small), task text2text-generation, WASM, q8.
//
// WHY A DEDICATED CLASS INSTEAD OF pipeline():
// ByT5 is token-free — it has NO learned vocabulary, so the repo ships no tokenizer.json and
// Transformers.js implements no ByT5Tokenizer (only the SentencePiece T5Tokenizer). The pipeline()
// path therefore can't build a tokenizer. But ByT5's tokenizer is a fixed, documented byte map
// (UTF-8 byte + 3; see byt5.js), so we run the REAL ONNX model directly via
// T5ForConditionalGeneration and feed the byte ids ourselves. This is genuine on-device inference —
// and the byte mapping is exactly the mechanism the demo exists to show.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";
import { EOS, firstFill, idsToText, partsToIds, textToParts } from "./byt5.js";

const MODEL_ID = "onnx-community/byt5-small-ONNX";
let model = null;
let Tensor = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (model) return;
  const mod = await import(TRANSFORMERS_URL);
  const { T5ForConditionalGeneration, Tensor: T, env } = mod;
  Tensor = T;
  env.allowLocalModels = false;
  console.log(`[byt5 worker] loading ${MODEL_ID} on wasm (q8) — byte-level, no tokenizer`);
  model = await T5ForConditionalGeneration.from_pretrained(MODEL_ID, {
    device: "wasm",
    dtype: "q8",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  device = "wasm";
  console.log("[byt5 worker] ready on wasm");
  post({ type: "ready", device });
}

async function run(id, text, opts) {
  await ensureLoaded();
  const parts = textToParts(text);
  const ids = partsToIds(parts); // byte ids + <extra_id_0> for any [BLANK], + eos
  const n = ids.length;
  const input_ids = new Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, n]);
  const attention_mask = new Tensor("int64", BigInt64Array.from(ids.map(() => 1n)), [1, n]);

  const beams = Math.max(1, opts.beams | 0 || 1);
  const gen = {
    max_new_tokens: Math.max(4, opts.maxTokens | 0 || 32),
    num_beams: beams,
    do_sample: false,
    no_repeat_ngram_size: opts.noRepeat ?? 2,
  };

  const t0 = performance.now();
  const output = await model.generate({ input_ids, attention_mask, ...gen });
  const ms = Math.round(performance.now() - t0);

  const outIds = Array.from(output.data ?? output[0].data, Number);
  const hasBlank = parts.some((p) => p.s != null);
  // For a span-fill (denoise) the useful answer is the first fill; otherwise decode the whole output.
  const outputText = hasBlank ? firstFill(outIds) : idsToText(outIds);
  const rawText = idsToText(outIds, { markSentinels: true });

  post({
    type: "result",
    id,
    inputIds: ids,
    inputLen: n,
    outputIds: outIds,
    outputText,
    rawText,
    hasBlank,
    ms,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await run(e.data.id, e.data.text, e.data.opts || {});
  } catch (err) {
    console.error("[byt5 worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
