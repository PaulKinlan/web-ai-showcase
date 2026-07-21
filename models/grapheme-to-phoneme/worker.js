// Grapheme-to-phoneme (G2P) worker — inference off the main thread so the input stays responsive.
//
// Model: CharsiuG2P multilingual byT5 (Santyyy/g2p_multilingual_byT5_tiny_16_layers_100-onnx, a faithful
// ONNX conversion of charsiu/g2p_multilingual_byT5). The weights are MIT — CharsiuG2P is published under
// MIT (github.com/lingjzhu/CharsiuG2P); MIT permits redistribution, so the weights stay MIT wherever
// converted. Task: text2text-generation. It maps SPELLING -> SOUND (IPA phonemes) for one word at a time,
// in a chosen language, from a query of the form "<lang-code>: word".
//
// WHY A DEDICATED CLASS INSTEAD OF pipeline():
// byT5 is token-free — no learned vocabulary, so the repo ships no tokenizer.json and transformers.js has
// no ByT5Tokenizer. But the tokenizer is a fixed byte map (UTF-8 byte + 3; see g2p.js), so we run the REAL
// ONNX model via T5ForConditionalGeneration and feed the byte ids ourselves — the same approach as the
// built byt5-byte-level demo. This is why ANY script (Latin, Cyrillic, kana, Greek) can be phonemized.
//
// Correctness proven FIRST in headless Chrome (transformers.js 3.7.5, WASM, fp32): "<eng-us>: hello" ->
// ˈhɛɫoʊ, "<fra>: merci" -> mɛʁsi, "<ger>: danke" -> ˈdaŋke, "<rus>: спасибо" -> spɐsʲibə,
// "<jpn>: ありがとう" -> aɾigatoɯ, "<gre>: ευχαριστώ" -> efxaɾisto (10 languages, incl. non-Latin scripts).

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";
import { idsToText, textToIds } from "./g2p.js";

const MODEL_ID = "Santyyy/g2p_multilingual_byT5_tiny_16_layers_100-onnx";
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
  model = await T5ForConditionalGeneration.from_pretrained(MODEL_ID, {
    device: "wasm",
    dtype: "fp32",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  device = "wasm";
  post({ type: "ready", device });
}

// Phonemize one word in `code` via a "<code>: word" query.
async function phonemizeWord(code, word) {
  const ids = textToIds(`<${code}>: ${word}`);
  const n = ids.length;
  const input_ids = new Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, n]);
  const attention_mask = new Tensor("int64", BigInt64Array.from(ids.map(() => 1n)), [1, n]);
  const output = await model.generate({
    input_ids,
    attention_mask,
    max_new_tokens: 64,
    num_beams: 3, // small model; a few beams trims the odd trailing-byte artifact
    do_sample: false,
  });
  const outIds = Array.from(output.data ?? output[0].data, Number);
  return idsToText(outIds);
}

async function run(id, code, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const words = String(text).trim().split(/\s+/).filter(Boolean).slice(0, 12);
  const out = [];
  for (const w of words) out.push({ word: w, ipa: await phonemizeWord(code, w) });
  post({
    type: "result",
    id,
    words: out,
    ipa: out.map((w) => w.ipa).join(" "),
    ms: Math.round(performance.now() - t0),
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await run(e.data.id, e.data.code, e.data.text);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
