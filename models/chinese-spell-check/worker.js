// macbert4csc worker — Chinese spelling correction off the main thread, WASM fp32.
// Model: shibing624/macbert4csc-base-chinese (BertForMaskedLM, onnx/ subfolder).
//
// Chinese spelling correction (CSC) fixes homophone/visual typos ("今天天汽很好" -> "今天天气很好").
// macbert4csc is a Chinese BERT fine-tuned for CSC. We run the MLM head directly: tokenize the whole
// sentence, do ONE forward pass, and for every position take the model's argmax token — that's the
// corrected character. No [MASK] trick, no seq2seq: the model predicts the most likely char at each
// position given the whole sentence, and we diff it against the input to surface changes. Every
// change + per-position confidence is the model's real output.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL = "shibing624/macbert4csc-base-chinese";
const SUBFOLDER = "onnx";
const DTYPE = "fp32"; // only fp32 export exists (474MB single-file, no quantized variant)

// NOTE: transformers.js 3.7.5 AutoTokenizer.from_pretrained does NOT accept `subfolder`
// (the option is silently dropped) and rejects non-repo path/URL forms, while model classes
// default to the onnx/ subfolder. So the tokenizer is constructed manually from the repo's
// onnx/ tokenizer.json + tokenizer_config.json (the exact `new cls(json, config)` call the
// library's AutoTokenizer performs internally), and the model uses the default onnx subfolder.
let tokenizer = null;
let model = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (model) return;
  const { AutoTokenizer, BertForMaskedLM } = await import(TRANSFORMERS_URL); // AutoTokenizer kept for parity; see note
  const { BertTokenizer } = await import(TRANSFORMERS_URL);
  const [tokenizerJSON, tokenizerConfig] = await Promise.all([
    (await fetch(`https://huggingface.co/${MODEL}/resolve/main/${SUBFOLDER}/tokenizer.json`)).json(),
    (await fetch(`https://huggingface.co/${MODEL}/resolve/main/${SUBFOLDER}/tokenizer_config.json`)).json(),
  ]);
  tokenizer = new BertTokenizer(tokenizerJSON, tokenizerConfig);
  model = await BertForMaskedLM.from_pretrained(MODEL, {
    dtype: DTYPE,
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  post({ type: "ready", device });
}

async function correct(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const enc = await tokenizer(text);
  const { input_ids, attention_mask } = enc;
  const seq = input_ids.dims[1];
  const out = await model({ input_ids, attention_mask });
  const logits = out.logits.data; // [1, seq, vocab]
  const vocab = out.logits.dims[2];

  const inputTokens = Array.from(input_ids.data);
  const changes = [];
  const correctedTokens = [];
  const confidences = [];
  for (let i = 0; i < seq; i++) {
    const base = i * vocab;
    let best = -Infinity, bi = 0;
    for (let v = 0; v < vocab; v++) {
      const val = logits[base + v];
      if (val > best) { best = val; bi = v; }
    }
    // confidence: softmax over the full vocab at this position
    let max = -Infinity;
    for (let v = 0; v < vocab; v++) if (logits[base + v] > max) max = logits[base + v];
    let sum = 0;
    for (let v = 0; v < vocab; v++) sum += Math.exp(logits[base + v] - max);
    confidences.push(Math.exp(best - max) / sum);
    correctedTokens.push(bi);
    const orig = tokenizer.decode([inputTokens[i]], { skip_special_tokens: true });
    const corr = tokenizer.decode([bi], { skip_special_tokens: true });
    if (orig && corr && orig !== corr && orig !== "[UNK]") {
      changes.push({ pos: i, from: orig, to: corr });
    }
  }
  // Chinese text has no word spaces — collapse the tokenizer's per-token spacing.
  const corrected = tokenizer.decode(correctedTokens, { skip_special_tokens: true }).replace(/\s+/g, "");
  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    corrected,
    changes,
    confidences,
    seq,
    ms,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type, id } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "correct") await correct(id, e.data.text);
  } catch (err) {
    post({ type: "error", id, message: String(err?.message ?? err) });
  }
});
