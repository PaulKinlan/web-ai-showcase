// Paraphrase-generation worker — inference off the main thread so the control UI stays smooth.
// Model: Felladrin/onnx-chatgpt_paraphraser_on_T5_base (task: text2text-generation), WASM, q8.
//   An ONNX / transformers.js export of humarin/chatgpt_paraphraser_on_T5_base — a T5-base
//   (~223M params) fine-tuned specifically to REWRITE a sentence while preserving its meaning.
//
// This is a true seq2seq paraphraser, NOT a summarizer and NOT a grammar fixer: the encoder reads the
// whole input sentence, the decoder writes a re-worded version one token at a time. There is no task
// prefix — the input sentence IS the prompt. To get N DIVERSE paraphrases we run the model with
// SAMPLING (do_sample) N times: each run draws a different path through the distribution, so the wording
// varies while the meaning holds. (transformers.js 3.7.5 returns a single sequence per call even with
// num_return_sequences, so N calls is the honest way to get N variants.) Nothing leaves the device.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let tokenizer = null;
let TextStreamer = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const { pipeline, TextStreamer: TS, env } = await import(TRANSFORMERS_URL);
  TextStreamer = TS;
  env.allowLocalModels = false;
  pipe = await pipeline("text2text-generation", "Felladrin/onnx-chatgpt_paraphraser_on_T5_base", {
    device: "wasm",
    dtype: "q8",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  tokenizer = pipe.tokenizer;
  device = "wasm";
  post({ type: "ready", device });
}

/** Encode a string → { count, tokens:string[] }. T5 SentencePiece ▁ marks a word boundary. */
function tokenize(text) {
  const enc = tokenizer(text);
  const ids = enc.input_ids;
  const arr = ids?.tolist ? ids.tolist()[0] : (ids?.data ? Array.from(ids.data, Number) : []);
  let tokens = [];
  try {
    tokens = arr.map((id) => tokenizer.decode([id], { skip_special_tokens: false }));
  } catch {
    tokens = arr.map((id) => String(id));
  }
  return { count: arr.length, tokens };
}

async function generate(id, input, opts) {
  await ensureLoaded();
  const maxNew = Math.max(8, opts.maxNewTokens | 0 || 64);
  // Two decode modes:
  //   sampling   — do_sample with temperature/top-k/top-p; stream tokens + per-token timing.
  //   diverse    — humarin's recommended group beam search (num_beam_groups + diversity_penalty);
  //                deterministic per group, no clean per-token stream, so we report aggregate timing.
  const diverse = !!opts.diverse;
  const gen = {
    max_new_tokens: maxNew,
    no_repeat_ngram_size: 2,
  };
  const times = [];
  let partial = "";
  if (diverse) {
    const beams = Math.max(2, opts.numBeams | 0 || 4);
    gen.num_beams = beams;
    gen.num_beam_groups = beams;
    gen.diversity_penalty = opts.diversityPenalty ?? 3.0;
    gen.do_sample = false;
  } else {
    gen.do_sample = true;
    gen.temperature = opts.temperature ?? 0.9;
    gen.top_k = opts.topK ?? 60;
    gen.top_p = opts.topP ?? 0.95;
    gen.streamer = new TextStreamer(tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (t) => {
        partial += t;
        post({ type: "stream", id, text: partial });
      },
      token_callback_function: () => times.push(performance.now()),
    });
  }

  const inTok = tokenize(input);
  const t0 = performance.now();
  const out = await pipe(input, gen);
  const ms = Math.round(performance.now() - t0);
  const output = (out[0]?.generated_text ?? "").trim();
  const intervals = times.map((t, i) => (i ? Math.round(t - times[i - 1]) : 0)).slice(1);
  const outTok = tokenize(output);

  post({
    type: "result",
    id,
    input,
    output,
    inTokens: inTok.count,
    inTokenStrings: inTok.tokens,
    outTokens: outTok.count,
    outTokenStrings: outTok.tokens,
    intervals,
    mode: diverse ? "diverse" : "sampling",
    ms,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await generate(e.data.id, e.data.input, e.data.opts || {});
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
