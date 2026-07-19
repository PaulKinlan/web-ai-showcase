// Text simplification worker — inference off the main thread so the control UI stays smooth.
// Model: grazh/agentlans-flan-t5-small-simplifier-onnx-v2 (task: text2text-generation), WASM, q8.
//
// This is a FLAN-T5-small ENCODER-DECODER (base: agentlans/flan-t5-small-simplifier) fine-tuned on one
// job: rewrite complex, jargon-heavy or dense English into plainer language while preserving meaning.
// Unlike a summariser it does not drop content, and unlike a paraphraser it is trained to REDUCE
// complexity (shorter words, simpler syntax). There is no chat template or task prefix: the entire
// input string IS the sentence to simplify. The encoder reads it; the decoder writes the simplified
// sentence one token at a time. We stream those tokens with a TextStreamer and timestamp each one so
// "See inside" shows the real decode cadence, plus the exact input string and its tokens. Greedy
// decoding keeps the rewrite stable and faithful.

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
  pipe = await pipeline(
    "text2text-generation",
    "grazh/agentlans-flan-t5-small-simplifier-onnx-v2",
    {
      device: "wasm",
      dtype: "q8",
      progress_callback: (p) => post({ type: "progress", p }),
    },
  );
  tokenizer = pipe.tokenizer;
  device = "wasm";
  post({ type: "ready", device });
}

/** Encode a string → { count, tokens:string[] }. Token strings expose T5's SentencePiece ▁ word marks. */
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

async function simplify(id, input, opts) {
  await ensureLoaded();
  // Greedy by default (faithful rewrites). maxNewTokens scales with input length so long inputs aren't
  // truncated; capped to keep latency reasonable.
  const cap = Math.min(256, Math.max(48, Math.round(input.split(/\s+/).length * 2) + 16));
  const gen = {
    max_new_tokens: opts.maxNewTokens ? Math.min(256, opts.maxNewTokens) : cap,
    num_beams: 1,
    do_sample: false,
    no_repeat_ngram_size: 3,
  };

  const times = [];
  let partial = "";
  gen.streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (t) => {
      partial += t;
      post({ type: "stream", id, text: partial });
    },
    token_callback_function: () => times.push(performance.now()),
  });

  const inTok = tokenize(input);
  const t0 = performance.now();
  const out = await pipe(input, gen);
  const ms = Math.round(performance.now() - t0);
  const output = (out[0]?.generated_text ?? "").trim();
  const intervals = times.map((t, i) => (i ? Math.round(t - times[i - 1]) : 0)).slice(1);
  const outTokens = times.length || tokenize(output).count;

  post({
    type: "result",
    id,
    input,
    output,
    inTokens: inTok.count,
    inTokenStrings: inTok.tokens,
    outTokens,
    intervals,
    ms,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await simplify(e.data.id, e.data.input, e.data.opts || {});
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
