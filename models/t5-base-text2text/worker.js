// T5-base (task-prefix seq2seq) worker — inference off the main thread so the control UI stays smooth.
// Model: Xenova/t5-base (task: text2text-generation), WASM, q8.
//
// This is the ORIGINAL T5 (Raffel et al. 2019), NOT the instruction-tuned FLAN-T5. Every task is one
// text-to-text problem, and the model only knows which task you mean from an explicit TASK PREFIX at
// the very front of the input string — the exact prefixes it was pre-trained on:
//   "translate English to German: Hello."     -> "Hallo."
//   "summarize: <long text>"                  -> a shorter text
//   "cola sentence: The books is on the desk." -> "acceptable" | "unacceptable"   (grammaticality)
//   "stsb sentence1: A. sentence2: B."         -> a number 0.0–5.0                (semantic similarity)
// Change the prefix and the SAME weights do a completely different job. That multi-task-via-prefix
// mechanism is the whole story of the demo, so the worker returns the exact prefixed string it fed the
// encoder plus its tokens, and streams the decoded answer one token at a time (see "see inside").

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/t5-base";
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
  console.log(`[t5-base worker] loading ${MODEL_ID} on wasm (q8) — original task-prefix T5`);
  pipe = await pipeline("text2text-generation", MODEL_ID, {
    device: "wasm",
    dtype: "q8",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  tokenizer = pipe.tokenizer;
  device = "wasm";
  console.log("[t5-base worker] ready on wasm");
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

async function generate(id, input, opts) {
  await ensureLoaded();
  const beams = Math.max(1, opts.numBeams | 0 || 1);
  const sample = !!opts.doSample && beams === 1;
  const gen = {
    max_new_tokens: Math.max(1, opts.maxNewTokens | 0 || 64),
    num_beams: beams,
    do_sample: sample,
    ...(sample ? { temperature: opts.temperature ?? 0.9, top_k: opts.topK ?? 50 } : {}),
    no_repeat_ngram_size: 3,
  };

  const times = [];
  let partial = "";
  // Per-token streaming/timing is only clean for a single sequence (greedy or sampling, beams === 1);
  // beam search reorders tokens at the end, so we skip the streamer and report aggregate timing honestly.
  if (beams === 1) {
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
    beams,
    sampled: sample,
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
    console.error("[t5-base worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
