// Text-to-SQL worker — all inference off the main thread so the control UI stays responsive.
// Model: Xenova/t5-small-awesome-text-to-sql (task: text2text-generation), WASM, q8.
//
// This is a T5-small (~60M) fine-tuned on b-mc2/sql-create-context + Clinton/Text-to-sql-v1. It is
// SCHEMA-CONDITIONED: the entire input is one string that pastes your table definitions in front of the
// natural-language question, in this exact shape:
//
//   tables:
//   CREATE TABLE students (student_id VARCHAR, name VARCHAR); CREATE TABLE attendance (student_id VARCHAR)
//   query for:List the names of students who never attended.
//
// The encoder reads the schema + the question together; the decoder writes the SQL one token at a time,
// feeding each token back to choose the next. We stream those tokens with a TextStreamer and timestamp
// each one so the "see inside" surface shows the real decode cadence, plus the exact fed string and its
// tokens. The generated SQL is REAL model output — right or wrong — never a canned string.

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
  pipe = await pipeline("text2text-generation", "Xenova/t5-small-awesome-text-to-sql", {
    device: "wasm",
    dtype: "q8",
    progress_callback: (p) => post({ type: "progress", p }),
  });
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

async function generate(id, input, opts) {
  await ensureLoaded();
  const beams = Math.max(1, opts.numBeams | 0 || 1);
  const sample = !!opts.doSample && beams === 1;
  const gen = {
    max_new_tokens: Math.max(1, opts.maxNewTokens | 0 || 128),
    num_beams: beams,
    do_sample: sample,
    ...(sample ? { temperature: opts.temperature ?? 0.7, top_k: opts.topK ?? 40 } : {}),
    // NOTE: no no_repeat_ngram_size here — SQL legitimately repeats tokens (commas, column refs,
    // parenthesised groups), so an n-gram block would corrupt otherwise-valid queries.
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
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
