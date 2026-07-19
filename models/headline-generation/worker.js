// Headline / title generation worker — inference off the main thread so the control UI stays smooth.
// Model: ldenoue/Title_Generation_T5Small_Model (task: text2text-generation), WASM, q8.
//
// This is a T5 seq2seq fine-tuned on the 190k-Medium-Articles title dataset (lineage:
// fabiochiu/t5-*-medium-title-generation). It takes an ARTICLE / paragraph and writes a short TITLE for
// it — NOT a summary of the whole thing, a single catchy line. The training convention reuses T5's
// "summarize: " task prefix, so the worker prepends it to whatever article text you give it.
//
// The demo's point is that title-writing is a GENERATION problem with no single right answer: with greedy/
// beam decoding you get one "best" headline; turn on sampling (temperature + top-k) and the SAME model on
// the SAME article proposes many DIFFERENT candidate headlines — an A/B set a copywriter could pick from.
// So the worker returns the exact fed string + its tokens, streams the best headline token-by-token, and
// (optionally) draws N sampled candidates.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "ldenoue/Title_Generation_T5Small_Model";
// The fine-tune reuses T5's "summarize:" task prefix as its input convention (see model lineage).
export const PREFIX = "summarize: ";

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
  console.log(`[headline worker] loading ${MODEL_ID} on wasm (q8) — T5 title generation`);
  pipe = await pipeline("text2text-generation", MODEL_ID, {
    device: "wasm",
    dtype: "q8",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  tokenizer = pipe.tokenizer;
  device = "wasm";
  console.log("[headline worker] ready on wasm");
  post({ type: "ready", device });
}

/** Encode a string → { count, tokens:string[] } exposing T5's SentencePiece ▁ word-boundary marks. */
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

const clean = (s) => (s ?? "").replace(/\s+/g, " ").trim();

async function generate(id, article, opts) {
  await ensureLoaded();
  const input = PREFIX + (article || "").trim();
  const maxNew = Math.min(64, Math.max(6, opts.maxNewTokens | 0 || 24));
  const nCand = Math.min(6, Math.max(0, opts.numCandidates | 0));
  const temperature = opts.temperature ?? 1.0;
  const topK = opts.topK ?? 50;
  const noRepeat = opts.noRepeat ?? 2;

  const inTok = tokenize(input);

  // 1) The "best" headline — beam search (deterministic), streamed token-by-token for the "see inside".
  const times = [];
  let partial = "";
  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (t) => {
      partial += t;
      post({ type: "stream", id, text: clean(partial) });
    },
    token_callback_function: () => times.push(performance.now()),
  });
  const t0 = performance.now();
  const bestOut = await pipe(input, {
    max_new_tokens: maxNew,
    num_beams: 4,
    no_repeat_ngram_size: noRepeat,
    streamer,
  });
  const bestMs = Math.round(performance.now() - t0);
  const best = clean(bestOut[0]?.generated_text);
  const intervals = times.map((t, i) => (i ? Math.round(t - times[i - 1]) : 0)).slice(1);
  const bestTokens = times.length || tokenize(best).count;

  // 2) N sampled candidates — SAME model, SAME article, different draws (the copywriter's A/B set).
  const candidates = [];
  for (let i = 0; i < nCand; i++) {
    const ct0 = performance.now();
    const out = await pipe(input, {
      max_new_tokens: maxNew,
      do_sample: true,
      temperature,
      top_k: topK,
      no_repeat_ngram_size: noRepeat,
    });
    const text = clean(out[0]?.generated_text);
    candidates.push({
      text,
      tokens: tokenize(text).count,
      ms: Math.round(performance.now() - ct0),
    });
  }

  const totalMs = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    input,
    prefix: PREFIX,
    best,
    bestTokens,
    intervals,
    candidates,
    inTokens: inTok.count,
    inTokenStrings: inTok.tokens,
    temperature,
    topK,
    bestMs,
    ms: totalMs,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await generate(e.data.id, e.data.article, e.data.opts || {});
  } catch (err) {
    console.error("[headline worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
