// DistilBART CNN 6-6 summarization worker — inference off the main thread so the control UI stays smooth.
// Model: Xenova/distilbart-cnn-6-6 (task: summarization), WASM, q8.
//
// Operations:
//   run       → summarize one text with length/beam controls; streams the summary token-by-token and
//               reports real per-token generation timing, input/output token counts, and latency.
//   recap     → recursively summarize (feed each summary back in) until it collapses toward one sentence.
//
// All timings are measured, never claimed. Streaming uses transformers.js TextStreamer so the "see
// inside" per-token timeline is the model's own decode cadence, not a synthetic animation.

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
  pipe = await pipeline("summarization", "Xenova/distilbart-cnn-6-6", {
    device: "wasm",
    dtype: "q8",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  tokenizer = pipe.tokenizer;
  device = "wasm";
  post({ type: "ready", device });
}

/** Count tokens the model actually sees for a string (input_ids length). */
function tokenCount(text) {
  const enc = tokenizer(text);
  const ids = enc.input_ids;
  return ids?.dims ? ids.dims.at(-1) : (ids?.length ?? 0);
}

async function generate(text, opts, onStream) {
  const beams = Math.max(1, opts.numBeams | 0 || 1);
  const times = [];
  const gen = {
    max_new_tokens: opts.maxLength,
    min_new_tokens: Math.min(opts.minLength, opts.maxLength - 1),
    num_beams: beams,
    no_repeat_ngram_size: 3,
    length_penalty: opts.lengthPenalty ?? 1.0,
  };
  // Streaming per-token timing is only meaningful for greedy decoding (beams === 1); with beam search
  // tokens are re-ordered at the end, so we skip the streamer and report aggregate timing honestly.
  if (beams === 1) {
    let partial = "";
    gen.streamer = new TextStreamer(tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (t) => {
        partial += t;
        onStream?.(partial);
      },
      token_callback_function: () => times.push(performance.now()),
    });
  }
  const t0 = performance.now();
  const out = await pipe(text, gen);
  const ms = Math.round(performance.now() - t0);
  const summary = (out[0]?.summary_text ?? "").trim();
  // Per-token intervals (ms) from the real decode timeline; empty when beam search was used.
  const intervals = times.map((t, i) => (i ? Math.round(t - times[i - 1]) : 0)).slice(1);
  const outTokens = times.length || tokenCount(summary);
  return { summary, ms, intervals, outTokens, beams };
}

async function summarize(id, text, opts) {
  await ensureLoaded();
  const inTokens = tokenCount(text);
  // Guard: never ask for more output tokens than ~90% of the input. On short inputs a large fixed
  // length budget makes the model pad and repeat itself; capping to the input keeps summaries honest.
  const cappedMax = Math.min(opts.maxLength, Math.max(24, Math.ceil(inTokens * 0.9)));
  const effOpts = {
    ...opts,
    maxLength: cappedMax,
    minLength: Math.min(opts.minLength ?? 15, cappedMax - 1),
  };
  const r = await generate(text, effOpts, (partial) => post({ type: "stream", id, text: partial }));
  post({
    type: "result",
    id,
    summary: r.summary,
    inTokens,
    outTokens: r.outTokens,
    inChars: text.length,
    outChars: r.summary.length,
    intervals: r.intervals,
    beams: r.beams,
    ms: r.ms,
    device,
  });
}

async function recap(id, text, opts) {
  await ensureLoaded();
  const rounds = [];
  let current = text;
  const maxRounds = opts.maxRounds ?? 6;
  for (let i = 0; i < maxRounds; i++) {
    const inTokens = tokenCount(current);
    // Shrink the budget each round so it genuinely collapses toward a single sentence.
    const budget = Math.max(opts.floor ?? 18, Math.round(inTokens * 0.55));
    const r = await generate(current, {
      maxLength: budget,
      minLength: Math.min(opts.floor ?? 12, budget - 1),
      numBeams: opts.numBeams ?? 1,
      lengthPenalty: 1.0,
    });
    rounds.push({ round: i + 1, inTokens, outTokens: r.outTokens, summary: r.summary, ms: r.ms });
    post({ type: "recap-step", id, step: rounds[rounds.length - 1] });
    // Stop when it stops shrinking meaningfully or is already a single short sentence.
    const sentenceCount = (r.summary.match(/[.!?]+/g) || []).length;
    if (r.outTokens >= inTokens - 2 || (sentenceCount <= 1 && r.outTokens <= (opts.floor ?? 24))) {
      current = r.summary;
      break;
    }
    current = r.summary;
  }
  post({ type: "recap-done", id, rounds, final: current, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await summarize(e.data.id, e.data.text, e.data.opts || {});
    else if (type === "recap") await recap(e.data.id, e.data.text, e.data.opts || {});
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
