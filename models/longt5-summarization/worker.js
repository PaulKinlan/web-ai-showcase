// LongT5 (TGlobal, base) summarization worker — inference off the main thread so the UI stays smooth.
// Model: Xenova/long-t5-tglobal-base (task: summarization), WASM, q8.
//
// Why this model: the ask was a Pegasus-family extreme/XSum summarizer, but Pegasus has no
// transformers.js ONNX build. LongT5 is the closest DISTINCT, browser-runnable relative: it was
// pretrained with Pegasus's own gap-sentence (principle-sentence) generation objective, and its
// transient-global attention lets it read far LONGER inputs than DistilBART's ~1024-token BART. So it
// fills a genuinely different architecture family, honestly. (This is the base checkpoint, not a
// summarization-fine-tune, so its gists lean extractive — it lifts the most salient sentence(s).)
//
// Operations:
//   run    → summarize one text with length/beam controls; streams token-by-token, reports real
//            per-token timing, input/output token counts, and latency.
//   recap  → recursively summarize (feed each summary back in) until it collapses toward one sentence.
// All timings are measured, never claimed.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL = "Xenova/long-t5-tglobal-base";

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
  pipe = await pipeline("summarization", MODEL, {
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
  // Per-token streaming timing is only meaningful for greedy decoding (beams === 1); beam search
  // reorders tokens at the end, so we skip the streamer and report aggregate timing honestly.
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
  const intervals = times.map((t, i) => (i ? Math.round(t - times[i - 1]) : 0)).slice(1);
  const outTokens = times.length || tokenCount(summary);
  return { summary, ms, intervals, outTokens, beams };
}

async function summarize(id, text, opts) {
  await ensureLoaded();
  const inTokens = tokenCount(text);
  // Never ask for more output than ~90% of the input — keeps short-input summaries from padding/looping.
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
    const budget = Math.max(opts.floor ?? 18, Math.round(inTokens * 0.55));
    const r = await generate(current, {
      maxLength: budget,
      minLength: Math.min(opts.floor ?? 12, budget - 1),
      numBeams: opts.numBeams ?? 1,
      lengthPenalty: 1.0,
    });
    rounds.push({ round: i + 1, inTokens, outTokens: r.outTokens, summary: r.summary, ms: r.ms });
    post({ type: "recap-step", id, step: rounds[rounds.length - 1] });
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
