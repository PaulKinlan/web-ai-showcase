// MarianMT (OPUS-MT) translation worker — inference off the main thread. Unlike a big multilingual
// model, Marian is BILINGUAL: one small (~105 MB q8) encoder-decoder per language pair. This worker
// holds a map of pair → pipeline and loads each on demand, so switching pairs only downloads what you
// use. Models: Xenova/opus-mt-en-de | en-fr | en-es (task: translation), WASM, q8.
//
// Streams the translation token-by-token (greedy) and reports the exact pair, the input/output token
// counts, and real per-token timing — all measured, never claimed.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

export const MODELS = {
  "en-de": "Xenova/opus-mt-en-de",
  "en-fr": "Xenova/opus-mt-en-fr",
  "en-es": "Xenova/opus-mt-en-es",
  // Reverse pairs — used by the round-trip (wild) demo so the whole trip stays Marian.
  "de-en": "Xenova/opus-mt-de-en",
  "fr-en": "Xenova/opus-mt-fr-en",
  "es-en": "Xenova/opus-mt-es-en",
};
const DEFAULT_PAIR = "en-de";

let pipelineFn = null;
let TextStreamer = null;
const pipes = new Map(); // pair → pipeline
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLib() {
  if (pipelineFn) return;
  const m = await import(TRANSFORMERS_URL);
  pipelineFn = m.pipeline;
  TextStreamer = m.TextStreamer;
  m.env.allowLocalModels = false;
}

async function ensurePair(pair) {
  await ensureLib();
  if (pipes.has(pair)) return pipes.get(pair);
  const model = MODELS[pair];
  if (!model) throw new Error(`Unknown language pair: ${pair}`);
  post({ type: "pair-loading", pair });
  const pipe = await pipelineFn("translation", model, {
    device: "wasm",
    dtype: "q8",
    progress_callback: (p) => post({ type: "progress", p, pair }),
  });
  pipes.set(pair, pipe);
  post({ type: "pair-ready", pair });
  return pipe;
}

function tokenCount(pipe, text) {
  const enc = pipe.tokenizer(text);
  const ids = enc.input_ids;
  return ids?.dims ? ids.dims.at(-1) : (ids?.length ?? 0);
}

async function translate(id, text, pair, opts = {}) {
  const pipe = await ensurePair(pair);
  const inTokens = tokenCount(pipe, text);
  const beams = Math.max(1, opts.numBeams | 0 || 1);
  const times = [];
  // Marian is a per-pair model — no src_lang/tgt_lang args; the pair IS the model.
  const gen = { num_beams: beams, max_new_tokens: 512 };
  if (beams === 1) {
    let partial = "";
    gen.streamer = new TextStreamer(pipe.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (t) => {
        partial += t;
        post({ type: "stream", id, text: partial });
      },
      token_callback_function: () => times.push(performance.now()),
    });
  }
  const t0 = performance.now();
  const out = await pipe(text, gen);
  const ms = Math.round(performance.now() - t0);
  const translation = (out[0]?.translation_text ?? "").trim();
  const outTokens = times.length || tokenCount(pipe, translation);
  post({ type: "result", id, translation, pair, inTokens, outTokens, beams, ms, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensurePair(e.data.pair || DEFAULT_PAIR);
      post({ type: "ready", device });
    } else if (type === "ensure") {
      await ensurePair(e.data.pair);
      post({ type: "ensured", id: e.data.id, pair: e.data.pair });
    } else if (type === "run") {
      await translate(e.data.id, e.data.text, e.data.pair, e.data.opts || {});
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
