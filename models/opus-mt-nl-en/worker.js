// OPUS-MT Dutch→English translation worker — inference off the main thread. A dedicated bilingual
// Marian encoder-decoder: one small (~106 MB q8) model that knows exactly ONE direction, Dutch →
// English (no language codes — the pair is baked into the weights). The reverse pair
// (English → Dutch) is loaded on demand for the wild round-trip demo.
// Models: Xenova/opus-mt-nl-en (primary) | Xenova/opus-mt-en-nl (reverse) — task: translation,
// WASM, q8. Streams the translation chunk-by-chunk (greedy) and reports real token counts and
// measured per-chunk timing — never claimed.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

export const MODELS = {
  "nl-en": "Xenova/opus-mt-nl-en",
  "en-nl": "Xenova/opus-mt-en-nl",
};
const DEFAULT_PAIR = "nl-en";

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
  // OPUS-MT is a per-pair model — no src_lang/tgt_lang args; the pair IS the model.
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
