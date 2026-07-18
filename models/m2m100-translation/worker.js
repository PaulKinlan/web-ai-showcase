// M2M100 (418M) translation worker — inference off the main thread. One multilingual model that
// translates directly between any pair of 100 languages via src_lang / tgt_lang (no English pivot).
// Model: Xenova/m2m100_418M (task: translation), WASM, q8.
//
// Streams the translation token-by-token (greedy) and reports the exact language codes used, the
// input/output token counts, and real per-token timing — all measured, never claimed.

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
  pipe = await pipeline("translation", "Xenova/m2m100_418M", {
    device: "wasm",
    dtype: "q8",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  tokenizer = pipe.tokenizer;
  device = "wasm";
  post({ type: "ready", device });
}

function tokenCount(text) {
  const enc = tokenizer(text);
  const ids = enc.input_ids;
  return ids?.dims ? ids.dims.at(-1) : (ids?.length ?? 0);
}

async function translate(id, text, srcLang, tgtLang, opts = {}) {
  await ensureLoaded();
  const inTokens = tokenCount(text);
  const beams = Math.max(1, opts.numBeams | 0 || 1);
  const times = [];
  const gen = { src_lang: srcLang, tgt_lang: tgtLang, num_beams: beams, max_new_tokens: 256 };
  if (beams === 1) {
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
  }
  const t0 = performance.now();
  const out = await pipe(text, gen);
  const ms = Math.round(performance.now() - t0);
  const translation = (out[0]?.translation_text ?? "").trim();
  const intervals = times.map((t, i) => (i ? Math.round(t - times[i - 1]) : 0)).slice(1);
  const outTokens = times.length || tokenCount(translation);
  post({
    type: "result",
    id,
    translation,
    srcLang,
    tgtLang,
    inTokens,
    outTokens,
    intervals,
    beams,
    ms,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") {
      await translate(e.data.id, e.data.text, e.data.srcLang, e.data.tgtLang, e.data.opts || {});
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
