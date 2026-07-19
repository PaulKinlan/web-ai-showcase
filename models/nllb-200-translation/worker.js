// NLLB-200 (distilled 600M) translation worker — all inference off the main thread. ONE model that
// translates directly between any pair of 200 languages (many low-resource) via FLORES-200 codes.
// Model: Xenova/nllb-200-distilled-600M (task: translation), WebAssembly, q8 (8-bit) ONNX.
//
// How NLLB picks the output language: the decoder is forced to START with the target language's BOS
// token (`forced_bos_token_id`). transformers.js derives that from `tgt_lang`; we also read the exact
// token id back out and report it, so the "see inside" surface can show the real mechanism — the
// language codes used, that forced BOS token, the input/output token counts, and measured timing.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let tokenizer = null;
let TextStreamer = null;
const device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const { pipeline, TextStreamer: TS, env } = await import(TRANSFORMERS_URL);
  TextStreamer = TS;
  env.allowLocalModels = false;
  // NLLB-200 distilled 600M is bigger than M2M100 (~900 MB q8); WASM runs anywhere, no WebGPU needed.
  pipe = await pipeline("translation", "Xenova/nllb-200-distilled-600M", {
    device: "wasm",
    dtype: "q8",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  tokenizer = pipe.tokenizer;
  post({ type: "ready", device });
}

function tokenCount(text) {
  const enc = tokenizer(text);
  const ids = enc.input_ids;
  return ids?.dims ? ids.dims.at(-1) : (ids?.length ?? 0);
}

// The real forced-BOS token id for a FLORES code — the single value that steers the output language.
// transformers.js exposes the code→id lookup in a few shapes across versions; try them, else null.
function forcedBosId(code) {
  try {
    if (typeof tokenizer.convert_tokens_to_ids === "function") {
      const id = tokenizer.convert_tokens_to_ids([code]);
      const v = Array.isArray(id) ? id[0] : id;
      if (typeof v === "number" && v >= 0) return v;
    }
  } catch { /* fall through */ }
  try {
    const m = tokenizer.lang_to_token_id || tokenizer.lang_code_to_id;
    if (m && m[code] != null) return m[code];
  } catch { /* fall through */ }
  try {
    const t2i = tokenizer.model?.tokens_to_ids;
    if (t2i?.get) {
      const v = t2i.get(code);
      if (typeof v === "number") return v;
    }
  } catch { /* fall through */ }
  return null;
}

async function translate(id, text, srcCode, tgtCode, opts = {}) {
  await ensureLoaded();
  const inTokens = tokenCount(text);
  const forcedBos = forcedBosId(tgtCode);
  const beams = Math.max(1, opts.numBeams | 0 || 1);
  const times = [];
  const gen = { src_lang: srcCode, tgt_lang: tgtCode, num_beams: beams, max_new_tokens: 256 };
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
  const outTokens = times.length || tokenCount(translation);
  post({
    type: "result",
    id,
    translation,
    srcCode,
    tgtCode,
    forcedBos,
    inTokens,
    outTokens,
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
      await translate(e.data.id, e.data.text, e.data.srcCode, e.data.tgtCode, e.data.opts || {});
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
