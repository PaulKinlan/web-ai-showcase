// StarCoder (fill-in-the-middle) worker — code completion off the main thread via Transformers.js.
//
// Model: onnx-community/tiny_starcoder_py-ONNX — the ONNX export of bigcode/tiny_starcoder_py, a
// 164M-parameter member of the BigCode **StarCoder** family (architecture: gpt_bigcode). It is the
// browser-runnable representative of the StarCoder lineage: StarCoder2-3B has NO transformers.js/ONNX
// or WebLLM build today (verified — see the page + models.json blocked record), so we run the genuine
// same-family model that DOES load in a browser and demonstrates the identical fill-in-the-middle
// mechanism. Nothing is relabelled: the page says exactly what it runs.
//
// Fill-in-the-middle (FIM) is what makes a code model more than a left-to-right autocomplete. The
// tokenizer carries four sentinel tokens — <fim_prefix>, <fim_suffix>, <fim_middle>, <fim_pad> — and
// the model was trained so that, given
//     <fim_prefix>CODE_BEFORE<fim_suffix>CODE_AFTER<fim_middle>
// it generates the code that belongs BETWEEN the two halves, conditioned on BOTH sides. That is the
// "wire" this worker assembles and reports back so the see-inside surface can show it verbatim.
//
// Task: text-generation. Backend: WASM, q8 (8-bit). Runs in a Web Worker so the control UI never janks.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/tiny_starcoder_py-ONNX";
const FIM_PREFIX = "<fim_prefix>";
const FIM_SUFFIX = "<fim_suffix>";
const FIM_MIDDLE = "<fim_middle>";
const SPECIAL_RE = /<\|endoftext\|>|<fim_(?:prefix|suffix|middle|pad)>/g;

let gen = null;
let device = "wasm";
let TextStreamer = null;
let InterruptableStoppingCriteria = null;
let stopper = null;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (gen) return;
  const mod = await import(TRANSFORMERS_URL);
  const { pipeline, env } = mod;
  TextStreamer = mod.TextStreamer;
  InterruptableStoppingCriteria = mod.InterruptableStoppingCriteria;
  env.allowLocalModels = false; // let the library own its Cache Storage; don't fight the SW.
  console.log(`[starcoder-fim worker] loading ${MODEL_ID} on wasm (q8) — code FIM/completion`);
  gen = await pipeline("text-generation", MODEL_ID, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  console.log("[starcoder-fim worker] ready on wasm");
  post({ type: "ready", device });
}

// Assemble the exact string the model sees. FIM interleaves the sentinels; plain completion is just
// the code continued left-to-right. Returned as `wire` so the page can show it verbatim (see-inside).
function buildWire(mode, { prefix, suffix, code }) {
  if (mode === "fim") return FIM_PREFIX + (prefix ?? "") + FIM_SUFFIX + (suffix ?? "") + FIM_MIDDLE;
  return code ?? "";
}

async function run(id, opts) {
  await ensureLoaded();
  const { mode, maxTokens = 96, temperature = 0.2 } = opts;
  const wire = buildWire(mode, opts);
  stopper = InterruptableStoppingCriteria ? new InterruptableStoppingCriteria() : null;

  const t0 = performance.now();
  let ttft = null;
  let ntok = 0;

  const streamer = new TextStreamer(gen.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    token_callback_function: () => {
      ntok++; // real generated-token count, for honest tok/s
    },
    callback_function: (text) => {
      if (!text) return;
      if (ttft === null) {
        ttft = performance.now() - t0;
        post({ type: "first", id, t: Math.round(ttft) });
      }
      post({ type: "token", id, delta: text });
    },
  });

  const doSample = temperature > 0;
  const out = await gen(wire, {
    max_new_tokens: maxTokens,
    do_sample: doSample,
    ...(doSample ? { temperature, top_p: 0.95 } : {}),
    return_full_text: false, // stream/return ONLY the newly generated middle/continuation
    streamer,
    ...(stopper ? { stopping_criteria: stopper } : {}),
  });

  const ms = Math.round(performance.now() - t0);
  let middle = (out?.[0]?.generated_text ?? "").replace(SPECIAL_RE, "");
  // The model often emits a trailing newline run once it "closes" the middle; trim only the tail.
  middle = middle.replace(/\s+$/, (m) => (m.includes("\n") ? "\n" : ""));
  let tokens = ntok;
  if (!tokens) {
    try {
      tokens = gen.tokenizer.encode(middle).length;
    } catch {
      tokens = 0;
    }
  }
  post({
    type: "done",
    id,
    middle,
    wire,
    ms,
    ttft: ttft === null ? ms : Math.round(ttft),
    tokens,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded();
    } else if (type === "run") {
      await run(e.data.id, e.data.opts);
    } else if (type === "stop") {
      if (stopper) stopper.interrupt();
    }
  } catch (err) {
    console.error("[starcoder-fim worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
