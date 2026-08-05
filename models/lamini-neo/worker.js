// LaMini-Neo-125M worker — instruction following off the main thread, WASM int8.
// Model: Xenova/LaMini-Neo-125M (task: text-generation).
//
// LaMini-Neo-125M is MBZUAI's instruction-tuned 125M GPT-Neo: the raw EleutherAI GPT-Neo-125M
// fine-tuned on ~2.6M instruction-response pairs (GPT4All/Alpaca-style). It's a tiny model that
// follows simple instructions — explain, summarize, rewrite, draft — with a couple of hundred
// megabytes of weights. We use the LaMini prompt format ("### Instruction:\n...\n\n### Response:\n")
// with the text-generation pipeline and a TextStreamer, so every emitted token streams to the page
// as it is produced — real on-device inference, nothing canned.
//
// LICENSE: CC-BY-NC-4.0 (non-commercial weights) — visible note on the page, per showcase policy.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/LaMini-Neo-125M";
const DTYPE = "int8";
let generator = null;
let mod = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function makePipeline() {
  const { pipeline } = mod;
  return pipeline("text-generation", "Xenova/LaMini-Neo-125M", {
    device,
    dtype: DTYPE,
    progress_callback: (p) => post({ type: "progress", p }),
  });
}

async function ensureLoaded() {
  if (generator) return;
  mod = await import(TRANSFORMERS_URL);
  generator = await makePipeline();
  post({ type: "ready", device });
}

// Build the LaMini instruction prompt (format from the model card).
function buildPrompt(instruction) {
  return `### Instruction:\n${instruction}\n\n### Response:\n`;
}

async function generate(id, instruction, opts) {
  await ensureLoaded();
  const { TextStreamer } = mod;
  const maxNew = Math.min(192, Math.max(1, opts.maxNew ?? 96));
  const greedy = !!opts.greedy;
  const prompt = buildPrompt(instruction);
  post({ type: "prompt", id, prompt });

  let count = 0;
  const t0 = performance.now();
  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (token) => {
      count++;
      post({ type: "token", id, piece: token, t: performance.now() - t0 });
    },
  });
  const out = await generator(prompt, {
    max_new_tokens: maxNew,
    do_sample: !greedy,
    temperature: greedy ? 1 : (opts.temperature ?? 0.7),
    top_k: greedy ? undefined : (opts.topK ?? 40),
    repetition_penalty: 1.1,
    streamer,
    return_full_text: false,
  });
  const ms = Math.round(performance.now() - t0);
  post({
    type: "done",
    id,
    text: String(out?.[0]?.generated_text ?? ""),
    ms,
    tokens: count,
    device,
    promptTemplate: prompt,
  });
}

self.addEventListener("message", async (e) => {
  const { type, id } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "generate") await generate(id, e.data.instruction, e.data.opts || {});
  } catch (err) {
    post({ type: "error", id, message: String(err?.message ?? err) });
  }
});
