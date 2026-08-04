// llama2.c-stories15M worker — story generation off the main thread, WASM int8.
// Model: Xenova/llama2.c-stories15M (task: text-generation).
//
// This is Karpathy's llama2.c 15M-parameter Llama, trained from scratch on TinyStories —
// ~2M short, simple stories written for 3–4 year olds. Despite being ~15MB of weights, it
// learned English grammar, sentence structure, and simple narrative arcs ("once upon a time…",
// a problem, a resolution). It's the smallest model on this showcase that can genuinely tell
// a story. We use the text-generation pipeline with a TextStreamer so every token streams to
// the page as it is produced — real on-device inference, nothing canned.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/llama2.c-stories15M";
const DTYPE = "int8";
let generator = null;
let mod = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function makePipeline() {
  const { pipeline } = mod;
  return pipeline("text-generation", MODEL_ID, {
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

async function generate(id, prompt, opts) {
  await ensureLoaded();
  const { TextStreamer } = mod;
  const maxNew = Math.min(160, Math.max(1, opts.maxNew ?? 80));
  const greedy = !!opts.greedy;
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
    temperature: greedy ? 1 : (opts.temperature ?? 0.8),
    top_k: greedy ? undefined : (opts.topK ?? 40),
    repetition_penalty: 1.1,
    streamer,
    return_full_text: false,
  });
  const ms = Math.round(performance.now() - t0);
  post({
    type: "done",
    id,
    text: prompt + String(out?.[0]?.generated_text ?? ""),
    generated: String(out?.[0]?.generated_text ?? ""),
    ms,
    tokens: count,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type, id } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "generate") await generate(id, e.data.prompt, e.data.opts || {});
  } catch (err) {
    post({ type: "error", id, message: String(err?.message ?? err) });
  }
});
