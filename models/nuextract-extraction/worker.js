// NuExtract-1.5-tiny structured-extraction worker — runs ALL generation off the main thread.
//
// This is NOT a general chat LLM. NuExtract is a small model (a fine-tune of Qwen2.5-0.5B) trained for
// ONE job: given a TEXT and a JSON TEMPLATE describing the fields you want, emit JSON that fills the
// template using spans copied from the text. It's "pure extraction" — the model is trained to output
// values that appear verbatim in the source, not to paraphrase or invent. That template-conditioning is
// the whole point, and it's why this is a distinct demo from the Qwen chat pages.
//
// Prompt format (from the NuExtract-1.5 model card):
//   <|input|>
//   ### Template:
//   {template, pretty-printed}
//   ### Text:
//   {text}
//
//   <|output|>
// The model then generates the filled JSON, terminated by <|end-output|>/EOS.
//
// Model: onnx-community/NuExtract-1.5-tiny-ONNX (text-generation). dtype "q4" — the q8/int8/uint8
// exports of THIS fine-tune are degenerate (they emit an immediate EOS → empty output); q4 (onnx/
// model_q4.onnx) is the honest runnable export, verified to produce valid extraction JSON. WASM
// (WebGPU when a real adapter exists). Transformers.js via the SHARED CDN url.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/NuExtract-1.5-tiny-ONNX";
const DEVICE = "wasm";
const DTYPE = "q4";
let generator = null;
let mod = null;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (generator) return;
  mod = await import(TRANSFORMERS_URL);
  const { pipeline } = mod;
  console.log(`[nuextract worker] loading ${MODEL_ID} on ${DEVICE} (${DTYPE})`);
  generator = await pipeline("text-generation", MODEL_ID, {
    device: DEVICE,
    dtype: DTYPE,
    progress_callback: (p) => post({ type: "progress", p }),
  });
  console.log(`[nuextract worker] ready on ${DEVICE}`);
  post({ type: "ready", device: DEVICE });
}

/** Build the exact NuExtract prompt. Throws if the template isn't valid JSON. */
function buildPrompt(templateStr, text) {
  const pretty = JSON.stringify(JSON.parse(templateStr), null, 4); // validates the template
  return `<|input|>\n### Template:\n${pretty}\n### Text:\n${text}\n\n<|output|>`;
}

async function extract(id, templateStr, text, maxTokens) {
  await ensureLoaded();
  const { TextStreamer } = mod;

  let prompt;
  try {
    prompt = buildPrompt(templateStr, text);
  } catch {
    post({ type: "error", id, message: "Template is not valid JSON — fix it and try again." });
    return;
  }
  post({ type: "prompt", id, prompt });

  let acc = "";
  let count = 0;
  const t0 = performance.now();
  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (token) => {
      count++;
      acc += token;
      post({ type: "token", id, token });
    },
  });

  // Pure extraction wants (near-)greedy decoding — the card recommends temperature ≈ 0.
  const out = await generator(prompt, {
    max_new_tokens: Math.max(16, Math.min(1024, maxTokens ?? 512)),
    do_sample: false,
    repetition_penalty: 1.0,
    streamer,
    return_full_text: true,
  });
  const ms = Math.round(performance.now() - t0);

  // The pipeline returns prompt + generation; the answer is everything after the final <|output|>.
  const full = Array.isArray(out) ? (out[0]?.generated_text ?? "") : String(out ?? "");
  let raw = full.includes("<|output|>")
    ? full.split("<|output|>").slice(1).join("<|output|>")
    : acc;
  raw = raw.replace(/<\|end-output\|>[\s\S]*$/, "").replace(/<\|endoftext\|>[\s\S]*$/, "").trim();

  // Validate as JSON (the model is trained to emit JSON; we prove it's machine-readable).
  let parsed = null;
  let valid = false;
  try {
    parsed = JSON.parse(raw);
    valid = true;
  } catch { /* leave raw; page shows it's not valid JSON */ }

  post({
    type: "result",
    id,
    prompt,
    raw,
    parsed: valid ? parsed : null,
    valid,
    tokens: count,
    ms,
    device: DEVICE,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded();
    } else if (type === "run") {
      await extract(e.data.id, e.data.template, e.data.text, e.data.maxTokens);
    }
  } catch (err) {
    console.error("[nuextract worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
