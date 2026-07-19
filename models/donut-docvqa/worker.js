// Donut document-VQA worker — all inference off the main thread so the control UI stays responsive.
// Donut is OCR-FREE: instead of running OCR then reading the text, a Swin Transformer encoder looks at
// the whole document image and a BART-style decoder generates the answer directly, conditioned on a
// task prompt that embeds your question. We stream each decoded answer token back to the page (via a
// TextStreamer) with a timestamp, so the page can show the answer forming and the per-token timing.
//
// Model: Xenova/donut-base-finetuned-docvqa (task: document-question-answering), WASM, q8. ~215 MB.

import { loadPipeline, TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let device = "wasm";
let mod = null;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  mod = await import(TRANSFORMERS_URL);
  const loaded = await loadPipeline({
    task: "document-question-answering",
    model: "Xenova/donut-base-finetuned-docvqa",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

// The exact decoder task prompt Donut builds for DocVQA — deterministic, and what the model actually
// receives. Shown verbatim in "See inside".
function buildPrompt(question) {
  return `<s_docvqa><s_question>${question}</s_question><s_answer>`;
}

async function ask(id, imageURL, question, maxTokens) {
  await ensureLoaded();
  const { TextStreamer } = mod;
  const prompt = buildPrompt(question);
  post({ type: "prompt", id, prompt });
  const t0 = performance.now();
  let i = 0;

  // Best-effort token streaming for the "See inside" trace. If the pipeline ignores the streamer we
  // still resolve with the real final answer below — never a faked result.
  let streamer;
  try {
    streamer = new TextStreamer(pipe.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (token) => {
        if (!token) return;
        post({ type: "token", id, token, t: Math.round(performance.now() - t0), i: i++ });
      },
    });
  } catch { /* tokenizer/streamer unavailable — fall back to final answer only */ }

  const out = await pipe(imageURL, question, {
    max_new_tokens: Math.max(1, Math.min(128, maxTokens || 64)),
    ...(streamer ? { streamer } : {}),
  });

  const first = Array.isArray(out) ? out[0] : out;
  const answer = (first?.answer ?? first?.generated_text ?? "").trim();
  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, answer, prompt, tokens: i, ms, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded();
    } else if (type === "run") {
      await ask(e.data.id, e.data.image, e.data.question, e.data.maxTokens);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
