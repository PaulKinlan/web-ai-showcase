// Summarizer LLM worker for the SmolDocling multi-model page — runs a small on-device text LLM off the
// main thread to turn SmolDocling's structured document (as Markdown) into a plain-language brief.
// Doc-VLM → LLM, the canonical multi-model composition. Model: onnx-community/Qwen2.5-0.5B-Instruct
// (text-generation), WebGPU, dtype q4f16 (the same build the smolvlm2 narrator uses). SHARED 3.7.5 pin
// (this is a plain text LLM, unrelated to SmolDocling's v4-pinned worker).

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/Qwen2.5-0.5B-Instruct";
let generator = null;
let mod = null;

function post(msg) {
  self.postMessage(msg);
}

async function probeGPU() {
  if (!("gpu" in navigator)) return { ok: false, reason: "no-gpu" };
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { ok: false, reason: "no-adapter" };
    return { ok: true };
  } catch (e) {
    return { ok: false, reason: "adapter-error", detail: String(e?.message ?? e) };
  }
}

async function ensureLoaded() {
  if (generator) return;
  mod = await import(TRANSFORMERS_URL);
  const { pipeline } = mod;
  generator = await pipeline("text-generation", MODEL_ID, {
    device: "webgpu",
    dtype: "q4", // q4 (not q4f16): the fp16-compute q4f16 export degenerates on some WebGPU backends
    progress_callback: (p) => post({ type: "progress", p }),
  });
  post({ type: "ready", device: "webgpu" });
}

async function summarize(id, docMarkdown, maxTokens) {
  await ensureLoaded();
  const { TextStreamer } = mod;
  const messages = [
    {
      role: "system",
      content:
        "You are a concise analyst. You are given a document that has been converted to Markdown (its structure — headings, paragraphs, tables — is preserved). Write a short plain-language brief (2-4 sentences) of what the document says and its key figures. Only use information present in the document; do not invent details.",
    },
    { role: "user", content: `Document:\n\n${docMarkdown}\n\nWrite the brief.` },
  ];

  const t0 = performance.now();
  let count = 0;
  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (tok) => {
      count++;
      post({ type: "token", id, token: tok });
    },
  });
  // Sampling (not greedy) — the canonical config the built qwen-tiny-llm demo uses. Greedy decoding on
  // this 0.5B model degenerates into repeated garbage on structured input; sampling + a light repetition
  // penalty keep the brief coherent (verified: produces an accurate multi-sentence summary).
  await generator(messages, {
    max_new_tokens: maxTokens ?? 180,
    do_sample: true,
    temperature: 0.7,
    top_p: 0.9,
    top_k: 50,
    repetition_penalty: 1.1,
    return_full_text: false,
    streamer,
  });
  post({ type: "done", id, ms: Math.round(performance.now() - t0), tokens: count });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "probe") post({ type: "probe-result", gpu: await probeGPU() });
    else if (type === "load") await ensureLoaded();
    else if (type === "summarize") await summarize(e.data.id, e.data.doc, e.data.maxTokens);
  } catch (err) {
    console.error("[summarize worker]", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
