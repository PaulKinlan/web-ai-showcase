// Narrator LLM worker for the SmolVLM2 multi-model page — runs a small on-device text LLM off the
// main thread to weave SmolVLM2's per-frame captions into one flowing narration. VLM → LLM, the
// canonical multi-model composition. Model: onnx-community/Qwen2.5-0.5B-Instruct (text-generation),
// WebGPU, dtype q4f16 (same build the qwen-tiny-llm demo uses). Shared 3.7.5 pin.

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
    dtype: "q4f16",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  post({ type: "ready", device: "webgpu" });
}

async function narrate(id, captions, style, maxTokens) {
  await ensureLoaded();
  const { TextStreamer } = mod;
  const lines = captions.map((c, i) => `Frame ${i + 1}: ${c}`).join("\n");
  const messages = [
    {
      role: "system",
      content:
        "You are a concise narrator. You are given short captions describing consecutive frames of a scene, in order. Weave them into one flowing description of what is happening, in " +
        (style || "a neutral, factual tone") +
        ". Do not invent details that aren't implied by the captions.",
    },
    { role: "user", content: `Here are the frame captions:\n${lines}\n\nNarrate the scene.` },
  ];
  const template = generator.tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    tokenize: false,
  });
  post({ type: "prompt", id, template });

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
  await generator(messages, { max_new_tokens: maxTokens ?? 200, do_sample: false, streamer });
  post({ type: "done", id, ms: Math.round(performance.now() - t0), tokens: count });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "probe") post({ type: "probe-result", gpu: await probeGPU() });
    else if (type === "load") await ensureLoaded();
    else if (type === "narrate") {
      await narrate(e.data.id, e.data.captions, e.data.style, e.data.maxTokens);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
