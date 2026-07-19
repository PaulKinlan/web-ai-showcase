// Cleanup worker for the wav2vec2 multi-model demo. wav2vec2 (CTC) emits ALL-CAPS text with no
// punctuation — it only knows 26 letters, an apostrophe and word boundaries. So we chain a small
// on-device instruct LLM (onnx-community/Qwen2.5-0.5B-Instruct, already shipped elsewhere in the
// showcase) to restore casing and punctuation. A real second model, off the main thread.
//
// Primary path: WebGPU + q4f16. Honest WASM q4 fallback when there's no GPU adapter (slower, still real).

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/Qwen2.5-0.5B-Instruct";
let generator = null;
let mod = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function webgpuUsable() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) return false;
  try {
    return !!(await navigator.gpu.requestAdapter());
  } catch {
    return false;
  }
}

async function ensureLoaded() {
  if (generator) return;
  mod = await import(TRANSFORMERS_URL);
  const { pipeline } = mod;
  const useGpu = await webgpuUsable();
  const dev = useGpu ? "webgpu" : "wasm";
  const dtype = useGpu ? "q4f16" : "q4";
  console.log(`[cleanup worker] loading ${MODEL_ID} on ${dev} (${dtype})`);
  generator = await pipeline("text-generation", MODEL_ID, {
    device: dev,
    dtype,
    progress_callback: (p) => post({ type: "progress", p }),
  });
  device = dev;
  console.log(`[cleanup worker] ready on ${dev}`);
  post({ type: "ready", device: dev });
}

const SYSTEM =
  "You restore punctuation and capitalisation to raw speech-to-text output. The user gives you ALL-CAPS " +
  "text with no punctuation. Return the SAME words with natural sentence casing and punctuation added. " +
  "Do not add, remove, or reword anything — only fix casing and punctuation. Reply with only the fixed text.";

async function cleanup(id, transcript) {
  await ensureLoaded();
  const { TextStreamer } = mod;
  const messages = [
    { role: "system", content: SYSTEM },
    { role: "user", content: transcript },
  ];
  let text = "";
  const t0 = performance.now();
  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (t) => {
      text += t;
      post({ type: "token", id, token: t });
    },
  });
  const out = await generator(messages, {
    max_new_tokens: 160,
    do_sample: false,
    repetition_penalty: 1.1,
    streamer,
    return_full_text: false,
  });
  const ms = Math.round(performance.now() - t0);
  const full = out?.[0]?.generated_text;
  const finalText = (Array.isArray(full) ? (full.at(-1)?.content ?? "") : String(full ?? "")) ||
    text;
  post({ type: "result", id, text: finalText.trim(), ms, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded();
    } else if (type === "run") {
      await cleanup(e.data.id, e.data.transcript);
    }
  } catch (err) {
    console.error("[cleanup worker]", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
