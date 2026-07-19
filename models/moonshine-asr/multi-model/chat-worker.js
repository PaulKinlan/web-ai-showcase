// Chat worker for the Moonshine multi-model demo. A small on-device instruct LLM answers whatever the
// user SPOKE (transcribed by Moonshine). A real second model, off the main thread. Same model the
// showcase already ships on its own page (onnx-community/Qwen2.5-0.5B-Instruct) — a proven choice.
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
  console.log(`[chat worker] loading ${MODEL_ID} on ${dev} (${dtype})`);
  generator = await pipeline("text-generation", MODEL_ID, {
    device: dev,
    dtype,
    progress_callback: (p) => post({ type: "progress", p }),
  });
  device = dev;
  console.log(`[chat worker] ready on ${dev}`);
  post({ type: "ready", device: dev });
}

const SYSTEM =
  "You are a concise, friendly voice assistant. Answer the user's spoken question in 1-3 " +
  "short sentences. If the question is unclear, say so briefly.";

async function answer(id, question) {
  await ensureLoaded();
  const { TextStreamer } = mod;
  const messages = [
    { role: "system", content: SYSTEM },
    { role: "user", content: question },
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
    max_new_tokens: 128,
    do_sample: true,
    temperature: 0.7,
    top_p: 0.9,
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
      await answer(e.data.id, e.data.question);
    }
  } catch (err) {
    console.error("[chat worker]", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
