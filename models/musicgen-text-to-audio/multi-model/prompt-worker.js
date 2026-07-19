// Prompt-writer worker for the MusicGen multi-model demo. A small on-device instruct LLM turns a short
// mood ("rainy Sunday", "victory") into a vivid MusicGen prompt — then the page hands that prompt to
// MusicGen. This is a REAL second model, off the main thread. Same model the showcase already ships on
// its own page (onnx-community/Qwen2.5-0.5B-Instruct), so it's a proven browser-runnable choice.
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
  console.log(`[prompt worker] loading ${MODEL_ID} on ${dev} (${dtype})`);
  generator = await pipeline("text-generation", MODEL_ID, {
    device: dev,
    dtype,
    progress_callback: (p) => post({ type: "progress", p }),
  });
  device = dev;
  console.log(`[prompt worker] ready on ${dev}`);
  post({ type: "ready", device: dev });
}

const SYSTEM =
  "You are a music prompt writer for a text-to-music model. Given a mood or theme, reply " +
  "with ONE short vivid music description (10-20 words): genre, instruments, tempo/energy. " +
  "No quotes, no preamble, no explanation — just the description.";

async function write(id, mood) {
  await ensureLoaded();
  const { TextStreamer } = mod;
  const messages = [
    { role: "system", content: SYSTEM },
    { role: "user", content: `Mood: ${mood}` },
  ];
  let text = "";
  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (t) => {
      text += t;
      post({ type: "token", id, token: t });
    },
  });
  const out = await generator(messages, {
    max_new_tokens: 60,
    do_sample: true,
    temperature: 0.8,
    top_p: 0.9,
    repetition_penalty: 1.1,
    streamer,
    return_full_text: false,
  });
  const full = out?.[0]?.generated_text;
  const finalText = (Array.isArray(full) ? (full.at(-1)?.content ?? "") : String(full ?? "")) ||
    text;
  // Clean up: single line, strip surrounding quotes.
  const cleaned = finalText.trim().split("\n")[0].replace(/^["'\s]+|["'\s]+$/g, "");
  post({ type: "result", id, prompt: cleaned, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded();
    } else if (type === "run") {
      await write(e.data.id, e.data.mood);
    }
  } catch (err) {
    console.error("[prompt worker]", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
