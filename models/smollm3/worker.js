// SmolLM3-3B (Transformers.js / ONNX) worker — dual-mode reasoning text generation off the main
// thread. Runtime is Transformers.js (NOT WebLLM): weights are the ONNX build
// HuggingFaceTB/SmolLM3-3B-ONNX, loaded through the canonical pipeline("text-generation", …) path on
// the WebGPU backend (dtype q4f16). The page gates on a real WebGPU adapter before it ever asks us to
// load (a 3B model needs a GPU), so an absent adapter shows the honest needs-WebGPU state, never a
// faked token.
//
// SmolLM3 is a HYBRID reasoning model: its chat template exposes an `enable_thinking` switch (the
// documented /think · /no_think modes). We apply the chat template OURSELVES with that switch so the
// mode is honest and inspectable, then stream the continuation with a TextStreamer. In thinking mode
// the model emits a <think>…</think> reasoning segment before the answer — we split that reasoning
// stream from the answer stream (partial-tag holdback) so the UI can show the private trace distinctly.

import { pickDevice, TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "HuggingFaceTB/SmolLM3-3B-ONNX";
const DTYPE = "q4f16";
let generator = null;
let mod = null;
let device = "webgpu";
let stopper = null;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (generator) return;
  mod = await import(TRANSFORMERS_URL);
  const { pipeline, env } = mod;
  env.allowLocalModels = false;
  // The page requires a real WebGPU adapter before calling us (3B is impractical on WASM), so this
  // resolves to "webgpu" on any supported device; we still probe honestly rather than assume.
  device = await pickDevice("webgpu");
  console.log(`[smollm3 worker] loading ${MODEL_ID} on ${device} (${DTYPE})`);
  generator = await pipeline("text-generation", MODEL_ID, {
    device,
    dtype: DTYPE,
    progress_callback: (p) => post({ type: "progress", p }),
  });
  console.log(`[smollm3 worker] ready on ${device}`);
  post({ type: "ready", device });
}

// Split a growing content string into { reasoning, answer } around a <think>…</think> block. While the
// closing tag hasn't arrived we hold back the last few reasoning chars in case they are a partial
// "</think>", so a partial tag is never mis-shown as reasoning text.
function splitThink(text) {
  const close = text.indexOf("</think>");
  if (close !== -1) {
    let r = text.slice(0, close);
    if (r.startsWith("<think>")) r = r.slice("<think>".length);
    r = r.replace(/^\s*\n/, "");
    const a = text.slice(close + "</think>".length).replace(/^\s*\n+/, "");
    return { reasoning: r, answer: a, closed: true };
  }
  if (text.startsWith("<think>")) {
    return {
      reasoning: text.slice("<think>".length).replace(/^\s*\n/, ""),
      answer: "",
      closed: false,
    };
  }
  return { reasoning: "", answer: text, closed: false };
}

async function chat(id, messages, opts) {
  await ensureLoaded();
  const { TextStreamer } = mod;
  const thinking = opts?.thinking !== false;

  // "See inside" — apply SmolLM3's chat template OURSELVES with the enable_thinking switch so the exact
  // prompt (including the /think vs /no_think generation prompt) is honest + inspectable.
  const prompt = generator.tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    tokenize: false,
    enable_thinking: thinking,
  });
  post({ type: "prompt", id, template: prompt, thinking });

  const t0 = performance.now();
  let ttft = null;
  let firstFired = false;
  let full = "";
  let emittedReasoning = 0, emittedAnswer = 0;
  let chunks = 0, reasoningChunks = 0;

  const fireFirst = () => {
    if (!firstFired) {
      firstFired = true;
      ttft = performance.now() - t0;
      post({ type: "first", id, t: Math.round(ttft) });
    }
  };
  const emit = (kind, delta) => {
    if (!delta) return;
    if (kind === "reasoning") reasoningChunks++;
    else chunks++;
    post({ type: "token", id, kind, delta });
  };

  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (piece) => {
      if (!piece) return;
      full += piece;
      fireFirst();
      const { reasoning, answer, closed } = splitThink(full);
      const holdback = closed ? reasoning.length : Math.max(0, reasoning.length - 8);
      if (holdback > emittedReasoning) {
        emit("reasoning", reasoning.slice(emittedReasoning, holdback));
        emittedReasoning = holdback;
      }
      if (answer.length > emittedAnswer) {
        emit("answer", answer.slice(emittedAnswer));
        emittedAnswer = answer.length;
      }
    },
  });

  stopper = mod.InterruptableStoppingCriteria ? new mod.InterruptableStoppingCriteria() : null;
  const doSample = opts?.doSample ?? true;
  const genOpts = {
    max_new_tokens: Math.max(1, Math.min(2048, opts?.maxTokens ?? 512)),
    do_sample: doSample,
    repetition_penalty: 1.1,
    streamer,
    return_full_text: false,
    ...(doSample
      ? {
        temperature: opts?.temperature ?? 0.6,
        top_p: opts?.topP ?? 0.95,
        top_k: opts?.topK ?? 50,
      }
      : {}),
    ...(stopper ? { stopping_criteria: stopper } : {}),
  };

  // Pass the pre-templated STRING (not the messages array) so our enable_thinking choice is honoured.
  await generator(prompt, genOpts);

  // Final authoritative split (flush any held-back reasoning tail).
  const finalSplit = splitThink(full);
  if (finalSplit.reasoning.length > emittedReasoning) {
    emit("reasoning", finalSplit.reasoning.slice(emittedReasoning));
    emittedReasoning = finalSplit.reasoning.length;
  }
  if (finalSplit.answer.length > emittedAnswer) {
    emit("answer", finalSplit.answer.slice(emittedAnswer));
    emittedAnswer = finalSplit.answer.length;
  }

  const ms = Math.round(performance.now() - t0);
  post({
    type: "done",
    id,
    text: finalSplit.answer,
    reasoning: finalSplit.reasoning,
    thinking: (finalSplit.reasoning || "").trim().length > 0,
    ms,
    ttft: ttft === null ? ms : Math.round(ttft),
    chunks,
    reasoningChunks,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded();
    } else if (type === "run") {
      await chat(e.data.id, e.data.messages, e.data.opts);
    } else if (type === "stop") {
      stopper?.interrupt?.();
    }
  } catch (err) {
    console.error("[smollm3 worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
