// Qwen3 1.7B (WebLLM) worker — hybrid-thinking chat generation off the main thread via MLC's WebGPU
// engine. Runtime is WebLLM, not Transformers.js: engine creation goes through the shared helper in
// lib/webllm.js (createEngine) so every WebLLM page shares one verified load path. WebLLM is
// WebGPU-ONLY — the page gates on webGPUAdapterAvailable() before it ever asks us to load.
// Model: Qwen3-1.7B-q4f16_1-MLC (Alibaba Qwen, 1.7B params, instruction-tuned, hybrid reasoning).
//
// The streaming loop is Qwen3-specific (the shared streamChat only forwards answer content): Qwen3
// emits a <think>…</think> reasoning chain before the answer in thinking mode. We separate that
// reasoning stream from the answer stream so the UI can surface the model's private reasoning trace.
// Two sources are handled: WebLLM may expose reasoning as delta.reasoning_content OR inline as
// <think>…</think> in delta.content — the splitter below handles both and tags every delta.

import { createEngine } from "/web-ai-showcase/lib/webllm.js";

const MODEL_ID = "Qwen3-1.7B-q4f16_1-MLC";
let engine = null;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (engine) return;
  engine = await createEngine({
    model: MODEL_ID,
    onProgress: (p) => post({ type: "progress", p }),
  });
  post({ type: "ready" });
}

// Split a growing content string into { reasoning, answer } around a <think>…</think> block. Called on
// the full accumulated content each chunk; callers diff against what was already emitted. While the
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

async function run(id, req) {
  await ensureLoaded();
  const { messages, temperature, top_p, max_tokens } = req;
  const t0 = performance.now();
  let ttft = null;
  let firstFired = false;
  let chunks = 0, reasoningChunks = 0;

  // Answer-content diffing (handles inline <think> tags).
  let fullContent = "";
  let emittedReasoning = 0, emittedAnswer = 0;
  // Explicit reasoning stream (WebLLM reasoning_content), if the build exposes it.
  let explicitReasoning = "";
  let sawExplicitReasoning = false;

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

  const stream = await engine.chat.completions.create({
    messages,
    temperature,
    top_p,
    max_tokens,
    stream: true,
  });

  for await (const chunk of stream) {
    const d = chunk.choices?.[0]?.delta ?? {};
    // 1) Explicit reasoning channel (Qwen3 via WebLLM reasoning parser), streamed straight through.
    if (typeof d.reasoning_content === "string" && d.reasoning_content) {
      sawExplicitReasoning = true;
      explicitReasoning += d.reasoning_content;
      fireFirst();
      emit("reasoning", d.reasoning_content);
    }
    // 2) Answer content — may contain inline <think>…</think> when no explicit channel is used.
    const c = d.content ?? "";
    if (c) {
      fullContent += c;
      fireFirst();
      if (sawExplicitReasoning) {
        // Reasoning already came via its own channel — content is pure answer.
        const add = fullContent.slice(emittedAnswer);
        emittedAnswer = fullContent.length;
        emit("answer", add);
      } else {
        const { reasoning, answer, closed } = splitThink(fullContent);
        const holdback = closed ? reasoning.length : Math.max(0, reasoning.length - 8);
        if (holdback > emittedReasoning) {
          emit("reasoning", reasoning.slice(emittedReasoning, holdback));
          emittedReasoning = holdback;
        }
        if (answer.length > emittedAnswer) {
          emit("answer", answer.slice(emittedAnswer));
          emittedAnswer = answer.length;
        }
      }
    }
  }

  // Flush any held-back reasoning tail (final split is authoritative).
  if (!sawExplicitReasoning) {
    const { reasoning, answer } = splitThink(fullContent);
    if (reasoning.length > emittedReasoning) {
      emit("reasoning", reasoning.slice(emittedReasoning));
      emittedReasoning = reasoning.length;
    }
    if (answer.length > emittedAnswer) {
      emit("answer", answer.slice(emittedAnswer));
      emittedAnswer = answer.length;
    }
  }

  const finalSplit = sawExplicitReasoning
    ? { reasoning: explicitReasoning, answer: fullContent }
    : splitThink(fullContent);
  const ms = Math.round(performance.now() - t0);
  let stats = null;
  try {
    if (typeof engine.runtimeStatsText === "function") stats = await engine.runtimeStatsText();
  } catch {
    stats = null;
  }
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
    stats,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded();
    } else if (type === "run") {
      await run(e.data.id, e.data.req);
    } else if (type === "stop") {
      if (engine && typeof engine.interruptGenerate === "function") engine.interruptGenerate();
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
