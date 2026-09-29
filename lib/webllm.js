// Shared WebLLM helper for larger chat LLMs (Llama / Qwen / Phi) that run via MLC's WebGPU engine.
// WebLLM is WebGPU-ONLY (no WASM fallback) and downloads large weights — pages MUST gate on a real
// GPU adapter and show an honest needs-WebGPU state, never a faked reply. Complements lib/webai.js
// (Transformers.js). Use this when a model's catalogue entry has "runtime":"webllm".

export const WEBLLM_URL = "https://esm.run/@mlc-ai/web-llm@0.2.85";

/** WebLLM needs a real WebGPU adapter (navigator.gpu alone is not enough — headless returns null). */
export async function webGPUAdapterAvailable() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) return false;
  try {
    return (await navigator.gpu.requestAdapter()) != null;
  } catch {
    return false;
  }
}

/**
 * Create an MLC engine for a WebLLM model id (e.g. "Llama-3.2-1B-Instruct-q4f16_1-MLC").
 * @param {object} opts
 * @param {string} opts.model     MLC model id
 * @param {(p:{text?:string,progress?:number})=>void} [opts.onProgress]
 * @returns {Promise<import("@mlc-ai/web-llm").MLCEngineInterface>}
 */
export async function createEngine({ model, onProgress }) {
  const webllm = await import(WEBLLM_URL);
  return webllm.CreateMLCEngine(model, {
    initProgressCallback: (r) => onProgress?.({ text: r.text, progress: r.progress }),
  });
}

/**
 * Stream a chat completion token-by-token. Returns the full text; calls onToken(delta) as it streams
 * and onUsage(usage) once the completion reports its real usage.
 *
 * The deltas are VISIBLE CHUNKS, not tokens (web-ai-showcase-0ly): a count built from them is a
 * chunk count wearing the word "tokens". WebLLM 0.2.85 is OpenAI-compatible and emits a final
 * chunk carrying `usage` when `stream_options.include_usage` is requested, so callers report the
 * completion's OWN token count and keep the delta count only for the live rate.
 *
 * @param {import("@mlc-ai/web-llm").MLCEngineInterface} engine
 * @param {{messages: Array<{role:string,content:string}>, temperature?:number, top_p?:number, max_tokens?:number}} req
 * @param {(delta:string)=>void} [onToken]
 * @param {(usage:{completion_tokens?:number, prompt_tokens?:number})=>void} [onUsage]
 */
export async function streamChat(engine, req, onToken, onUsage) {
  let full = "";
  const chunks = await engine.chat.completions.create({
    ...req,
    stream: true,
    stream_options: { include_usage: true },
  });
  for await (const chunk of chunks) {
    const delta = chunk.choices?.[0]?.delta?.content ?? "";
    if (delta) {
      full += delta;
      onToken?.(delta);
    }
    if (chunk.usage && typeof chunk.usage.completion_tokens === "number") onUsage?.(chunk.usage);
  }
  return full;
}
