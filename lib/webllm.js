// Shared WebLLM helper for larger chat LLMs (Llama / Qwen / Phi) that run via MLC's WebGPU engine.
// WebLLM is WebGPU-ONLY (no WASM fallback) and downloads large weights — pages MUST gate on a real
// GPU adapter and show an honest needs-WebGPU state, never a faked reply. Complements lib/webai.js
// (Transformers.js). Use this when a model's catalogue entry has "runtime":"webllm".

export const WEBLLM_URL = "https://esm.run/@mlc-ai/web-llm";

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
 * Stream a chat completion token-by-token. Returns the full text; calls onToken(delta) as it streams.
 * @param {import("@mlc-ai/web-llm").MLCEngineInterface} engine
 * @param {{messages: Array<{role:string,content:string}>, temperature?:number, top_p?:number, max_tokens?:number}} req
 * @param {(delta:string)=>void} [onToken]
 */
export async function streamChat(engine, req, onToken) {
  let full = "";
  const chunks = await engine.chat.completions.create({ ...req, stream: true });
  for await (const chunk of chunks) {
    const delta = chunk.choices?.[0]?.delta?.content ?? "";
    if (delta) {
      full += delta;
      onToken?.(delta);
    }
  }
  return full;
}
