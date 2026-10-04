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
 *
 * web-ai-showcase-p12: gemma-3's own mlc-chat-config declares sliding_window_size: 512 (its
 * interleaved sliding-window attention) while the prebuilt record overrides
 * context_window_size: 4096 — so the merged record has BOTH positive and the pinned engine
 * (@mlc-ai/web-llm@0.2.85) refuses it before any generation:
 * "WindowSizeConfigurationError: Only one of context_window_size and sliding_window_size can
 * be positive". The engine merges record.overrides LAST (they win over the model's config),
 * and its disabled sentinel is -1 — a 0 still throws — so the prebuilt record is re-supplied
 * with the sliding window disabled: the full 4096-token context is attended, no sliding
 * window. The engine's model universe is the passed model_list, so the entry must carry the
 * WHOLE prebuilt record (weights URL, model_lib, vram), not a partial one. Other models keep
 * the stock path.
 *
 * @param {object} opts
 * @param {string} opts.model     MLC model id
 * @param {(p:{text?:string,progress?:number})=>void} [opts.onProgress]
 * @returns {Promise<import("@mlc-ai/web-llm").MLCEngineInterface>}
 */
export async function createEngine({ model, onProgress }) {
  const webllm = await import(WEBLLM_URL);
  const init = {
    initProgressCallback: (r) => onProgress?.({ text: r.text, progress: r.progress }),
  };
  if (/gemma-?3/i.test(model)) {
    const prebuilt = webllm.prebuiltAppConfig?.model_list?.find?.((r) => r.model_id === model);
    if (prebuilt) {
      init.appConfig = {
        model_list: [
          { ...prebuilt, overrides: { ...prebuilt.overrides, sliding_window_size: -1 } },
        ],
      };
    }
  }
  return webllm.CreateMLCEngine(model, init);
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
