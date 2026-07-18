// Shared Web-AI helpers for every model page. Keep model pages thin: import from here.
// Uses @huggingface/transformers (Transformers.js v3) from jsDelivr. WebGPU with WASM fallback.

export const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5";

/** navigator.gpu merely EXISTING is not enough — headless/adapter-less browsers stall transformers.js
 *  when told device:"webgpu". Probe for a real adapter. */
export async function hasWebGPU() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) return false;
  try {
    return (await navigator.gpu.requestAdapter()) != null;
  } catch {
    return false;
  }
}

/** Pick the device: prefer the declared backend, but only use webgpu when a real adapter exists. */
export async function pickDevice(preferred) {
  if (preferred === "webgpu" && (await hasWebGPU())) return "webgpu";
  return "wasm";
}

/**
 * Load a Transformers.js pipeline with progress + honest errors.
 * @param {object} opts
 * @param {string} opts.task      e.g. "automatic-speech-recognition"
 * @param {string} opts.model     HF id, e.g. "onnx-community/whisper-base"
 * @param {"webgpu"|"wasm"} [opts.backend]
 * @param {string} [opts.dtype]   e.g. "q4f16", "fp16", "q8"
 * @param {(p:{status:string,progress?:number,file?:string})=>void} [opts.onProgress]
 * @returns {Promise<{pipe:Function, device:string}>}
 */
export async function loadPipeline({ task, model, backend, dtype, onProgress }) {
  const { pipeline, env } = await import(TRANSFORMERS_URL);
  // Let the library own its Cache Storage; do not fight the service worker.
  env.allowLocalModels = false;
  const device = await pickDevice(backend);
  const pipe = await pipeline(task, model, {
    device,
    ...(dtype ? { dtype } : {}),
    progress_callback: (p) => onProgress?.(p),
  });
  return { pipe, device };
}

/** Render load progress into a <progress> + status line. Errors on the page, never only console. */
export function progressReporter(progressEl, statusEl) {
  const files = new Map();
  return {
    onProgress(p) {
      if (p.status === "progress" && p.file) {
        files.set(p.file, p.progress ?? 0);
        const avg = [...files.values()].reduce((a, b) => a + b, 0) / files.size;
        if (progressEl) {
          progressEl.value = avg;
          progressEl.max = 100;
        }
        if (statusEl) {
          statusEl.textContent = `Downloading ${files.size} file(s)… ${avg.toFixed(0)}%`;
        }
      } else if (p.status === "ready" || p.status === "done") {
        if (statusEl) statusEl.textContent = "Model ready.";
      } else if (p.status === "initiate") {
        if (statusEl) statusEl.textContent = `Fetching ${p.file ?? "model"}…`;
      }
    },
    fail(err) {
      if (statusEl) {
        statusEl.textContent = `Couldn't load the model: ${err?.message ?? err}`;
        statusEl.classList.add("err");
      }
    },
  };
}

/** Small timing helper so pages can show real latency, not a claim. */
export async function timed(fn) {
  const t0 = performance.now();
  const result = await fn();
  return { result, ms: Math.round(performance.now() - t0) };
}
