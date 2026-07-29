// XLS-R binary acoustic-label worker. Inference is real and off-main-thread. Every request uses the
// shared typed/versioned protocol, and every waveform is deterministically crop/padded to exactly
// five seconds (80,000 mono samples at 16 kHz), matching the canonical model-card example.
import { loadPipeline } from "/web-ai-showcase/lib/webai.js";
import { serveWorker, yieldToMain } from "/web-ai-showcase/lib/worker-protocol.js";

const RUNTIME_REVISION = "6bea1eddcfca9842add425123f4955d5b4f153f7";
const CANONICAL_REVISION = "7a28165f33e1dbb37adbce09c0a9afcd6095dd4d";
const Q8_FILE = "onnx/model_quantized.onnx";
const Q8_BYTES = 318834205;
const SR = 16000;
const MODEL_SAMPLES = SR * 5;
let pipe = null;
let device = "wasm";
let observedQ8Request = null;

// Validate what the library ACTUALLY requests. This wrapper does not construct model URLs; it
// observes Transformers.js fetches, rejects mutable `/main` requests for this repo, and validates
// the q8 response's declared byte length before the runtime can consume it.
const nativeFetch = self.fetch.bind(self);
self.fetch = async (input, init) => {
  const requestedUrl = String(input instanceof Request ? input.url : input);
  const isThisModel = requestedUrl.includes(
    "Xenova/wav2vec2-large-xlsr-53-gender-recognition-librispeech/resolve/",
  );
  if (isThisModel && !requestedUrl.includes(`/resolve/${RUNTIME_REVISION}/`)) {
    throw new Error(`Refusing unpinned model request: ${requestedUrl}`);
  }
  const response = await nativeFetch(input, init);
  if (isThisModel && requestedUrl.endsWith(`/${Q8_FILE}`)) {
    const contentLength = Number(
      response.headers.get("content-length") || response.headers.get("x-linked-size") || 0,
    );
    if (!response.ok) throw new Error(`Pinned q8 request failed with HTTP ${response.status}`);
    if (contentLength !== Q8_BYTES) {
      throw new Error(`Pinned q8 byte length ${contentLength}; expected ${Q8_BYTES}`);
    }
    observedQ8Request = {
      requestedUrl,
      responseUrl: response.url,
      status: response.status,
      contentLength,
    };
  }
  return response;
};

async function webgpuUsable() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) return false;
  try {
    return !!(await navigator.gpu.requestAdapter());
  } catch {
    return false;
  }
}

async function ensureLoaded(preferred, onProgress, signal) {
  if (pipe) return;
  if (signal?.aborted) throw signal.reason;
  const want = preferred || ((await webgpuUsable()) ? "webgpu" : "wasm");
  const load = (backend) =>
    loadPipeline({
      task: "audio-classification",
      model: "Xenova/wav2vec2-large-xlsr-53-gender-recognition-librispeech",
      backend,
      dtype: "q8",
      revision: "6bea1eddcfca9842add425123f4955d5b4f153f7",
      onProgress,
    });
  try {
    const loaded = await load(want);
    pipe = loaded.pipe;
    device = loaded.device;
  } catch (error) {
    if (want === "wasm") throw error;
    onProgress({ status: "initiate", file: "WebGPU failed; retrying on WASM" });
    const loaded = await load("wasm");
    pipe = loaded.pipe;
    device = loaded.device;
  }
  if (signal?.aborted) throw signal.reason;
}

function fiveSecondInput(source) {
  const input = new Float32Array(MODEL_SAMPLES);
  const copiedSamples = Math.min(source.length, MODEL_SAMPLES);
  input.set(source.subarray(0, copiedSamples));
  return {
    input,
    preprocessing: {
      contract: "crop-or-zero-pad-to-5.000s",
      sourceSamples: source.length,
      copiedSamples,
      paddedSamples: MODEL_SAMPLES - copiedSamples,
      croppedSamples: Math.max(0, source.length - MODEL_SAMPLES),
      modelSamples: MODEL_SAMPLES,
      sampleRate: SR,
    },
  };
}

async function classify(payload, { signal, onProgress }) {
  await ensureLoaded(payload?.device, onProgress, signal);
  if (signal.aborted) throw signal.reason;
  const source = new Float32Array(payload.audio);
  const { input, preprocessing } = fiveSecondInput(source);
  await yieldToMain();
  if (signal.aborted) throw signal.reason;
  const started = performance.now();
  const features = await pipe.processor(input);
  if (signal.aborted) throw signal.reason;
  const output = await pipe.model(features);
  if (signal.aborted) throw signal.reason;
  const logits = Array.from(output.logits.data);
  if (logits.length !== 2 || logits.some((value) => !Number.isFinite(value))) {
    throw new Error(`Unexpected logits: expected [1,2], received ${logits.length}`);
  }
  const id2label = pipe.model.config.id2label || { 0: "female", 1: "male" };
  const max = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - max));
  const total = exps[0] + exps[1];
  const labels = logits
    .map((logit, index) => ({
      id: index,
      label: id2label[index] ?? `LABEL_${index}`,
      logit,
      score: exps[index] / total,
    }))
    .sort((a, b) => b.score - a.score);
  return {
    labels,
    logits,
    margin: Math.abs(labels[0].score - labels[1].score),
    ms: Math.round(performance.now() - started),
    device,
    durationS: MODEL_SAMPLES / SR,
    preprocessing,
    runtimeEvidence: {
      model: "Xenova/wav2vec2-large-xlsr-53-gender-recognition-librispeech",
      revision: RUNTIME_REVISION,
      canonicalRevision: CANONICAL_REVISION,
      q8File: Q8_FILE,
      q8Bytes: Q8_BYTES,
      observedQ8Request,
    },
  };
}

serveWorker({
  methods: {
    async load(payload, context) {
      await ensureLoaded(payload?.device, context.onProgress, context.signal);
      return {
        device,
        runtimeRevision: RUNTIME_REVISION,
        canonicalRevision: CANONICAL_REVISION,
        q8File: Q8_FILE,
        q8Bytes: Q8_BYTES,
        observedQ8Request,
      };
    },
    classify,
  },
  async onDispose() {
    const current = pipe;
    pipe = null;
    if (typeof current?.dispose === "function") await current.dispose();
    else if (typeof current?.model?.dispose === "function") await current.model.dispose();
    if (typeof current?.processor?.dispose === "function") await current.processor.dispose();
  },
});
