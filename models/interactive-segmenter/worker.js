// MediaPipe MagicTouch Interactive Segmenter — genuine local inference in a dedicated worker.
// Exact artifact: float32 generation 1, verified before task creation (SHA-256 below). The worker
// owns fetch, integrity verification, inference, confidence reduction, and dense RGBA compositing.

const TASKS_VISION_VERSION = "0.10.18";
const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}`;
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/1/magic_touch.tflite";
const MODEL_SHA256 = "e24338a717c1b7ad8d159666677ef400babb7f33b8ad60c4d96db4ecf694cd25";
const CACHE_NAME = "webai-runtime-models-v1";

let vision;
let segmenter;
let delegate = "CPU";
let last = null;

// MediaPipe routes an informational XNNPACK creation line through console.error. Reclassify that
// one known non-error without suppressing genuine runtime errors.
const nativeConsoleError = console.error.bind(console);
console.error = (...args) => {
  if (String(args[0] || "").startsWith("INFO: Created TensorFlow Lite XNNPACK delegate")) {
    console.info(...args);
  } else {
    nativeConsoleError(...args);
  }
};

async function sha256Hex(bytes) {
  const hash = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(hash, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function fetchVerifiedModel(onProgress) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(MODEL_URL);
  if (cached) {
    const bytes = await cached.arrayBuffer();
    if (await sha256Hex(bytes) === MODEL_SHA256) {
      onProgress({
        status: "done",
        file: MODEL_URL,
        loaded: bytes.byteLength,
        total: bytes.byteLength,
      });
      return bytes;
    }
    await cache.delete(MODEL_URL);
  }

  onProgress({ status: "initiate", file: MODEL_URL });
  const response = await fetch(MODEL_URL, { cache: "no-store" });
  if (!response.ok) throw new Error(`MagicTouch download failed: HTTP ${response.status}`);
  const total = Number(response.headers.get("content-length")) || 6_227_884;
  const reader = response.body?.getReader();
  let loaded = 0;
  const chunks = [];
  if (reader) {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      onProgress({ status: "progress", file: MODEL_URL, loaded, total, progress: loaded / total });
    }
  } else {
    const value = new Uint8Array(await response.arrayBuffer());
    chunks.push(value);
    loaded = value.byteLength;
  }
  const joined = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    joined.set(chunk, offset);
    offset += chunk.byteLength;
  }
  const digest = await sha256Hex(joined.buffer);
  if (digest !== MODEL_SHA256) {
    throw new Error(`MagicTouch integrity mismatch: expected ${MODEL_SHA256}, got ${digest}`);
  }
  await cache.put(
    MODEL_URL,
    new Response(joined.slice().buffer, {
      headers: {
        "content-type": "application/octet-stream",
        "content-length": String(joined.byteLength),
        "x-webai-sha256": MODEL_SHA256,
      },
    }),
  );
  onProgress({ status: "done", file: MODEL_URL, loaded, total });
  return joined.buffer;
}

async function hasWebGPU() {
  if (!("gpu" in navigator)) return false;
  try {
    return (await navigator.gpu.requestAdapter()) != null;
  } catch {
    return false;
  }
}

Promise.all([
  import("/web-ai-showcase/lib/worker-protocol.js"),
  import("./mask-math.js"),
]).then(([{ serveWorker }, { composeMask, summarizeConfidence }]) => {
  async function compose(payload) {
    if (!last) throw new Error("Select a point before changing the mask view");
    const started = performance.now();
    const rgba = composeMask(last.source, last.confidence, payload);
    const summary = summarizeConfidence(
      last.confidence,
      payload.threshold ?? 0.5,
      last.point,
      last.width,
      last.height,
    );
    return {
      result: {
        width: last.width,
        height: last.height,
        rgba: rgba.buffer,
        ...summary,
        composeMs: Math.round((performance.now() - started) * 10) / 10,
      },
      transfer: [rgba.buffer],
    };
  }

  serveWorker({
    methods: {
      async configure(_payload, { onProgress }) {
        const modelBytes = await fetchVerifiedModel(onProgress);
        vision ??= await import(CDN);
        const resolver = await vision.FilesetResolver.forVisionTasks(`${CDN}/wasm`);
        delegate = (await hasWebGPU()) ? "GPU" : "CPU";
        const make = (chosen) =>
          vision.InteractiveSegmenter.createFromOptions(resolver, {
            baseOptions: { modelAssetBuffer: new Uint8Array(modelBytes), delegate: chosen },
            outputCategoryMask: true,
            outputConfidenceMasks: true,
          });
        try {
          segmenter = await make(delegate);
        } catch (error) {
          if (delegate !== "GPU") throw error;
          delegate = "CPU";
          segmenter = await make(delegate);
        }
        return {
          result: {
            delegate,
            webgpu: await hasWebGPU(),
            artifactSha256: MODEL_SHA256,
            artifactBytes: modelBytes.byteLength,
          },
        };
      },

      async segment(payload) {
        if (!segmenter) throw new Error("InteractiveSegmenter not configured");
        const { bitmap, point } = payload || {};
        if (!bitmap || !point) throw new Error("segment requires a transferred bitmap and point");
        const started = performance.now();
        try {
          const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
          const context = canvas.getContext("2d", { willReadFrequently: true });
          context.drawImage(bitmap, 0, 0);
          const source = context.getImageData(0, 0, bitmap.width, bitmap.height).data;
          const result = segmenter.segment(bitmap, {
            keypoint: {
              x: Math.min(1, Math.max(0, Number(point.x))),
              y: Math.min(1, Math.max(0, Number(point.y))),
            },
          });
          // InteractiveSegmenter exposes the selected object's probability as its single confidence
          // mask. The underlying MagicTouch tensor has background/foreground channels; the task API
          // has already selected and normalized the foreground channel for this keypoint.
          const foreground = result.confidenceMasks?.[0];
          if (!foreground) throw new Error("MagicTouch returned no foreground confidence mask");
          const confidence = foreground.getAsFloat32Array().slice();
          const width = foreground.width;
          const height = foreground.height;
          if (width !== bitmap.width || height !== bitmap.height) {
            throw new Error(
              `Unexpected mask size ${width}×${height} for ${bitmap.width}×${bitmap.height} input`,
            );
          }
          result.categoryMask?.close?.();
          for (const mask of result.confidenceMasks || []) mask.close?.();
          last = { source, confidence, width, height, point };
          const composed = await compose(payload);
          composed.result.inferenceMs = Math.round((performance.now() - started) * 10) / 10;
          composed.result.delegate = delegate;
          composed.result.outputShape = `${height}×${width} foreground confidence + category mask`;
          return composed;
        } finally {
          bitmap.close?.();
        }
      },

      compose,
    },

    onDispose() {
      try {
        segmenter?.close?.();
      } catch { /* already released */ }
      segmenter = null;
      last = null;
    },
  });
});
