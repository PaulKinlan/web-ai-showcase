// FaceDetector inference worker — moves MediaPipe BlazeFace detection OFF the main thread.
//
// Why a CLASSIC worker (not the module worker invariant 15 prefers): MediaPipe's
// `FilesetResolver.forVisionTasks()` loads the vision WASM glue via `importScripts()`, and
// `importScripts` is FORBIDDEN in module workers ("Module scripts don't support importScripts()").
// Measured: a `{type:"module"}` worker throws exactly that at `FaceDetector.createFromOptions`. A
// classic worker keeps `importScripts` available for MediaPipe AND still supports dynamic `import()`,
// so we load BOTH the ESM worker-protocol helper and the `@mediapipe/tasks-vision` ESM through
// `import()`. This is the documented-constraint exception to the module-worker default, scoped to this
// one worker; the shared `lib/worker-protocol.js` still governs the protocol (WorkerClient uses
// `module:false` to talk to it).
//
// Contract: main creates an ImageBitmap of the frame, TRANSFERS it here, we run detect()/detectForVideo
// and transfer nothing back but a small plain-object detection list (boxes in pixels, 6 keypoints
// normalized, a confidence score) plus the delegate + inference ms. Drawing stays on the main thread
// (cheap: draw the source + stroke boxes). Backpressure/staleness/cancellation come from the protocol.
//
// modern-web-guidance retained + applied:
//   • performance — "INP & Main-Thread Unblocking": DO "separate UI updates from heavy computations,
//     then push background processing to a Web Worker." Inference (the heavy part) now runs here; the
//     main thread only paints. The synchronous BlazeFace detect that used to block every rAF/video
//     frame no longer touches the main thread.
//   • schedule-tasks-by-priority — the main thread keeps its frame budget for input + paint.

const CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";
// BlazeFace short-range ships as a .tflite asset (the .task variant 404s on Google storage).
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";

let detector = null;
let delegate = "CPU";
let mode = "IMAGE"; // MediaPipe running mode; switched on demand like the main-thread wrapper did.

/** Real WebGPU adapter probe (workers expose navigator.gpu) — MediaPipe uses the GPU delegate if so. */
async function gpuDelegateAvailable() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) return false;
  try {
    return (await navigator.gpu.requestAdapter()) != null;
  } catch {
    return false;
  }
}

async function ensureMode(want) {
  if (mode !== want) {
    await detector.setOptions({ runningMode: want });
    mode = want;
  }
}

// Map MediaPipe Detection objects to plain, structured-cloneable objects in exactly the shape the
// main-thread drawer (facedet.js drawDetections / detScore) already expects.
function serializeDetections(res) {
  const out = [];
  for (const d of res?.detections || []) {
    const b = d.boundingBox || {};
    out.push({
      boundingBox: {
        originX: b.originX,
        originY: b.originY,
        width: b.width,
        height: b.height,
      },
      keypoints: (d.keypoints || []).map((k) => ({ x: k.x, y: k.y })),
      categories: [{ score: d.categories?.[0]?.score ?? 0 }],
    });
  }
  return out;
}

(async () => {
  // Dynamic import works in a classic worker; it loads the ESM protocol helper + the MediaPipe ESM.
  const { serveWorker } = await import("/web-ai-showcase/lib/worker-protocol.js");
  const vision = await import(CDN);

  serveWorker({
    // Runs before "ready" is posted: fetch the WASM fileset + the ~230 KB model and build the detector.
    async init() {
      const resolver = await vision.FilesetResolver.forVisionTasks(`${CDN}/wasm`);
      delegate = (await gpuDelegateAvailable()) ? "GPU" : "CPU";
      detector = await vision.FaceDetector.createFromOptions(resolver, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate },
        runningMode: "IMAGE",
      });
    },
    methods: {
      // Cheap metadata request the main side calls once after ready (delegate for the readout).
      // deno-lint-ignore require-await
      async info() {
        return { delegate };
      },
      // Detect on one transferred frame. payload:{ bitmap:ImageBitmap, mode:"IMAGE"|"VIDEO", tsMs }.
      async detect(payload, { signal }) {
        const bitmap = payload.bitmap;
        try {
          await ensureMode(payload.mode === "VIDEO" ? "VIDEO" : "IMAGE");
          if (signal.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
          const t0 = performance.now();
          const res = mode === "VIDEO"
            ? detector.detectForVideo(bitmap, payload.tsMs ?? performance.now())
            : detector.detect(bitmap);
          const ms = Math.round(performance.now() - t0);
          return { result: { detections: serializeDetections(res), delegate, ms } };
        } finally {
          // Always release the transferred frame's memory in the worker.
          bitmap.close?.();
        }
      },
    },
    onDispose() {
      try {
        detector?.close?.();
      } catch { /* ignore */ }
      detector = null;
    },
  });
})();
