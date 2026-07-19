// Dedicated worker: runs Google's MediaPipe GestureRecognizer OFF the main thread.
//
// Why: a single main-thread recognize on the bundled sample was measured as a ~90ms long task
// (cold WASM-graph run; warm runs ~30ms) — a real INP/responsiveness bug for the live-webcam loop.
// modern-web-guidance retained + applied:
//   • performance (INP & Main-Thread Unblocking) — "Offload to a Web Worker" for heavy compute; the
//     main thread now only decodes a frame to an ImageBitmap, transfers it, and paints the overlay.
//   • break-up-long-tasks / identify-inp-causes — the blocking work is moved off the UI thread and the
//     retrofit was MEASURED with a longtask PerformanceObserver, not inferred.
//
// CLASSIC worker (not module) — a DELIBERATE, evidence-backed exception to the module-worker default:
// @mediapipe/tasks-vision's FilesetResolver loads its WASM runtime via importScripts(), which a MODULE
// worker forbids ("Module scripts don't support importScripts()"). A module-worker probe fails on every
// tasks-vision build tried (0.10.18/0.10.20 throw importScripts; 0.10.22 fails to load); a classic
// worker hosts it correctly. lib/worker-protocol.js supports exactly this via WorkerClient {module:false}
// ("Fall back to classic only if asked"). Classic workers can still dynamic-import() ES modules, so the
// protocol + MediaPipe + the shared loader are pulled in below — no importScripts of our own code.
//
// The .task download + FilesetResolver init is UNCHANGED — it reuses lib/mediapipe.js createVisionTask
// (same CDN, version, delegate probe, and auto-init contract); only the THREAD it runs on moved. The
// transport is lib/worker-protocol.js serveWorker: typed/versioned envelope, per-request ids, transferred
// ImageBitmaps (not cloned), cooperative AbortSignal cancel, bounded queue, and deterministic teardown.

let task = null;
let mode = "IMAGE"; // createVisionTask defaults to IMAGE running mode

(async () => {
  let serveWorker;
  try {
    ({ serveWorker } = await import("/web-ai-showcase/lib/worker-protocol.js"));
    const { createVisionTask } = await import("/web-ai-showcase/lib/mediapipe.js");
    const { MODEL_URL } = await import("./gesture.js");

    serveWorker({
      // Runs once before "ready" is posted: downloads the ~8 MB .task bundle + inits the GPU/CPU delegate.
      async init() {
        task = await createVisionTask({
          taskClass: "GestureRecognizer",
          modelUrl: MODEL_URL,
          options: { numHands: 1, runningMode: "IMAGE" },
        });
        mode = "IMAGE";
      },
      methods: {
        // Recognize one transferred ImageBitmap. mode "IMAGE" (still) or "VIDEO" (live frame, monotonic ts).
        async recognize(payload, { signal }) {
          const { bitmap, mode: want, timestamp } = payload;
          try {
            if (signal.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
            if (!task) throw new Error("recognizer not initialised");
            if (mode !== want) {
              await task.setOptions({ runningMode: want });
              mode = want;
            }
            const res = want === "VIDEO"
              ? task.recognizeForVideo(bitmap, timestamp ?? performance.now())
              : task.recognize(bitmap);
            return { result: normalize(res, task.__delegate || "CPU") };
          } finally {
            // Always release the transferred frame's GPU/CPU memory, even on error/abort.
            bitmap?.close?.();
          }
        },
      },
      // Free the recognizer (and its WASM/graph) on terminate().
      onDispose() {
        try {
          task?.close?.();
        } catch { /* ignore */ }
        task = null;
      },
    });
  } catch (err) {
    // Bootstrap (module import) failed — surface a fatal protocol error so WorkerClient.ready rejects
    // and the loader shows a real error + Retry (never a hang). p===PROTOCOL_VERSION (1).
    self.postMessage({
      p: 1,
      kind: "error",
      error: { name: err?.name || "InitError", message: String(err?.message ?? err) },
    });
  }
})();

// Flatten the MediaPipe result into transfer-free, structured-cloneable arrays for the main thread.
function normalize(res, delegate) {
  const gestures = res.gestures || [];
  return {
    landmarks: res.landmarks || [],
    handedness: res.handedness || res.handednesses || [],
    gestures,
    top: gestures[0]?.[0] ?? null,
    delegate,
  };
}
