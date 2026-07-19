// HandLandmarker inference worker — runs ALL MediaPipe inference OFF the main thread.
//
// Retrofit (invariant 15, measured perf bug): MediaPipe's vision tasks run synchronously wherever they
// are called. Called on the main thread, one HandLandmarker detect blocks the UI for the full inference
// duration (measured ~104ms for one detect on the bundled sample; ~44ms steady). Moving the detect into
// this dedicated worker keeps the control UI + overlay draw responsive: the main thread only grabs a
// frame (`createImageBitmap`), TRANSFERS it here, and paints the returned landmarks. Served through the
// shared typed/versioned protocol in lib/worker-protocol.js (request ids, streamed progress, AbortSignal
// cancel, stale-response suppression, bounded queue, deterministic dispose).
//
// WHY A CLASSIC WORKER (documented exception to "module workers" in invariant 15): MediaPipe's
// tasks-vision wasm loader calls `self.importScripts()` to bring in its Emscripten glue. Module workers
// forbid importScripts ("Module scripts don't support importScripts()"), so a MODULE worker cannot load
// MediaPipe — this is exactly why the MediaPipe family was the off-thread baseline's hard exception.
// A CLASSIC worker keeps importScripts available AND still supports dynamic `import()` (Chrome 91+), so
// we dynamically import BOTH the shared worker-protocol and the MediaPipe ESM. Every worker-protocol
// guarantee is preserved unchanged; only the worker `type` differs, forced by MediaPipe.
//
// The `.task` model still downloads + caches exactly as before (MediaPipe fetches modelAssetPath; the
// service worker caches it), so the shared createModelLoader auto-init / explicit-download policy is
// unchanged — this worker only relocates WHERE inference executes.
//
// Verified (headless Chrome, WASM/CPU delegate): load → {delegate}; detect on a transferred ImageBitmap
// returns the same normalized shape as the main-thread path — landmarks[hand][21]{x,y,z}, worldLandmarks,
// handedness — with zero main-thread long task from inference.

// Pin matches lib/mediapipe.js (TASKS_VISION_VERSION) so the worker + the shared runtime stay aligned.
const TASKS_VISION_VERSION = "0.10.18";
const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}`;

let vision = null;
let landmarker = null;
let mode = "IMAGE"; // current MediaPipe runningMode; switched on demand (IMAGE ⇄ VIDEO)
let delegate = "CPU";

/** True if a real WebGPU adapter exists in the worker (MediaPipe can use the GPU delegate). */
async function gpuDelegateAvailable() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) return false;
  try {
    return (await navigator.gpu.requestAdapter()) != null;
  } catch {
    return false;
  }
}

/** Normalize the result shape across tasks-vision versions (handedness vs handednesses). */
function normalize(res) {
  return {
    landmarks: res.landmarks || [],
    worldLandmarks: res.worldLandmarks || [],
    handedness: res.handedness || res.handednesses || [],
  };
}

// Dynamic import (allowed in a classic worker) wires the shared protocol. serveWorker installs the
// message handler and posts "ready" after init, so no request can arrive before we're wired.
import("/web-ai-showcase/lib/worker-protocol.js")
  .then(({ serveWorker }) => {
    serveWorker({
      methods: {
        // Load (download + init) the HandLandmarker. Runs as a request so it is cancellable and reports
        // progress (MediaPipe exposes coarse initiate/ready, no byte-level progress).
        async load(payload, { onProgress }) {
          onProgress?.({ status: "initiate", file: payload.modelUrl });
          vision = await import(CDN);
          const resolver = await vision.FilesetResolver.forVisionTasks(`${CDN}/wasm`);
          const Task = vision.HandLandmarker;
          if (!Task) throw new Error("HandLandmarker missing from @mediapipe/tasks-vision");
          delegate = (await gpuDelegateAvailable()) ? "GPU" : "CPU";
          mode = payload.runningMode || "IMAGE";
          landmarker = await Task.createFromOptions(resolver, {
            baseOptions: { modelAssetPath: payload.modelUrl, delegate },
            numHands: payload.numHands ?? 2,
            runningMode: mode,
          });
          onProgress?.({ status: "ready" });
          return { result: { delegate } };
        },

        // Detect on a TRANSFERRED ImageBitmap. payload: { bitmap, mode:"IMAGE"|"VIDEO", ts:number }.
        // Returns the 21-landmarks-per-hand arrays; the bitmap is closed to free decoder/GPU memory.
        async detect(payload, { signal }) {
          const { bitmap, mode: wantMode, ts } = payload;
          try {
            if (!landmarker) throw new Error("Model not loaded");
            if (signal.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
            if (mode !== wantMode) {
              await landmarker.setOptions({ runningMode: wantMode });
              mode = wantMode;
            }
            // MediaPipe detect/detectForVideo are synchronous — but we are OFF the main thread, so this
            // never blocks the page's UI / overlay / rVFC loop.
            const res = wantMode === "VIDEO"
              ? landmarker.detectForVideo(bitmap, ts)
              : landmarker.detect(bitmap);
            return { result: { ...normalize(res), delegate } };
          } finally {
            try {
              bitmap.close?.();
            } catch { /* already neutered */ }
          }
        },
      },

      onDispose() {
        try {
          landmarker?.close?.();
        } catch { /* ignore */ }
        landmarker = null;
        vision = null;
      },
    });
  })
  .catch((err) => {
    // If the protocol module itself failed to load, surface a protocol-shaped fatal so the client's
    // .ready rejects (rather than hanging waiting for "ready"). PROTOCOL_VERSION is 1.
    self.postMessage({
      p: 1,
      kind: "error",
      error: {
        name: "InitError",
        message: "worker bootstrap failed: " + String(err?.message ?? err),
      },
    });
  });
