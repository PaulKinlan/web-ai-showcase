// Dedicated worker that runs MediaPipe Tasks Vision landmarkers OFF the main thread.
//
// Why this exists (invariant 15 — off-main-thread reference architecture): MediaPipe's vision tasks run
// their WASM/GL inference SYNCHRONOUSLY in whatever thread calls `detect()` / `detectForVideo()`. On the
// main thread a single PoseLandmarker detect on the bundled sample blocks for ~143ms (measured) — a long
// task that janks the whole UI and makes the live webcam loop pile up. Moving the call into this worker
// leaves the main thread with only a cheap frame-grab (createImageBitmap) + skeleton draw (<8ms).
//
// SCOPED EXCEPTION to the "dedicated MODULE worker" default (invariant 15): this is a CLASSIC worker
// (WorkerClient is created with `module:false`). MediaPipe's Emscripten WASM loader
// (vision_wasm_internal.js) is pulled in with importScripts(), which ONLY exists in classic workers — a
// module worker forbids it ("Module scripts don't support importScripts()") and createFromOptions throws.
// A classic worker keeps importScripts working natively. We still speak the shared, typed/versioned
// protocol in lib/worker-protocol.js (request ids, streamed progress, AbortSignal cancel, latest-wins
// supersession, bounded queue, deterministic dispose) — it is loaded via dynamic import(), which IS
// allowed in classic workers, as is the dynamic import() of @mediapipe/tasks-vision below.
//
// The worker is GENERIC over the MediaPipe vision task class, so the same script backs PoseLandmarker AND
// HandLandmarker (the multi-model page spins up one client per task).
//
// modern-web-guidance retained + applied: performance / break-up-long-tasks / identify-inp-causes —
// "> 250ms: Offload to a Web Worker" and "separate UI updates from heavy computation": the heavy detect
// runs here; the main thread only grabs a frame and paints. No dedicated Worker/OffscreenCanvas/
// transferable guide exists in the catalogue, so the MDN primitives (Worker, Transferable ImageBitmap,
// WorkerNavigator, importScripts) are the source of truth — same posture as lib/worker-protocol.js.
//
// Transfer discipline: the main thread transfers an ImageBitmap in (ownership moved, no pixel copy); the
// worker returns only the tiny landmark arrays (33 pts × a few floats) — no need to send pixels back.

const PROTOCOL_URL = "/web-ai-showcase/lib/worker-protocol.js";
const TASKS_VISION_VERSION = "0.10.18";
const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}`;

let task = null;
let mode = "IMAGE";
let delegate = "CPU";
let lastTs = -1;

/** Map a MediaPipe result into a small, structured-clone-friendly plain object. */
function toPlain(res) {
  const mapPts = (groups) =>
    (groups || []).map((g) =>
      g.map((p) => ({
        x: p.x,
        y: p.y,
        z: p.z,
        ...(p.visibility != null ? { visibility: p.visibility } : {}),
      }))
    );
  return {
    landmarks: mapPts(res.landmarks),
    worldLandmarks: mapPts(res.worldLandmarks),
    handedness: res.handedness || res.handednesses || [],
    delegate,
  };
}

async function createTask(vision, resolver, taskClass, wantGpu, options) {
  const Task = vision[taskClass];
  if (!Task) throw new Error(`Unknown MediaPipe task: ${taskClass}`);
  const build = (del) =>
    Task.createFromOptions(resolver, {
      baseOptions: { modelAssetPath: options.modelUrl, delegate: del },
      runningMode: options.runningMode || "IMAGE",
      ...options.taskOptions,
    });
  if (wantGpu) {
    try {
      const t = await build("GPU");
      delegate = "GPU";
      return t;
    } catch (_e) {
      // GPU delegate unavailable in this worker (no WebGL/OffscreenCanvas GPU) — honest CPU fallback.
    }
  }
  const t = await build("CPU");
  delegate = "CPU";
  return t;
}

// Dynamic import of the ESM protocol (allowed in a classic worker), then serve the method table.
(async () => {
  const { serveWorker } = await import(PROTOCOL_URL);
  serveWorker({
    methods: {
      // Load the MediaPipe task in the worker. Returns the chosen delegate. Progress is coarse — MediaPipe
      // fetches the .task/WASM internally without a byte stream — so we emit initiate/ready like the shared
      // main-thread helper did (the shared cache/auto-init layer on the main thread is unchanged).
      async load(payload, { onProgress }) {
        const { taskClass, modelUrl, options = {}, preferGpu } = payload;
        onProgress?.({ status: "initiate", file: modelUrl });
        const vision = await import(CDN);
        const resolver = await vision.FilesetResolver.forVisionTasks(`${CDN}/wasm`);
        // Match the original main-thread helper's delegate signal: a REAL WebGPU adapter must exist (not
        // just navigator.gpu presence) before we prefer the GPU delegate. MediaPipe itself runs on WebGL,
        // so createTask still probes-then-falls-back to CPU if the GPU delegate can't actually initialise.
        let wantGpu = false;
        if (preferGpu && typeof navigator !== "undefined" && "gpu" in navigator) {
          try {
            wantGpu = (await navigator.gpu.requestAdapter()) != null;
          } catch {
            wantGpu = false;
          }
        }
        task = await createTask(vision, resolver, taskClass, wantGpu, {
          modelUrl,
          runningMode: options.runningMode || "IMAGE",
          taskOptions: options.taskOptions || {},
        });
        mode = options.runningMode || "IMAGE";
        onProgress?.({ status: "ready" });
        return { delegate };
      },

      // Detect on one transferred ImageBitmap. Runs synchronously here (off the main thread), then closes
      // the bitmap so its GPU/CPU memory is released deterministically.
      async detect(payload, { signal }) {
        if (!task) throw new Error("MediaPipe task not loaded");
        const { bitmap, mode: want = "IMAGE", timestamp } = payload;
        try {
          if (mode !== want) {
            await task.setOptions({ runningMode: want });
            mode = want;
          }
          if (signal.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
          let res;
          if (want === "VIDEO") {
            // MediaPipe requires strictly increasing timestamps per task instance.
            let ts = typeof timestamp === "number" ? timestamp : performance.now();
            if (ts <= lastTs) ts = lastTs + 1;
            lastTs = ts;
            res = task.detectForVideo(bitmap, ts);
          } else {
            res = task.detect(bitmap);
          }
          return toPlain(res);
        } finally {
          try {
            bitmap.close?.();
          } catch { /* already neutered */ }
        }
      },
    },
    onDispose() {
      try {
        task?.close?.();
      } catch { /* ignore */ }
      task = null;
    },
  });
})();
