// Dedicated worker that runs MediaPipe FaceLandmarker inference OFF the main thread.
//
// Why: measured baseline — one FaceLandmarker.detect() on the bundled sample was a ~73ms main-thread
// long task (478 landmarks + 52 blendshapes on the CPU/XNNPACK delegate). On a live webcam loop that
// blocked every frame on the main thread and spiked INP. This worker moves the inference off-thread so
// the control UI only paints (mesh overlay + blendshape bars from returned arrays — cheap).
//
// It speaks the shared typed/versioned protocol in lib/worker-protocol.js (request ids, transfer not
// clone, AbortSignal cancel, stale-response suppression, bounded queue, lifecycle + deterministic
// dispose). The model is created here with the SAME options the page passes (numFaces,
// outputFaceBlendshapes, outputFacialTransformationMatrixes, runningMode) so behaviour is identical.
//
// CLASSIC worker (not a module worker) — a MEASURED, evidence-backed exception to invariant 15's
// "module workers" default: @mediapipe/tasks-vision's WASM loader (FilesetResolver) calls
// `importScripts`, which a `{type:"module"}` worker rejects ("Module scripts don't support
// importScripts()"). This is the same platform reality that made the 5 MediaPipe demos the hard
// exceptions in scratchpad/worker-compliance-baseline.md. A classic worker still uses dynamic
// `import()` for the ES modules below (protocol + tasks-vision) AND permits MediaPipe's importScripts.
//
// modern-web-guidance retained: identify-inp-causes + performance (INP / main-thread unblocking —
// long work → offload to a Web Worker), break-up-long-tasks (the offload channel).

// Pinned to the same tasks-vision version the rest of the site uses (lib/mediapipe.js).
const TASKS_VISION_VERSION = "0.10.18";
const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}`;

let vision = null;
let landmarker = null;
let delegate = "CPU";
let runningMode = "IMAGE";

/** Honest GPU-delegate probe: navigator.gpu existing isn't enough — request a real adapter. */
async function gpuAvailable() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) return false;
  try {
    return (await navigator.gpu.requestAdapter()) != null;
  } catch {
    return false;
  }
}

/** Copy MediaPipe's result objects into plain, structured-cloneable data the page's helpers expect. */
function normalize(res) {
  return {
    landmarks: (res.faceLandmarks || []).map((pts) => pts.map((p) => ({ x: p.x, y: p.y, z: p.z }))),
    blendshapes: (res.faceBlendshapes || []).map((b) => ({
      categories: (b.categories || []).map((c) => ({
        index: c.index,
        score: c.score,
        categoryName: c.categoryName,
        displayName: c.displayName,
      })),
    })),
    matrixes: (res.facialTransformationMatrixes || []).map((m) => ({
      rows: m.rows,
      columns: m.columns,
      data: Array.from(m.data || []),
    })),
  };
}

// Dynamic import() works in a classic worker (it's the module loader, not importScripts). We wire the
// shared protocol only after it resolves; WorkerClient waits for the "ready" message serveWorker posts
// before it sends any request, so there's no lost-message race.
import("/web-ai-showcase/lib/worker-protocol.js").then(({ serveWorker }) => {
  serveWorker({
    methods: {
      // Create the FaceLandmarker with the page's exact options. Runs the ~4MB .task + wasm download
      // (still cached by the SW / browser exactly as before — same URLs, same origin, worker-visible).
      async configure(payload) {
        const { modelUrl, options = {} } = payload || {};
        if (!modelUrl) throw new Error("configure: modelUrl required");
        vision ??= await import(CDN);
        const resolver = await vision.FilesetResolver.forVisionTasks(`${CDN}/wasm`);
        delegate = (await gpuAvailable()) ? "GPU" : "CPU";
        runningMode = options.runningMode || "IMAGE";
        landmarker?.close?.();
        landmarker = await vision.FaceLandmarker.createFromOptions(resolver, {
          baseOptions: { modelAssetPath: modelUrl, delegate },
          runningMode,
          ...options,
        });
        return { result: { delegate } };
      },

      // Detect on a transferred ImageBitmap. IMAGE mode for stills; VIDEO mode (detectForVideo, needs a
      // monotonic timestamp) for the live loop. The bitmap is closed after use so it can't leak.
      async detect(payload) {
        const { bitmap, mode, tsMs } = payload || {};
        if (!landmarker) throw new Error("FaceLandmarker not configured");
        try {
          if (mode && mode !== runningMode) {
            await landmarker.setOptions({ runningMode: mode });
            runningMode = mode;
          }
          const res = runningMode === "VIDEO"
            ? landmarker.detectForVideo(bitmap, tsMs)
            : landmarker.detect(bitmap);
          return { result: normalize(res) };
        } finally {
          bitmap?.close?.();
        }
      },
    },

    onDispose() {
      try {
        landmarker?.close?.();
      } catch { /* ignore */ }
      landmarker = null;
    },
  });
});
