// Dedicated worker that runs MediaPipe ImageSegmenter inference OFF the main thread.
//
// Same architecture as the 5 built MediaPipe demos (face-landmarker et al. — the measured hard
// exceptions to the module-worker default): CLASSIC worker because @mediapipe/tasks-vision's WASM
// loader (FilesetResolver) calls importScripts(); dynamic import() still loads the shared protocol
// and the tasks-vision ES module. The page transfers an ImageBitmap in; the worker runs segment()
// and returns the category mask (Uint8Array, one class index per pixel at the input's size), a
// per-class pixel histogram, per-class mean confidence, and the model's real label map.
//
// Model: Google's official deeplab_v3.tflite (PASCAL VOC 2012, 21 classes) — probe-verified
// 2026-07-27 in headless Chrome to return a real multi-class category mask + 21 confidence masks.

// Pinned to the same tasks-vision version the rest of the site uses (lib/mediapipe.js).
const TASKS_VISION_VERSION = "0.10.18";
const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}`;

let vision = null;
let segmenter = null;
let delegate = "CPU";

/** Honest GPU-delegate probe: navigator.gpu existing isn't enough — request a real adapter. */
async function gpuAvailable() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) return false;
  try {
    return (await navigator.gpu.requestAdapter()) != null;
  } catch {
    return false;
  }
}

import("/web-ai-showcase/lib/worker-protocol.js").then(({ serveWorker }) => {
  serveWorker({
    methods: {
      // Create the ImageSegmenter with both masks requested. The ~2.8MB .tflite + wasm download is
      // cached by the browser/SW exactly as on the other MediaPipe demos (same URLs, same origin).
      async configure(payload) {
        const { modelUrl, options = {} } = payload || {};
        if (!modelUrl) throw new Error("configure: modelUrl required");
        vision ??= await import(CDN);
        const resolver = await vision.FilesetResolver.forVisionTasks(`${CDN}/wasm`);
        delegate = (await gpuAvailable()) ? "GPU" : "CPU";
        segmenter?.close?.();
        segmenter = await vision.ImageSegmenter.createFromOptions(resolver, {
          baseOptions: { modelAssetPath: modelUrl, delegate },
          runningMode: "IMAGE",
          outputCategoryMask: true,
          outputConfidenceMasks: true,
          ...options,
        });
        // The model's real label map, from the bundle itself (e.g. 21 PASCAL VOC classes).
        const labels = (typeof segmenter.getLabels === "function" ? segmenter.getLabels() : []).map(
          (l) => l.categoryName || l.displayName || String(l),
        );
        return { result: { delegate, labels } };
      },

      // Segment one transferred ImageBitmap (IMAGE mode). The bitmap is closed after use.
      async segment(payload) {
        const { bitmap } = payload || {};
        if (!segmenter) throw new Error("ImageSegmenter not configured");
        try {
          const res = segmenter.segment(bitmap);
          if (!res.categoryMask) throw new Error("model returned no category mask");
          const mask = res.categoryMask.getAsUint8Array();
          const width = res.categoryMask.width;
          const height = res.categoryMask.height;
          const nLabels = (res.confidenceMasks || []).length || 21;
          const hist = new Uint32Array(nLabels);
          for (let i = 0; i < mask.length; i++) hist[mask[i]]++;
          // Mean confidence per present class, from the model's float confidence masks.
          const conf = new Array(nLabels).fill(null);
          const sums = new Float64Array(nLabels);
          const counts = new Uint32Array(nLabels);
          const confData = (res.confidenceMasks || []).map((m) => m.getAsFloat32Array());
          for (let i = 0; i < mask.length; i++) {
            const c = mask[i];
            if (confData[c]) {
              sums[c] += confData[c][i];
              counts[c]++;
            }
          }
          for (let c = 0; c < nLabels; c++) {
            if (counts[c]) conf[c] = Math.round((sums[c] / counts[c]) * 1000) / 1000;
          }
          res.categoryMask.close?.();
          for (const m of res.confidenceMasks || []) m.close?.();
          return { result: { width, height, mask, hist: Array.from(hist), conf } };
        } finally {
          bitmap?.close?.();
        }
      },
    },

    onDispose() {
      try {
        segmenter?.close?.();
      } catch { /* ignore */ }
      segmenter = null;
    },
  });
});
