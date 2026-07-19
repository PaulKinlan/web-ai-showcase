// Front-end helpers for the MediaPipe FaceDetector pages. FaceDetector (BlazeFace short-range) finds
// faces fast and returns, per face, a pixel bounding box, 6 keypoints (eyes, nose tip, mouth, ears) and
// a confidence score. Like the other MediaPipe helpers this wraps one task instance, switches
// IMAGE/VIDEO running mode on demand, and draws boxes + keypoints onto a canvas. It also includes a
// privacy blur helper. The model loads through lib/model-loader.js via lib/mediapipe.js createVisionTask.

export { escapeHTML, LANDMARK_CSS } from "/web-ai-showcase/models/hand-landmarker/hand.js";
import { SupersededError, WorkerClient } from "/web-ai-showcase/lib/worker-protocol.js";

// BlazeFace short-range ships as a .tflite asset (the .task variant 404s on Google storage).
export const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";

// The 6 keypoints BlazeFace returns, in order.
export const KEYPOINT_NAMES = [
  "Right eye",
  "Left eye",
  "Nose tip",
  "Mouth",
  "Right ear",
  "Left ear",
];

const BOX_COLORS = ["#1565c0", "#b3261e", "#1a7a3a", "#6a1b9a", "#e8760a", "#00838f"];
export function boxColor(i) {
  return BOX_COLORS[i % BOX_COLORS.length];
}

/** Wraps a FaceDetector task so a page can detect on a still image OR a video frame safely. */
export class FaceDetectorTask {
  constructor(task) {
    this.task = task;
    this.mode = task?.runningMode || "IMAGE";
  }
  async _ensure(mode) {
    if (this.mode !== mode) {
      await this.task.setOptions({ runningMode: mode });
      this.mode = mode;
    }
  }
  async detectImage(imgEl) {
    await this._ensure("IMAGE");
    return normalize(this.task.detect(imgEl));
  }
  async detectVideo(videoEl, tsMs) {
    await this._ensure("VIDEO");
    return normalize(this.task.detectForVideo(videoEl, tsMs));
  }
  get delegate() {
    return this.task?.__delegate || "CPU";
  }
}

function normalize(res) {
  return { detections: res.detections || [] };
}

// ── Off-main-thread detector ─────────────────────────────────────────────────
// A drop-in for FaceDetectorTask whose detect() runs in models/face-detector/worker.js, so the
// synchronous BlazeFace inference no longer blocks the main thread (invariant 15). The main thread
// only snapshots the frame (createImageBitmap), TRANSFERS it, and later draws boxes (cheap).
//
// Interface parity with FaceDetectorTask: `detectImage(source)` / `detectVideo(source, tsMs)` →
// { detections }, plus a `delegate` getter. `close()` disposes the worker + its model.

/** Re-export so pages can ignore expected latest-wins drops in a live loop. */
export { SupersededError };

export class WorkerFaceDetector {
  /** @param {WorkerClient} client */
  constructor(client) {
    this.client = client;
    this._delegate = "CPU";
  }
  get delegate() {
    return this._delegate;
  }
  async _detect(source, mode, tsMs, opts = {}) {
    // Snapshot the current frame off the live element and hand ownership to the worker (transfer, not
    // clone). createImageBitmap is async + cheap; the heavy detect happens in the worker.
    const bitmap = await createImageBitmap(source);
    const { result } = await this.client.request(
      "detect",
      { bitmap, mode, tsMs },
      { transfer: [bitmap], channel: opts.channel, signal: opts.signal },
    );
    if (result.delegate) this._delegate = result.delegate;
    return { detections: result.detections || [], ms: result.ms };
  }
  /** Detect on a still image (IMAGE running mode). */
  detectImage(source) {
    return this._detect(source, "IMAGE", 0);
  }
  /** Detect on a video frame (VIDEO running mode). channel:"live" supersedes stale in-flight frames. */
  detectVideo(source, tsMs, opts = {}) {
    return this._detect(source, "VIDEO", tsMs ?? performance.now(), { channel: "live", ...opts });
  }
  /** Deterministic teardown — dispose the worker's model + terminate it. */
  async close() {
    await this.client.terminate();
  }
}

/**
 * Spin up the FaceDetector worker and resolve to a ready WorkerFaceDetector. Used as the model-loader
 * `load` fn so the shared auto-init / explicit-download UX (invariant 12) is UNCHANGED — the worker
 * just replaces the main-thread task instance.
 * @param {(p:{status:string})=>void} [onProgress]
 * @param {(s:string)=>void} [onState]
 * @returns {Promise<WorkerFaceDetector>}
 */
export async function loadFaceDetectorWorker(onProgress, onState) {
  onProgress?.({ status: "initiate", file: MODEL_URL });
  const client = new WorkerClient({
    url: new URL("./worker.js", import.meta.url),
    name: "face-detector",
    module: false, // classic worker: MediaPipe's FilesetResolver needs importScripts (see worker.js).
    maxInFlight: 1,
    maxQueue: 1, // one frame in flight, one waiting; newer live frames supersede the waiter.
    onState,
  });
  try {
    await client.ready; // resolves once the worker has loaded the WASM + model and posted "ready".
    const det = new WorkerFaceDetector(client);
    try {
      const { result } = await client.request("info");
      if (result?.delegate) det._delegate = result.delegate;
    } catch { /* delegate stays default until first detect fills it in */ }
    onProgress?.({ status: "ready" });
    return det;
  } catch (err) {
    await client.terminate().catch(() => {});
    throw err;
  }
}

/** Read a detection's confidence score (categories[0].score), 0..1. */
export function detScore(det) {
  return det?.categories?.[0]?.score ?? 0;
}

function drawSource(canvas, source, mirror) {
  const w = source.naturalWidth || source.videoWidth || source.width;
  const h = source.naturalHeight || source.videoHeight || source.height;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (mirror) {
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(source, 0, 0, w, h);
    ctx.restore();
  } else {
    ctx.drawImage(source, 0, 0, w, h);
  }
  return { w, h, ctx };
}

/**
 * Draw the image/video frame + detected face boxes and keypoints onto `canvas` at natural size.
 * Bounding boxes come back in PIXELS; keypoints are normalized 0..1.
 */
export function drawDetections(canvas, source, result, opts = {}) {
  const { w, h, ctx } = drawSource(canvas, source, opts.mirror);
  const mx = (x) => (opts.mirror ? w - x : x);
  const lw = Math.max(2, w / 320), r = Math.max(2.5, w / 260);
  (result.detections || []).forEach((det, i) => {
    const color = boxColor(i);
    const b = det.boundingBox;
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    // Mirror the box by flipping its x origin around the width.
    const x0 = opts.mirror ? w - (b.originX + b.width) : b.originX;
    ctx.strokeRect(x0, b.originY, b.width, b.height);
    // Confidence label chip
    const label = `${(detScore(det) * 100).toFixed(0)}%`;
    ctx.font = `${Math.max(12, w / 45)}px sans-serif`;
    const tw = ctx.measureText(label).width + 8;
    ctx.fillStyle = color;
    ctx.fillRect(x0, Math.max(0, b.originY - Math.max(16, w / 34)), tw, Math.max(16, w / 34));
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x0 + 4, Math.max(12, b.originY - Math.max(4, w / 130)));
    // Keypoints (normalized → pixels)
    ctx.fillStyle = color;
    for (const kp of det.keypoints || []) {
      ctx.beginPath();
      ctx.arc(mx(kp.x * w), kp.y * h, r, 0, Math.PI * 2);
      ctx.fillStyle = "#fff";
      ctx.fill();
      ctx.lineWidth = lw;
      ctx.strokeStyle = color;
      ctx.stroke();
    }
  });
  return result;
}

/**
 * Draw the source with every detected face region blurred out (privacy). Returns the canvas so callers
 * can export it. Blur strength scales with the box size; a light box outline marks each redaction.
 */
export function drawBlurredFaces(canvas, source, result, opts = {}) {
  const { w, h, ctx } = drawSource(canvas, source, opts.mirror);
  for (const det of result.detections || []) {
    const b = det.boundingBox;
    const x0 = opts.mirror ? w - (b.originX + b.width) : b.originX;
    // Pad the region a little so hair/jaw are covered too.
    const pad = Math.round(Math.max(b.width, b.height) * 0.18);
    const rx = Math.max(0, x0 - pad), ry = Math.max(0, b.originY - pad);
    const rw = Math.min(w - rx, b.width + pad * 2), rh = Math.min(h - ry, b.height + pad * 2);
    ctx.save();
    ctx.beginPath();
    ctx.rect(rx, ry, rw, rh);
    ctx.clip();
    ctx.filter = `blur(${Math.max(6, Math.round(rw / 6))}px)`;
    // Re-draw the whole source through the clip+blur so only this region is blurred.
    if (opts.mirror) {
      ctx.translate(w, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(source, 0, 0, w, h);
    } else {
      ctx.drawImage(source, 0, 0, w, h);
    }
    ctx.restore();
  }
  return canvas;
}
