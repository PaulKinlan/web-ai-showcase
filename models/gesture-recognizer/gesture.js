// Front-end helpers for the MediaPipe GestureRecognizer pages. GestureRecognizer bundles hand-landmark
// detection with a small gesture classifier, so it returns the 21 hand landmarks AND a named gesture
// (Closed_Fist, Open_Palm, Pointing_Up, Thumb_Up, Thumb_Down, Victory, ILoveYou, or None) with a score.
// We reuse the hand-landmarker drawing helpers for the skeleton and add gesture normalisation + labels.
//
// Inference runs OFF the main thread in ./worker.js (a dedicated MODULE worker) — a single main-thread
// recognize on the bundled sample was measured as a ~90ms long task, an INP/responsiveness bug for the
// live loop. createGestureWorker() boots that worker (which still downloads the .task + inits MediaPipe
// via lib/mediapipe.js, unchanged — just on the worker thread) and returns a handle whose interface
// matches the old main-thread wrapper: recognizeImage(el) / recognizeVideo(el, ts) / .delegate / close().
// The page decodes each frame to an ImageBitmap and TRANSFERS it to the worker; the worker returns the
// gestures + hand-landmark arrays and the page paints the cheap overlay.

import { WorkerClient } from "/web-ai-showcase/lib/worker-protocol.js";

export {
  drawHands,
  escapeHTML,
  HAND_CONNECTIONS,
  HAND_LANDMARK_NAMES,
  handColor,
  LANDMARK_CSS,
} from "/web-ai-showcase/models/hand-landmarker/hand.js";

const WORKER_URL = new URL("./worker.js", import.meta.url);

export const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";

// The 8 built-in gesture classes and a friendly emoji for each.
export const GESTURE_EMOJI = {
  Closed_Fist: "✊",
  Open_Palm: "✋",
  Pointing_Up: "☝️",
  Thumb_Down: "👎",
  Thumb_Up: "👍",
  Victory: "✌️",
  ILoveYou: "🤟",
  None: "·",
};

export function gestureLabel(name) {
  const pretty = (name || "None").replace(/_/g, " ");
  return `${GESTURE_EMOJI[name] ?? ""} ${pretty}`.trim();
}

/** Wraps a GestureRecognizer task so a page can recognise on a still image OR a video frame safely. */
export class GestureTask {
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
  async recognizeImage(imgEl) {
    await this._ensure("IMAGE");
    return normalize(this.task.recognize(imgEl));
  }
  async recognizeVideo(videoEl, tsMs) {
    await this._ensure("VIDEO");
    return normalize(this.task.recognizeForVideo(videoEl, tsMs));
  }
  get delegate() {
    return this.task?.__delegate || "CPU";
  }
}

/** Normalise across tasks-vision versions and expose the top gesture per hand plus its full ranking. */
function normalize(res) {
  const gestures = res.gestures || [];
  return {
    landmarks: res.landmarks || [],
    handedness: res.handedness || res.handednesses || [],
    gestures,
    // The single best gesture for the first hand (or null if no hand / no gesture).
    top: gestures[0]?.[0] ?? null,
  };
}

/**
 * Main-thread handle for the off-thread GestureRecognizer. Drop-in for the old GestureTask interface:
 * recognizeImage(el) / recognizeVideo(el, tsMs) / .delegate / close(). It decodes the given image or
 * video element to an ImageBitmap and TRANSFERS it (ownership moved, no structured clone) to the worker,
 * which recognises and returns the gestures + hand-landmark arrays. Live frames go on the latest-wins
 * "live" channel so a stale in-flight frame is superseded — the loop stays one-in-flight and responsive.
 */
export class GestureWorkerHandle {
  constructor(client) {
    this.client = client;
    this._delegate = "CPU";
  }
  get delegate() {
    return this._delegate;
  }
  async _recognize(source, wantMode, timestamp, channel) {
    // createImageBitmap decodes off the main thread and yields a Transferable frame.
    const bitmap = await createImageBitmap(source);
    const { result } = await this.client.request(
      "recognize",
      { bitmap, mode: wantMode, timestamp },
      { transfer: [bitmap], channel },
    );
    this._delegate = result.delegate || this._delegate;
    return result;
  }
  recognizeImage(imgEl) {
    return this._recognize(imgEl, "IMAGE", 0);
  }
  recognizeVideo(videoEl, tsMs) {
    return this._recognize(videoEl, "VIDEO", tsMs, "live");
  }
  /** Deterministic teardown: dispose the worker's recognizer + terminate the worker. */
  async close() {
    try {
      await this.client.terminate();
    } catch { /* ignore */ }
  }
}

/**
 * Boot the dedicated MODULE worker, load the GestureRecognizer model INSIDE it (same .task download +
 * FilesetResolver init as the main-thread path, just off-thread), and resolve a GestureWorkerHandle.
 * Wired for lib/model-loader.js: pass its onProgress; the returned handle is what onReady receives.
 */
export async function createGestureWorker({ onProgress } = {}) {
  onProgress?.({ status: "initiate", file: MODEL_URL });
  const client = new WorkerClient({
    url: WORKER_URL,
    name: "gesture",
    module: false, // CLASSIC worker: MediaPipe's FilesetResolver needs importScripts (module workers forbid it)
    maxInFlight: 1, // one recognize at a time — MediaPipe holds a single stateful recognizer
    maxQueue: 2, // bounded backpressure; live frames use latest-wins so stale frames drop, not pile up
  });
  try {
    await client.ready; // resolves once the worker has downloaded + initialised the model
  } catch (err) {
    try {
      await client.terminate();
    } catch { /* ignore */ }
    throw err;
  }
  onProgress?.({ status: "ready" });
  return new GestureWorkerHandle(client);
}
