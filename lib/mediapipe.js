// Shared MediaPipe Tasks (Web) runtime helper — Google's model-backed vision landmarkers/segmenters
// (hand / pose / face landmarks, gesture, image/interactive segmentation, object/face detection) that
// run in the browser via @mediapipe/tasks-vision. These are real downloadable `.task` models. Runs on
// the GPU delegate when available, CPU otherwise. Complements lib/webai.js (Transformers.js) and
// lib/webllm.js (WebLLM). Use when a catalogue entry has "runtime":"mediapipe".

export const TASKS_VISION_VERSION = "0.10.18";
const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}`;

let filesetPromise = null;
function fileset(vision) {
  filesetPromise ??= vision.FilesetResolver.forVisionTasks(`${CDN}/wasm`);
  return filesetPromise;
}

/** True if a real WebGPU adapter exists (MediaPipe can use the GPU delegate). CPU fallback otherwise. */
export async function gpuDelegateAvailable() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) return false;
  try {
    return (await navigator.gpu.requestAdapter()) != null;
  } catch {
    return false;
  }
}

/**
 * Create a MediaPipe vision task.
 * @param {object} o
 * @param {string} o.taskClass  e.g. "HandLandmarker", "PoseLandmarker", "FaceLandmarker",
 *                              "GestureRecognizer", "ImageSegmenter", "ObjectDetector", "FaceDetector"
 * @param {string} o.modelUrl   the .task model asset URL (canonical Google storage URL)
 * @param {object} [o.options]  task-specific options (numHands, runningMode, etc.)
 * @param {(p:{status:string,progress?:number})=>void} [o.onProgress]
 * @returns {Promise<any>} the created task (call .detect()/.recognize()/.segment(), and .close() to free)
 */
export async function createVisionTask({ taskClass, modelUrl, options = {}, onProgress }) {
  onProgress?.({ status: "initiate", file: modelUrl });
  const vision = await import(CDN);
  const resolver = await fileset(vision);
  const Task = vision[taskClass];
  if (!Task) throw new Error(`Unknown MediaPipe task: ${taskClass}`);
  const delegate = (await gpuDelegateAvailable()) ? "GPU" : "CPU";
  const task = await Task.createFromOptions(resolver, {
    baseOptions: { modelAssetPath: modelUrl, delegate },
    runningMode: options.runningMode || "IMAGE",
    ...options,
  });
  onProgress?.({ status: "ready" });
  task.__delegate = delegate;
  return task;
}
