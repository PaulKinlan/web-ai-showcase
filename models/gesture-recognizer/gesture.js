// Front-end helpers for the MediaPipe GestureRecognizer pages. GestureRecognizer bundles hand-landmark
// detection with a small gesture classifier, so it returns the 21 hand landmarks AND a named gesture
// (Closed_Fist, Open_Palm, Pointing_Up, Thumb_Up, Thumb_Down, Victory, ILoveYou, or None) with a score.
// We reuse the hand-landmarker drawing helpers for the skeleton and add gesture normalisation + labels.
// The model loads through lib/model-loader.js via lib/mediapipe.js createVisionTask.

export {
  drawHands,
  escapeHTML,
  HAND_CONNECTIONS,
  HAND_LANDMARK_NAMES,
  handColor,
  LANDMARK_CSS,
} from "/web-ai-showcase/models/hand-landmarker/hand.js";

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
