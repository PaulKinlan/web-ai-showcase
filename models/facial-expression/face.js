// Front-end helpers for the facial-expression pages. Keeps each page thin: it owns the worker
// handshake, turns files / sample images / webcam frames into data URLs, runs a live webcam preview
// (nothing is uploaded or stored — a frame is only classified on demand), renders the emotion
// probability bars, and formats the "see inside" confidence numbers. All inference lives in worker.js
// (off the main thread). Privacy by construction: the image never leaves the device.

const WORKER_URL = "/web-ai-showcase/models/facial-expression/worker.js";

export class FaceEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this._loadWaiters = [];
    this._pending = new Map();
    this._id = 0;
    this.onProgress = null;
    this.device = "wasm";
    this.worker.addEventListener("message", (e) => this._onMessage(e.data));
    this.worker.addEventListener("error", (e) => {
      const err = new Error(e.message || "Worker failed to start");
      for (const w of this._loadWaiters) w.reject(err);
      this._loadWaiters = [];
      for (const [, p] of this._pending) p.reject(err);
      this._pending.clear();
    });
  }

  _onMessage(msg) {
    if (msg.type === "progress") {
      this.onProgress?.(msg.p);
    } else if (msg.type === "ready") {
      this.ready = true;
      this.device = msg.device;
      for (const w of this._loadWaiters) w.resolve(msg.device);
      this._loadWaiters = [];
    } else if (msg.type === "result") {
      const p = this._pending.get(msg.id);
      if (p) {
        this._pending.delete(msg.id);
        p.resolve(msg);
      }
    } else if (msg.type === "error") {
      if (msg.id != null && this._pending.has(msg.id)) {
        this._pending.get(msg.id).reject(new Error(msg.message));
        this._pending.delete(msg.id);
      } else {
        const err = new Error(msg.message);
        for (const w of this._loadWaiters) w.reject(err);
        this._loadWaiters = [];
      }
    }
  }

  load(onProgress) {
    if (onProgress) this.onProgress = onProgress;
    if (this.ready) return Promise.resolve(this.device);
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load" });
    });
  }

  /** Classify a face image (data URL). Returns { all, entropy, margin, numClasses, ms, device }. */
  classify(imageURL) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, image: imageURL });
    });
  }
}

/** Read a File (upload or drop) into a data URL usable by the worker. */
export function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}

/**
 * A minimal webcam helper. start() attaches the live stream to a <video>; grab() captures the CURRENT
 * frame into a data URL (square, centre-cropped) WITHOUT ever uploading or persisting it. Honest about
 * a missing camera / denied permission. Only the frames you explicitly classify are ever read.
 */
export class Webcam {
  constructor(video) {
    this.video = video;
    this.stream = null;
  }
  static supported() {
    return !!navigator.mediaDevices?.getUserMedia;
  }
  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    this.video.srcObject = this.stream;
    await this.video.play().catch(() => {});
  }
  /** Capture the current frame, centre-cropped square, as a data URL. Nothing leaves the device. */
  grab(size = 320) {
    const vw = this.video.videoWidth, vh = this.video.videoHeight;
    if (!vw || !vh) return null;
    const side = Math.min(vw, vh);
    const sx = (vw - side) / 2, sy = (vh - side) / 2;
    const c = document.createElement("canvas");
    c.width = size;
    c.height = size;
    const ctx = c.getContext("2d");
    ctx.drawImage(this.video, sx, sy, side, side, 0, 0, size, size);
    return c.toDataURL("image/jpeg", 0.9);
  }
  running() {
    return !!this.stream;
  }
  stop() {
    try {
      this.stream?.getTracks().forEach((t) => t.stop());
    } catch { /* ignore */ }
    this.stream = null;
    this.video.srcObject = null;
  }
}

// One emoji + short gloss per FER emotion, for a legible affective read-out.
export const EMOTION_META = {
  happy: { emoji: "😄", gloss: "joy / smiling" },
  sad: { emoji: "😢", gloss: "sadness" },
  angry: { emoji: "😠", gloss: "anger" },
  surprise: { emoji: "😲", gloss: "surprise" },
  fear: { emoji: "😨", gloss: "fear" },
  disgust: { emoji: "🤢", gloss: "disgust" },
  neutral: { emoji: "😐", gloss: "neutral" },
};

export function emotionEmoji(label) {
  return EMOTION_META[label?.toLowerCase?.()]?.emoji ?? "🙂";
}

/**
 * Render emotion probability bars into `container`. `items` = [{label, prob}] sorted high→low.
 * The top emotion is highlighted. Each bar is an accessible meter.
 */
export function renderEmotionBars(container, items) {
  const sorted = [...items].sort((a, b) => b.prob - a.prob);
  const top = sorted[0]?.label;
  container.replaceChildren(
    ...sorted.map((it) => {
      const row = document.createElement("div");
      row.className = "emo-row" + (it.label === top ? " emo-top" : "");
      const pct = (it.prob * 100).toFixed(1);
      const emoji = emotionEmoji(it.label);
      row.innerHTML = `
        <div class="emo-head">
          <span class="emo-label"><span class="emo-emoji" aria-hidden="true">${emoji}</span>${
        escapeHTML(it.label)
      }</span>
          <span class="emo-val">${pct}%</span>
        </div>
        <div class="emo-track" role="meter" aria-valuemin="0" aria-valuemax="100"
             aria-valuenow="${pct}" aria-label="${escapeHTML(it.label)}: ${pct} percent">
          <div class="emo-fill" style="inline-size:${pct}%"></div>
        </div>`;
      return row;
    }),
  );
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Shared inline styles for the facial-expression widgets. Injected once per page. */
export const FACE_CSS = `
.dropzone {
  border: 2px dashed var(--border-strong); border-radius: var(--radius);
  background: var(--bg-raised); padding: 1rem; text-align: center; cursor: pointer;
  transition: border-color .15s, background .15s;
}
.dropzone.drag { border-color: var(--accent); background: var(--bg-secondary); }
.dropzone:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.stage-grid { display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start; }
.stage-col { flex: 1 1 260px; min-inline-size: 0; }
.preview-wrap { position: relative; inline-size: 100%; aspect-ratio: 1 / 1; background: var(--bg-raised);
  border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }
.preview-img, .cam-video {
  inline-size: 100%; block-size: 100%; object-fit: cover; display: block;
}
.cam-video { transform: scaleX(-1); } /* mirror the live preview so it reads like a mirror */
.privacy-note { font-size: .78rem; color: var(--muted); margin: .35rem 0 0; display: flex; gap: .3rem; align-items: center; }
.sample-strip { display: flex; gap: .5rem; flex-wrap: wrap; margin: .5rem 0; }
.sample-thumb {
  inline-size: 60px; block-size: 60px; object-fit: cover; border-radius: 8px;
  border: 2px solid transparent; cursor: pointer; padding: 0;
}
.sample-thumb.active { border-color: var(--accent); }
.sample-thumb:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.emo-bars { display: flex; flex-direction: column; gap: .5rem; margin-top: .3rem; }
.emo-head { display: flex; justify-content: space-between; gap: .5rem; font-size: .85rem; }
.emo-label { font-family: var(--font-body); text-transform: capitalize; display: flex; gap: .35rem; align-items: center; }
.emo-emoji { font-size: 1rem; }
.emo-val { font-family: var(--font-mono); color: var(--muted); white-space: nowrap; }
.emo-track {
  block-size: .7rem; background: var(--bg-raised); border: 1px solid var(--border);
  border-radius: 999px; overflow: hidden; margin-top: .15rem;
}
.emo-fill { block-size: 100%; background: var(--muted); border-radius: 999px; transition: inline-size .35s ease; }
.emo-top .emo-fill { background: var(--accent); }
.emo-top .emo-label { font-weight: 600; }
.field-row { display: flex; flex-wrap: wrap; gap: .6rem; align-items: center; margin: .6rem 0; }
.readout {
  display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem;
}
.readout b { color: var(--color); font-weight: 600; }
.big-verdict { font-size: 1.5rem; font-weight: 600; display: flex; gap: .5rem; align-items: center; margin: .2rem 0; }
.big-verdict .ev-emoji { font-size: 2rem; }
.chip {
  font: inherit; font-size: .78rem; padding: .3rem .7rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer;
}
.chip:hover { border-color: var(--accent); }
.chip[aria-pressed="true"] { border-color: var(--accent); background: var(--bg-secondary); }
.inside-table { inline-size: 100%; border-collapse: collapse; font-size: .82rem; margin-top: .5rem; }
.inside-table th, .inside-table td {
  text-align: left; padding: .3rem .5rem; border-bottom: 1px solid var(--border);
  font-family: var(--font-mono);
}
.inside-table th { color: var(--muted); font-weight: 600; }
.conf-meter { block-size: .9rem; border-radius: 999px; overflow: hidden; border: 1px solid var(--border);
  background: linear-gradient(to right, var(--bad), var(--warn), var(--good)); position: relative; margin: .3rem 0; }
.conf-needle { position: absolute; top: -3px; bottom: -3px; inline-size: 3px; background: var(--color); border-radius: 2px; }
.fallback { border: 1px solid var(--warn); border-radius: var(--radius); background: var(--bg-raised); padding: .8rem; margin: .5rem 0; font-size: .85rem; }
`;
