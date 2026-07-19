// Front-end helpers for the age-estimation pages. Keeps each page thin: it owns the worker handshake,
// turns files / sample images / webcam frames into data URLs, runs a live webcam preview (nothing is
// uploaded or stored — a frame is only classified on demand), renders the age-bucket probability bars,
// and formats the "see inside" confidence numbers. All inference lives in worker.js (off the main
// thread). Privacy by construction: the image never leaves the device.
//
// HONESTY NOTE baked into the helpers: this predicts *apparent* age from pixels. It is an imperfect,
// bias-carrying estimate — a soft signal, never identity, verification, or a birth date.

const WORKER_URL = "/web-ai-showcase/models/age-estimation/worker.js";

export class AgeEngine {
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

  /** Estimate age for an image (data URL). Returns { all, top, expectedAge, entropy, margin, ms }. */
  estimate(imageURL) {
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
 * frame into a data URL (square, centre-cropped) WITHOUT ever uploading or persisting it. Honest about a
 * missing camera / denied permission. Only the frames you explicitly classify are ever read.
 * (modern-web-guidance: export-html-media-from-canvas — capture via canvas drawImage + toDataURL.)
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

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** A short human gloss for a life stage, keyed by the winning bucket — purely descriptive, not a verdict. */
export function bucketGloss(label) {
  return ({
    "0-2": "infant",
    "3-9": "child",
    "10-19": "teen",
    "20-29": "twenties",
    "30-39": "thirties",
    "40-49": "forties",
    "50-59": "fifties",
    "60-69": "sixties",
    "more than 70": "70+",
  })[label] ?? "";
}

/**
 * Render age-bucket probability bars into `container`. `items` = [{label, prob}] in AGE ORDER (young →
 * old) so the bars read like a distribution across the lifespan. The most-likely bucket is highlighted.
 * Each bar is an accessible meter.
 */
export function renderAgeBars(container, items) {
  const top = [...items].sort((a, b) => b.prob - a.prob)[0]?.label;
  container.replaceChildren(
    ...items.map((it) => {
      const row = document.createElement("div");
      row.className = "age-row" + (it.label === top ? " age-top" : "");
      const pct = (it.prob * 100).toFixed(1);
      row.innerHTML = `
        <div class="age-head">
          <span class="age-label">${escapeHTML(it.label)}<span class="age-gloss">${
        escapeHTML(bucketGloss(it.label))
      }</span></span>
          <span class="age-val">${pct}%</span>
        </div>
        <div class="age-track" role="meter" aria-valuemin="0" aria-valuemax="100"
             aria-valuenow="${pct}" aria-label="Age ${escapeHTML(it.label)}: ${pct} percent">
          <div class="age-fill" style="inline-size:${pct}%"></div>
        </div>`;
      return row;
    }),
  );
}

/** Shared inline styles for the age-estimation widgets. Injected once per page. */
export const AGE_CSS = `
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
.preview-img, .cam-video { inline-size: 100%; block-size: 100%; object-fit: cover; display: block; }
.cam-video { transform: scaleX(-1); }
.privacy-note { font-size: .78rem; color: var(--muted); margin: .35rem 0 0; display: flex; gap: .3rem; align-items: center; }
.sample-strip { display: flex; gap: .5rem; flex-wrap: wrap; margin: .5rem 0; }
.sample-thumb {
  inline-size: 60px; block-size: 60px; object-fit: cover; border-radius: 8px;
  border: 2px solid transparent; cursor: pointer; padding: 0;
}
.sample-thumb.active { border-color: var(--accent); }
.sample-thumb:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.age-bars { display: flex; flex-direction: column; gap: .4rem; margin-top: .3rem; }
.age-head { display: flex; justify-content: space-between; gap: .5rem; font-size: .82rem; }
.age-label { font-family: var(--font-mono); display: flex; gap: .4rem; align-items: baseline; }
.age-gloss { color: var(--muted); font-size: .72rem; font-family: var(--font-body); }
.age-val { font-family: var(--font-mono); color: var(--muted); white-space: nowrap; }
.age-track {
  block-size: .65rem; background: var(--bg-raised); border: 1px solid var(--border);
  border-radius: 999px; overflow: hidden; margin-top: .1rem;
}
.age-fill { block-size: 100%; background: var(--muted); border-radius: 999px; transition: inline-size .35s ease; }
.age-top .age-fill { background: var(--accent); }
.age-top .age-label { font-weight: 600; color: var(--color); }
.field-row { display: flex; flex-wrap: wrap; gap: .6rem; align-items: center; margin: .6rem 0; }
.readout {
  display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem;
}
.readout b { color: var(--color); font-weight: 600; }
.big-verdict { display: flex; flex-direction: column; gap: .1rem; margin: .2rem 0 .4rem; }
.big-verdict .est { font-size: 2rem; font-weight: 600; font-family: var(--font-display); line-height: 1.1; }
.big-verdict .sub { font-size: .82rem; color: var(--muted); }
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
.caveat { border: 1px solid var(--warn); border-radius: var(--radius); background: var(--bg-raised);
  padding: .7rem .9rem; margin: .6rem 0; font-size: .85rem; }
`;
