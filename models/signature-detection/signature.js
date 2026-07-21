// Front-end helpers for the Signature detection page. Owns the worker handshake, turns an uploaded/sample
// image into a data/URL, and draws detection boxes over it. All inference (YOLOS object detection) lives in
// worker.js, off the main thread.

const WORKER_URL = "/web-ai-showcase/models/signature-detection/worker.js";

export class SignatureEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.onProgress = null;
    this._loadWaiters = [];
    this._pending = new Map();
    this._id = 0;
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
    if (msg.type === "progress") this.onProgress?.(msg.p);
    else if (msg.type === "ready") {
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
        for (const w of this._loadWaiters) w.reject(new Error(msg.message));
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
  detect(imageURL, threshold = 0.3) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "detect", id, imageURL, threshold });
    });
  }
}

export function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result);
    fr.onerror = reject;
    fr.readAsDataURL(file);
  });
}

/** Overlay detection boxes (in ORIGINAL pixel coords) on the displayed image, scaled to its rendered size. */
export function drawBoxes(overlay, img, dets) {
  const scaleX = img.clientWidth / (img.naturalWidth || 1);
  const scaleY = img.clientHeight / (img.naturalHeight || 1);
  overlay.style.width = img.clientWidth + "px";
  overlay.style.height = img.clientHeight + "px";
  overlay.replaceChildren(...dets.map((d) => {
    const b = document.createElement("div");
    b.className = "sig-box";
    b.style.left = (d.xmin * scaleX) + "px";
    b.style.top = (d.ymin * scaleY) + "px";
    b.style.width = ((d.xmax - d.xmin) * scaleX) + "px";
    b.style.height = ((d.ymax - d.ymin) * scaleY) + "px";
    const tag = document.createElement("span");
    tag.className = "sig-tag";
    tag.textContent = `signature ${(d.score * 100).toFixed(0)}%`;
    b.append(tag);
    return b;
  }));
}

export const SIGNATURE_CSS = `
.sig-drop { border: 2px dashed var(--border); border-radius: 12px; padding: 1.1rem; text-align: center;
  background: var(--bg-raised); transition: border-color .15s, background .15s; }
.sig-drop.drag { border-color: var(--accent); background: color-mix(in srgb, var(--accent) 10%, transparent); }
.sig-tools { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; justify-content: center; margin: .3rem 0; }
.sig-btn { font: inherit; font-size: .85rem; padding: .35rem .8rem; border-radius: 8px; border: 1px solid var(--border);
  background: var(--bg-raised); color: var(--color); cursor: pointer; }
.sig-btn:hover:not([disabled]), .sig-btn:focus-visible { border-color: var(--accent); }
.sig-btn[disabled] { opacity: .5; cursor: default; }
.sig-hint { font-size: .82rem; color: var(--muted); margin: .3rem 0; }
.sig-samples { display: flex; flex-wrap: wrap; gap: .5rem; justify-content: center; margin-top: .5rem; }
.sig-sample { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; padding: 0; cursor: pointer; background: #fff; line-height: 0; }
.sig-sample img { display: block; height: 56px; width: auto; }
.sig-sample:hover, .sig-sample:focus-visible { border-color: var(--accent); }
.sig-stage { position: relative; display: inline-block; margin-top: .8rem; max-width: 100%; }
.sig-stage img { display: block; max-width: 100%; height: auto; border: 1px solid var(--border); border-radius: 8px; }
.sig-overlay { position: absolute; inset: 0; pointer-events: none; }
.sig-box { position: absolute; border: 2px solid #e0348b; border-radius: 4px; box-shadow: 0 0 0 1px rgba(255,255,255,.5) inset; }
.sig-tag { position: absolute; top: -1.35rem; left: -2px; font-family: var(--font-mono, monospace); font-size: .68rem;
  font-weight: 700; background: #e0348b; color: #fff; padding: .05rem .35rem; border-radius: 4px; white-space: nowrap; }
.sig-verdict { font-size: 1.15rem; font-weight: 700; margin: .8rem 0 .3rem; min-height: 1.5rem; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono, monospace);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
`;
