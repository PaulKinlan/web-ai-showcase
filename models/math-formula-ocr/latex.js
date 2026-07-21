// Front-end helpers for the Math-formula OCR page. Keeps the page thin: owns the worker handshake, turns an
// uploaded/pasted/sample image into a data URL, and exposes the recognised LaTeX. All inference (Donut-Swin
// encoder + mBART decoder generating LaTeX) lives in worker.js, off the main thread.

const WORKER_URL = "/web-ai-showcase/models/math-formula-ocr/worker.js";

export class LatexEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.onProgress = null;
    this.onToken = null;
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
    } else if (msg.type === "token") {
      this.onToken?.(msg);
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
  /** Recognise the LaTeX of an equation image (data/URL string). Returns { latex, ms, device, tokens }. */
  recognise(imageURL, maxTokens) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "recognise", id, imageURL, maxTokens });
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

/** texify wraps output in $$…$$ (or \[…\]). Strip the display-math delimiters for a clean, copyable body. */
export function cleanLatex(s) {
  let t = String(s).trim();
  t = t.replace(/^\$\$([\s\S]*?)\$\$$/, "$1");
  t = t.replace(/^\\\[([\s\S]*?)\\\]$/, "$1");
  t = t.replace(/^\$([\s\S]*?)\$$/, "$1");
  return t.trim();
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

export const LATEX_CSS = `
.mf-drop { border: 2px dashed var(--border); border-radius: 12px; padding: 1.1rem; text-align: center;
  background: var(--bg-raised); transition: border-color .15s, background .15s; }
.mf-drop.drag { border-color: var(--accent); background: color-mix(in srgb, var(--accent) 10%, transparent); }
.mf-tools { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; justify-content: center; margin: .3rem 0; }
.mf-btn { font: inherit; font-size: .85rem; padding: .35rem .8rem; border-radius: 8px; border: 1px solid var(--border);
  background: var(--bg-raised); color: var(--color); cursor: pointer; }
.mf-btn:hover, .mf-btn:focus-visible { border-color: var(--accent); }
.mf-btn[disabled] { opacity: .5; cursor: default; }
.mf-hint { font-size: .82rem; color: var(--muted); margin: .3rem 0; }
.mf-samples { display: flex; flex-wrap: wrap; gap: .5rem; justify-content: center; margin-top: .5rem; }
.mf-sample { border: 1px solid var(--border); border-radius: 8px; background: #fff; padding: .2rem; cursor: pointer; }
.mf-sample img { display: block; height: 42px; width: auto; }
.mf-sample:hover, .mf-sample:focus-visible { border-color: var(--accent); }
.mf-preview { margin: .8rem 0 .3rem; text-align: center; }
.mf-preview img { max-width: 100%; max-height: 180px; background: #fff; border: 1px solid var(--border); border-radius: 8px; padding: .4rem; }
.mf-outwrap { margin-top: .6rem; }
.mf-outlabel { font-size: .78rem; color: var(--muted); font-family: var(--font-mono, monospace); margin-bottom: .3rem; display: flex; justify-content: space-between; align-items: center; gap: .5rem; }
.mf-latex { font-family: var(--font-mono, monospace); font-size: .95rem; white-space: pre-wrap; word-break: break-word;
  background: var(--bg-raised); border: 1px solid var(--border); border-radius: 8px; padding: .7rem .8rem; min-block-size: 2.6rem; }
.mf-rendered { margin-top: .6rem; text-align: center; min-height: 2.4rem; overflow-x: auto; }
.mf-copy { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 7px; border: 1px solid var(--border);
  background: var(--bg-raised); color: var(--color); cursor: pointer; }
.mf-copy:hover { border-color: var(--accent); }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono, monospace);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
`;
