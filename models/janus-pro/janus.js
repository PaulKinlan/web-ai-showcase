// Front-end helpers for the Janus-Pro pages: the worker handshake, streaming plumbing for BOTH the
// understanding (text stream) and generation (image-token stream + final raster) paths, image
// helpers, and the widget CSS. All inference is in worker.js (off the main thread).

const WORKER_URL = "/web-ai-showcase/models/janus-pro/worker.js";

export class JanusEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.onProgress = null;
    this._loadWaiters = [];
    this._probeWaiters = [];
    this._active = null; // { id, kind, handlers..., resolve, reject }
    this._id = 0;
    this.worker.addEventListener("message", (e) => this._onMessage(e.data));
    this.worker.addEventListener("error", (e) => {
      this._rejectAll(new Error(e.message || "Worker failed to start"));
    });
  }

  _rejectAll(err) {
    for (const w of this._loadWaiters) w.reject(err);
    this._loadWaiters = [];
    if (this._active) {
      this._active.reject(err);
      this._active = null;
    }
  }

  _onMessage(msg) {
    const a = this._active;
    switch (msg.type) {
      case "progress":
        this.onProgress?.(msg.p);
        break;
      case "probe-result":
        for (const w of this._probeWaiters) w.resolve(msg.gpu);
        this._probeWaiters = [];
        break;
      case "ready":
        this.ready = true;
        for (const w of this._loadWaiters) w.resolve(msg.device);
        this._loadWaiters = [];
        break;
      case "prompt":
        if (a && a.id === msg.id) a.onPrompt?.(msg.template);
        break;
      case "token":
        if (a && a.id === msg.id) a.onToken?.(msg.token, msg.t);
        break;
      case "gen-start":
        if (a && a.id === msg.id) a.onGenStart?.(msg.total);
        break;
      case "gen-progress":
        if (a && a.id === msg.id) a.onGenProgress?.(msg.step, msg.total, msg.t, msg.tps);
        break;
      case "gen-done":
        if (a && a.id === msg.id) {
          a.resolve({
            blob: msg.blob,
            width: msg.width,
            height: msg.height,
            ms: msg.ms,
            tokens: msg.tokens,
          });
          this._active = null;
        }
        break;
      case "done":
        if (a && a.id === msg.id) {
          a.resolve({ ms: msg.ms, tokens: msg.tokens });
          this._active = null;
        }
        break;
      case "error":
        if (a && a.id === msg.id) {
          a.reject(new Error(msg.message));
          this._active = null;
        } else {
          this._rejectAll(new Error(msg.message));
        }
        break;
    }
  }

  probeGPU() {
    return new Promise((resolve) => {
      this._probeWaiters.push({ resolve });
      this.worker.postMessage({ type: "probe" });
    });
  }

  load(onProgress) {
    if (onProgress) this.onProgress = onProgress;
    if (this.ready) return Promise.resolve("webgpu");
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load" });
    });
  }

  /** Understanding: image (+ prompt) -> streamed text. */
  understand(imageURL, prompt, { maxTokens = 256, onToken, onPrompt } = {}) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._active = { id, onToken, onPrompt, resolve, reject };
      this.worker.postMessage({ type: "understand", id, image: imageURL, prompt, maxTokens });
    });
  }

  /** Generation: prompt -> a decoded image (Blob). onGenProgress(step,total,t,tps) per image token. */
  generate(prompt, { onGenStart, onGenProgress } = {}) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._active = { id, onGenStart, onGenProgress, resolve, reject };
      this.worker.postMessage({ type: "generate", id, prompt });
    });
  }
}

export function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

export const JANUS_CSS = `
.mode-switch { display:flex; gap:.4rem; margin:.2rem 0 .8rem; flex-wrap:wrap; }
.mode-switch button { background:var(--bg-raised); color:var(--color); border:1px solid var(--border);
  border-radius:999px; padding:.35rem .9rem; font-size:.85rem; }
.mode-switch button[aria-pressed="true"] { background:var(--accent); color:var(--accent-ink); border-color:var(--border-strong); }
.mode-switch button:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.janus-grid { display:flex; flex-wrap:wrap; gap:1rem; align-items:flex-start; }
.janus-col { flex:1 1 300px; min-inline-size:280px; }
.preview-img { max-inline-size:100%; max-block-size:320px; border-radius:8px; display:block; }
.sample-strip { display:flex; gap:.5rem; flex-wrap:wrap; margin:.5rem 0; }
.sample-thumb { inline-size:76px; block-size:56px; object-fit:cover; border-radius:6px;
  border:2px solid transparent; cursor:pointer; padding:0; }
.sample-thumb.active { border-color:var(--accent); }
.sample-thumb:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.dropzone { border:2px dashed var(--border-strong); border-radius:var(--radius); background:var(--bg-raised);
  padding:.8rem; text-align:center; cursor:pointer; transition:border-color .15s, background .15s; margin-bottom:.5rem; }
.dropzone.drag { border-color:var(--accent); background:var(--bg-secondary); }
.dropzone:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.chips { display:flex; flex-wrap:wrap; gap:.4rem; margin:.4rem 0; }
.chip { font:inherit; font-size:.78rem; padding:.2rem .6rem; border-radius:999px;
  border:1px solid var(--border); background:var(--bg-raised); color:var(--color); cursor:pointer; }
.chip:hover { border-color:var(--accent); }
.chip:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.field { display:flex; flex-direction:column; gap:.25rem; font-size:.82rem; }
.answer { min-block-size:3rem; padding:.7rem; border:1px solid var(--border); border-radius:8px;
  background:var(--bg-raised); white-space:pre-wrap; line-height:1.6; }
.answer .caret { display:inline-block; inline-size:.5rem; background:var(--accent);
  animation:blink 1s steps(2) infinite; }
@keyframes blink { 50% { opacity:0; } }
@media (prefers-reduced-motion: reduce) { .answer .caret, .gen-canvas.working { animation:none; } }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem;
  color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.tok-stream { display:flex; flex-wrap:wrap; gap:.2rem; margin-top:.5rem; font-family:var(--font-mono); font-size:.72rem; }
.tok { padding:.05rem .3rem; border-radius:4px; background:var(--bg-secondary); border:1px solid var(--border); white-space:pre; }
.tok b { color:var(--muted); font-weight:400; font-size:.62rem; margin-inline-start:.2rem; }
.tmpl { font-family:var(--font-mono); font-size:.78rem; white-space:pre-wrap; word-break:break-word; }
.gen-stage { display:flex; flex-wrap:wrap; gap:1rem; align-items:flex-start; }
.gen-canvas { inline-size:288px; block-size:288px; max-inline-size:100%; border:1px solid var(--border);
  border-radius:10px; background:var(--bg-raised); image-rendering:auto; display:block; }
.gen-canvas.working { animation:pulse 1.4s ease-in-out infinite; }
@keyframes pulse { 50% { opacity:.7; } }
.token-grid { display:grid; grid-template-columns:repeat(24, 1fr); gap:1px; inline-size:192px; block-size:192px; }
.token-grid i { background:var(--border); border-radius:1px; }
.token-grid i.on { background:var(--accent); }
`;
