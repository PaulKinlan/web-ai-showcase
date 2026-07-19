// Front-end helpers for the Moondream2 pages: the worker handshake, streaming plumbing, a real
// WebGPU probe (for the honest fallback), image helpers, and the widget CSS. Inference is in worker.js.

const WORKER_URL = "/web-ai-showcase/models/moondream2-vlm/worker.js";

export class MoondreamEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.onProgress = null;
    this._loadWaiters = [];
    this._probeWaiters = [];
    this._active = null; // { id, onToken, onPrompt, resolve, reject }
    this._id = 0;
    this.worker.addEventListener("message", (e) => this._onMessage(e.data));
    this.worker.addEventListener("error", (e) => {
      const err = new Error(e.message || "Worker failed to start");
      this._rejectAll(err);
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
        if (this._active && this._active.id === msg.id) this._active.onPrompt?.(msg.template);
        break;
      case "token":
        if (this._active && this._active.id === msg.id) this._active.onToken?.(msg.token, msg.t);
        break;
      case "done":
        if (this._active && this._active.id === msg.id) {
          this._active.resolve({ ms: msg.ms, tokens: msg.tokens, promptLen: msg.promptLen });
          this._active = null;
        }
        break;
      case "error":
        if (this._active && this._active.id === msg.id) {
          this._active.reject(new Error(msg.message));
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

  /** Stream a generation. onToken(token, tMs) fires per token; onPrompt(template) once. */
  generate(imageURL, prompt, { maxTokens = 200, onToken, onPrompt } = {}) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._active = { id, onToken, onPrompt, resolve, reject };
      this.worker.postMessage({ type: "run", id, image: imageURL, prompt, maxTokens });
    });
  }
}

/** Probe WebGPU on the main thread (for instant UI gating before we ever load). */
export async function probeWebGPUMain() {
  if (!("gpu" in navigator)) return { ok: false, reason: "no-gpu" };
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { ok: false, reason: "no-adapter" };
    return { ok: true, shaderF16: adapter.features?.has?.("shader-f16") ?? false };
  } catch (e) {
    return { ok: false, reason: "adapter-error", detail: String(e?.message ?? e) };
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

export const MOON_CSS = `
.vlm-grid { display:flex; flex-wrap:wrap; gap:1rem; align-items:flex-start; }
.vlm-img-col { flex:1 1 280px; max-inline-size:420px; }
.vlm-out-col { flex:1 1 300px; }
.preview-img { max-inline-size:100%; max-block-size:340px; border-radius:8px; display:block; }
.sample-strip { display:flex; gap:.5rem; flex-wrap:wrap; margin:.5rem 0; }
.sample-thumb { inline-size:76px; block-size:56px; object-fit:cover; border-radius:6px;
  border:2px solid transparent; cursor:pointer; padding:0; }
.sample-thumb.active { border-color:var(--accent); }
.sample-thumb:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.dropzone { border:2px dashed var(--border-strong); border-radius:var(--radius); background:var(--bg-raised);
  padding:.8rem; text-align:center; cursor:pointer; transition:border-color .15s, background .15s; }
.dropzone.drag { border-color:var(--accent); background:var(--bg-secondary); }
.dropzone:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.chips { display:flex; flex-wrap:wrap; gap:.4rem; margin:.4rem 0; }
.chip { font:inherit; font-size:.78rem; padding:.2rem .6rem; border-radius:999px;
  border:1px solid var(--border); background:var(--bg-raised); color:var(--color); cursor:pointer; }
.chip:hover { border-color:var(--accent); }
.chip[aria-pressed="true"] { border-color:var(--accent); background:var(--bg-secondary); }
.chip:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.answer { min-block-size:3rem; padding:.7rem; border:1px solid var(--border); border-radius:8px;
  background:var(--bg-raised); white-space:pre-wrap; line-height:1.6; }
.answer .caret { display:inline-block; inline-size:.5rem; background:var(--accent);
  animation:blink 1s steps(2) infinite; }
@keyframes blink { 50% { opacity:0; } }
@media (prefers-reduced-motion: reduce) { .answer .caret { animation:none; } }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem;
  color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.tok-stream { display:flex; flex-wrap:wrap; gap:.2rem; margin-top:.5rem; font-family:var(--font-mono); font-size:.72rem; }
.tok { padding:.05rem .3rem; border-radius:4px; background:var(--bg-secondary); border:1px solid var(--border); white-space:pre; }
.tok b { color:var(--muted); font-weight:400; font-size:.62rem; margin-inline-start:.2rem; }
.tmpl { font-family:var(--font-mono); font-size:.78rem; white-space:pre-wrap; word-break:break-word; }
.turn { padding:.5rem .7rem; border-radius:8px; margin:.35rem 0; border:1px solid var(--border); }
.turn.q { background:var(--bg-secondary); }
.turn.a { background:var(--bg-raised); white-space:pre-wrap; line-height:1.6; }
.turn .who { font-size:.68rem; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); display:block; margin-block-end:.2rem; }
`;
