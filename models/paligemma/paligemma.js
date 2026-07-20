// Front-end helpers for the PaliGemma pages: the worker handshake, streaming plumbing, a real WebGPU
// probe (for the honest fallback), image helpers, <loc> box decoding, and the widget CSS.
// Inference is in worker.js.

const WORKER_URL = "/web-ai-showcase/models/paligemma/worker.js";

export class PaliGemmaEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.onProgress = null; // legacy single-% callback (kept for back-compat)
    this.onEvent = null; // NEW: raw download-tracker events {status, file, loaded, total, …}
    this.onPaused = null; // fires when a Pause takes effect (partials preserved → Resume works)
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
      case "dl":
        this.onEvent?.(msg.evt);
        if (msg.evt?.status === "progress" && typeof msg.evt.progress === "number") {
          this.onProgress?.(msg.evt); // legacy consumers still get a %-ish signal
        }
        break;
      case "progress": // legacy path (pre-resumable worker)
        this.onProgress?.(msg.p);
        break;
      case "paused":
        this.onPaused?.();
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

  // load({ onEvent, onProgress, onPaused }) → resolves when the model is READY. Pause keeps the
  // promise pending (a later resume() continues the same resumable prefetch from persisted partials).
  load(arg = {}) {
    // Back-compat: a bare function arg is the legacy onProgress callback.
    const { onEvent, onProgress, onPaused } = typeof arg === "function" ? { onProgress: arg } : arg;
    if (onEvent) this.onEvent = onEvent;
    if (onProgress) this.onProgress = onProgress;
    if (onPaused) this.onPaused = onPaused;
    if (this.ready) return Promise.resolve("webgpu");
    const first = this._loadWaiters.length === 0;
    const p = new Promise((resolve, reject) => this._loadWaiters.push({ resolve, reject }));
    if (first) this.worker.postMessage({ type: "load" });
    return p;
  }

  /** Pause the in-flight resumable download; partials are preserved on disk. Safe no-op if not loading. */
  pause() {
    this.worker.postMessage({ type: "pause" });
  }

  /** Resume a paused download from persisted partials (a genuine continuation, never a fresh restart). */
  resume() {
    this.worker.postMessage({ type: "resume" });
  }

  /**
   * Stream a generation. `task` is a PaliGemma prefix (e.g. "caption en", "detect car").
   * onToken(token, tMs) fires per token; onPrompt(prompt) once. Set skipSpecial:false to keep
   * <locXXXX>/<segXXX> tokens (needed to decode detection boxes).
   */
  generate(imageURL, task, { maxTokens = 100, skipSpecial = true, onToken, onPrompt } = {}) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._active = { id, onToken, onPrompt, resolve, reject };
      this.worker.postMessage({
        type: "run",
        id,
        image: imageURL,
        prompt: task,
        maxTokens,
        skipSpecial,
      });
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

/**
 * Decode PaliGemma detection output into boxes. Detection output looks like:
 *   "<loc0512><loc0128><loc0800><loc0640> cat ; <loc0100>...<label>"
 * Each group of four <locNNNN> tokens is [y_min, x_min, y_max, x_max], normalised to 0..1023.
 * These are REAL tokens the model emitted (decoded, never invented). Returns [{label, box:[x,y,w,h]}]
 * in fractions of the image (0..1), ready to scale to any displayed size.
 */
export function decodeDetections(rawText) {
  const results = [];
  const locRe = /<loc(\d{4})>/g;
  const segments = rawText.split(";");
  for (const s of segments) {
    const locs = [...s.matchAll(locRe)].map((m) => parseInt(m[1], 10));
    if (locs.length < 4) continue;
    const [yMin, xMin, yMax, xMax] = locs.slice(0, 4).map((v) => v / 1024);
    const label = s.replace(locRe, "").replace(/<[^>]*>/g, "").trim() || "object";
    results.push({
      label,
      box: [xMin, yMin, Math.max(0, xMax - xMin), Math.max(0, yMax - yMin)],
    });
  }
  return results;
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

export const PALI_CSS = `
.vlm-grid { display:flex; flex-wrap:wrap; gap:1rem; align-items:flex-start; }
.vlm-img-col { flex:1 1 280px; max-inline-size:440px; }
.vlm-out-col { flex:1 1 300px; }
.preview-wrap { position:relative; display:inline-block; max-inline-size:100%; }
.preview-img { max-inline-size:100%; max-block-size:360px; border-radius:8px; display:block; }
.box-layer { position:absolute; inset:0; pointer-events:none; }
.det-box { position:absolute; border:2px solid var(--accent); border-radius:3px; box-shadow:0 0 0 1px rgba(0,0,0,.4); }
.det-label { position:absolute; inset-block-start:-1.1rem; inset-inline-start:-2px; font-family:var(--font-mono);
  font-size:.66rem; background:var(--accent); color:#fff; padding:.02rem .3rem; border-radius:3px; white-space:nowrap; }
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
/* Shared multi-file download panel (lib/download-ui.mjs) — inherited by every PaliGemma page. */
#model-loader { container-type:inline-size; }
.dl-panel { display:flex; flex-direction:column; gap:.5rem; }
.dl-phase { margin:0; font-family:var(--font-display); }
.dl-bar { inline-size:100%; block-size:.7rem; }
.dl-agg, .dl-storage { margin:0; font-size:.82rem; font-family:var(--font-mono); }
.dl-actions { display:flex; flex-wrap:wrap; gap:.5rem; }
.dl-actions button { min-block-size:2.6rem; }
.dl-files { margin-top:.2rem; }
.dl-files > summary { cursor:pointer; font-size:.85rem; }
.dl-file-list { display:flex; flex-direction:column; gap:.3rem; margin-top:.5rem; }
.dl-file { display:grid; grid-template-columns:minmax(0,1fr) auto; grid-template-areas:"name state" "bytes bytes" "bar bar";
  gap:.15rem .5rem; padding:.4rem .5rem; border:1px solid var(--border); border-radius:8px; background:var(--bg-raised); font-size:.78rem; }
@container (min-width: 30rem) { .dl-file { grid-template-columns:minmax(0,1fr) 6rem 8rem 6rem; grid-template-areas:"name state bytes bar"; align-items:center; } }
.dl-fname { grid-area:name; font-family:var(--font-mono); overflow-wrap:anywhere; }
.dl-fstate { grid-area:state; font-size:.72rem; text-transform:uppercase; letter-spacing:.03em; color:var(--muted); }
.dl-fstate[data-state="complete"], .dl-fstate[data-state="cached"] { color:var(--accent); }
.dl-fstate[data-state="failed"] { color:#c0392b; }
.dl-fbytes { grid-area:bytes; font-family:var(--font-mono); color:var(--muted); }
.dl-fbar { grid-area:bar; inline-size:100%; block-size:.5rem; }
.attribution { font-size:.85rem; color:var(--muted); margin:.2rem 0 1.2rem; line-height:1.6; }
.attribution a { color:inherit; }
`;
