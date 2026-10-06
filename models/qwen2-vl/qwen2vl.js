// Front-end helpers for the Qwen2-VL pages: the worker handshake, streaming plumbing, a real WebGPU
// probe (for the honest fallback), image helpers, and the widget CSS. Inference is in worker.js.

const WORKER_URL = "/web-ai-showcase/models/qwen2-vl/worker.js";

export class Qwen2VLEngine {
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

  /**
   * Stream a generation. onToken(token, tMs) fires per token; onPrompt(template) once.
   * `history` is optional prior turns [{role:'user'|'assistant', text}] for multi-turn chat — the
   * image is attached to the first user turn only.
   */
  generate(imageURL, prompt, { maxTokens = 200, history = null, onToken, onPrompt } = {}) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._active = { id, onToken, onPrompt, resolve, reject };
      this.worker.postMessage({ type: "run", id, image: imageURL, prompt, maxTokens, history });
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

// Honest, labelled degradation for runtime failures (web-ai-showcase-oow). Qwen2-VL ships
// WebGPU-only, and the ORT WebGPU backend bundled with the pinned transformers.js (3.7.5) can crash
// inside the model's OWN forward pass on some GPU/driver/browser builds — measured:
//   [WebGPU] Kernel "[Concat] /model/layers.0/self_attn/Concat_7" failed. Error: Failed to generate
//   kernel's output[0] with dims [1,2,286,128]…
// A visitor must never get that raw kernel string as the user-facing message. explainRuntimeFailure
// classifies the known failure classes into { kind, headline, advice, raw } (plain-language what
// happened + what the visitor can do); showRuntimeFailure renders it into the page's existing
// role=status live region and demotes the raw text to a collapsed <details> (and it stays in the
// console via the worker's console.error). Returns null for states the shared loader already labels
// honestly (needs-WebGPU), so callers must leave those alone.
const KERNEL_RE =
  /\[WebGPU\][\s\S]{0,200}?\b(?:kernel|fail(?:ed|ure)?|crash(?:ed)?)\b|\bKernel\b[\s\S]{0,200}?failed|Failed to generate kernel|\[Concat\][\s\S]{0,200}?failed/i;
const DEVICE_LOST_RE = /device lost|GPUDevice.?lost|DeviceLostError/i;
const OOM_RE =
  /\bout of memory\b|\bOOM\b|Array\s*buffer allocation failed|\b(?:memory|gpu\s*memory|vram|tensor|heap)\s+allocation failed/i;

export function explainRuntimeFailure(err) {
  const raw = String(err?.message ?? err ?? "Unknown error");
  if (/^needs-webgpu$/i.test(raw.trim())) return null; // loader already shows the labelled state
  if (DEVICE_LOST_RE.test(raw)) {
    return {
      kind: "device-lost",
      raw,
      headline: "The browser took the GPU away while Qwen2-VL was running.",
      advice:
        "This usually means a driver reset or GPU resource pressure. Reload the page and try " +
        "again — closing other GPU-heavy tabs or apps first helps.",
    };
  }
  if (OOM_RE.test(raw)) {
    return {
      kind: "out-of-memory",
      raw,
      headline: "This device ran out of memory running Qwen2-VL.",
      advice:
        "It is a ~2B-parameter model that needs several GB of free RAM and GPU memory. Close " +
        "other tabs and apps, then retry — or use a smaller on-device vision-language demo such " +
        "as Moondream 2 or SmolVLM.",
    };
  }
  if (KERNEL_RE.test(raw)) {
    return {
      kind: "webgpu-kernel",
      raw,
      headline: "This browser's WebGPU backend crashed while running Qwen2-VL.",
      advice:
        "The model loaded, but the GPU compute step failed inside the model itself — a known " +
        "problem with this ONNX build on some browser, GPU and driver combinations, not something " +
        "you did. What you can do: update your browser (a newer WebGPU runtime may fix it), try a " +
        "different browser or GPU, or run one of the smaller on-device vision-language demos " +
        "instead — SmolVLM, Moondream 2 or FastVLM work on far more devices.",
    };
  }
  return null; // unclassified: the caller keeps its existing behaviour (raw message on the page)
}

// Remove any sibling .err-detail technical block from `statusEl` (e.g. on retry or success).
export function clearRuntimeFailure(statusEl) {
  if (!statusEl) return;
  while (statusEl.nextElementSibling?.classList?.contains("err-detail")) {
    statusEl.nextElementSibling.remove();
  }
}

// Render a classified failure into `statusEl` (the page's role=status line): labelled headline +
// advice as the user-facing message, raw runtime text in a collapsed <details> inserted right
// after. Returns true when it rendered a classified degradation, false when the error was left to
// the caller's existing handling.
export function showRuntimeFailure(statusEl, err, { phase = "Generation" } = {}) {
  const info = explainRuntimeFailure(err);
  // Clear any detail block from a previous failure so retries never stack them.
  clearRuntimeFailure(statusEl);
  if (!info) return false;
  statusEl.textContent = `${phase} failed. ${info.headline} ${info.advice}`;
  statusEl.classList.add("err");
  statusEl.classList.remove("ok");
  const detail = document.createElement("details");
  detail.className = "err-detail";
  const summary = document.createElement("summary");
  summary.textContent = "Technical detail (useful for a bug report)";
  const pre = document.createElement("pre");
  pre.textContent = info.raw;
  detail.append(summary, pre);
  statusEl.insertAdjacentElement("afterend", detail);
  return true;
}

export const QWEN_CSS = `
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
.err-detail { margin:.3rem 0 0; font-size:.78rem; color:var(--muted); }
.err-detail summary { cursor:pointer; }
.err-detail summary:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.err-detail pre { white-space:pre-wrap; word-break:break-word; font-size:.72rem; margin:.3rem 0 0;
  padding:.5rem; border:1px solid var(--border); border-radius:6px; background:var(--bg-raised); }
`;
