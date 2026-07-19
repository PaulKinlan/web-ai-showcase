// Shared front-end helpers for the SmolVLM2 pages: the worker handshake with multi-image + streaming
// plumbing, a real WebGPU probe (for the honest fallback), video-frame sampling, and the widget CSS.
// All inference lives in worker.js (off the main thread).

const WORKER_URL = "/web-ai-showcase/models/smolvlm2-video/worker.js";

export class SmolVLM2Engine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.onProgress = null;
    this._loadWaiters = [];
    this._probeWaiters = [];
    this._active = null; // { id, onToken, onPrompt, onShape, resolve, reject }
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
        if (this._active && this._active.id === msg.id) {
          this._active.onPrompt?.(msg.template, msg.imageCount);
        }
        break;
      case "shape":
        if (this._active && this._active.id === msg.id) {
          this._active.onShape?.({ promptTokens: msg.promptTokens, pixelValues: msg.pixelValues });
        }
        break;
      case "token":
        if (this._active && this._active.id === msg.id) this._active.onToken?.(msg.token, msg.t);
        break;
      case "done":
        if (this._active && this._active.id === msg.id) {
          this._active.resolve({
            ms: msg.ms,
            tokens: msg.tokens,
            promptTokens: msg.promptTokens,
            imageCount: msg.imageCount,
          });
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
   * Stream a generation over ONE OR MORE images/frames. onToken(token, tMs) fires per token;
   * onPrompt(template, imageCount) once; onShape({promptTokens, pixelValues}) once.
   * @param {string[]} imageURLs  data URLs, one per frame
   */
  generate(imageURLs, prompt, { maxTokens = 200, onToken, onPrompt, onShape } = {}) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._active = { id, onToken, onPrompt, onShape, resolve, reject };
      this.worker.postMessage({ type: "run", id, images: imageURLs, prompt, maxTokens });
    });
  }
}

/** Probe WebGPU on the main thread too (for instant UI gating before we ever load). */
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

/** Fetch a same-origin sample and return a self-contained data URL for the worker. */
export async function urlToDataURL(src) {
  const blob = await (await fetch(src)).blob();
  return fileToDataURL(new File([blob], "sample", { type: blob.type }));
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/**
 * Sample N evenly-spaced frames from a video File as data URLs. Runs on the main thread but yields
 * between seeks (each seek is async), so it never blocks a long synchronous task. Frames are drawn to
 * a capped-size canvas so the payload to the worker stays small.
 * @param {File} file  a video file
 * @param {number} n   number of frames to sample
 * @param {number} maxSide  cap the longest edge (px)
 * @returns {Promise<string[]>} data URLs, in temporal order
 */
export function sampleVideoFrames(file, n = 4, maxSide = 384) {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.preload = "auto";
    const url = URL.createObjectURL(file);
    video.src = url;
    const frames = [];
    let times = [];
    let idx = 0;

    const cleanup = () => URL.revokeObjectURL(url);

    video.addEventListener("error", () => {
      cleanup();
      reject(new Error("Could not decode that video."));
    });

    video.addEventListener("loadedmetadata", () => {
      const dur = video.duration && isFinite(video.duration) ? video.duration : 0;
      if (!dur) {
        cleanup();
        reject(new Error("Video has no readable duration."));
        return;
      }
      // Evenly spaced, avoiding the very first/last frame (often black).
      times = Array.from({ length: n }, (_, i) => (dur * (i + 0.5)) / n);
      seekNext();
    });

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    function seekNext() {
      if (idx >= times.length) {
        cleanup();
        resolve(frames);
        return;
      }
      video.currentTime = times[idx];
    }

    video.addEventListener("seeked", () => {
      const iw = video.videoWidth, ih = video.videoHeight;
      const scale = Math.min(1, maxSide / Math.max(iw, ih));
      canvas.width = Math.max(1, Math.round(iw * scale));
      canvas.height = Math.max(1, Math.round(ih * scale));
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      frames.push(canvas.toDataURL("image/jpeg", 0.85));
      idx++;
      seekNext();
    });
  });
}

/** The labelled needs-WebGPU state — never faked output, always the real reason + how to enable. */
export function webgpuFallbackHTML(probe) {
  const reasons = {
    "no-gpu": "This browser doesn't expose the WebGPU API at all.",
    "no-adapter":
      "WebGPU is present but no GPU adapter is available here (common in headless, VMs, or with the GPU blocklisted).",
    "adapter-error": "Requesting a WebGPU adapter threw an error.",
  };
  const why = reasons[probe.reason] ?? "WebGPU isn't usable here.";
  return `
    <strong>SmolVLM2 needs WebGPU — and it isn't available in this browser.</strong>
    <p class="muted" style="margin:.4rem 0">${why}${
    probe.detail ? " (" + escapeHTML(probe.detail) + ")" : ""
  }
    A 256M-parameter video-VLM in 4-bit still needs a GPU to run at a usable speed, so this page won't
    fake a result. To try it for real:</p>
    <ul class="muted" style="margin:.2rem 0">
      <li>Open in <strong>Chrome or Edge 113+</strong> on a machine with a GPU (desktop or a recent laptop/phone).</li>
      <li>Check <code>chrome://gpu</code> — "WebGPU" should read <em>Hardware accelerated</em>.</li>
      <li>If it's blocklisted, enable <code>chrome://flags/#enable-unsafe-webgpu</code> and relaunch.</li>
      <li>q4f16 also wants the <code>shader-f16</code> feature (present on most modern GPUs).</li>
    </ul>`;
}

export const SMOL2_CSS = `
.vlm-grid { display:flex; flex-wrap:wrap; gap:1rem; align-items:flex-start; }
.vlm-img-col { flex:1 1 300px; min-inline-size:0; max-inline-size:460px; }
.vlm-out-col { flex:1 1 300px; min-inline-size:0; }
.preview-img { max-inline-size:100%; max-block-size:320px; border-radius:8px; display:block; }
.frame-strip { display:flex; gap:.4rem; flex-wrap:wrap; margin:.5rem 0; }
.frame-thumb { position:relative; inline-size:84px; block-size:64px; border-radius:6px; overflow:hidden;
  border:2px solid var(--border); background:var(--bg-raised); }
.frame-thumb img { inline-size:100%; block-size:100%; object-fit:cover; display:block; }
.frame-thumb .fnum { position:absolute; inset-block-start:2px; inset-inline-start:2px; font-size:.6rem;
  background:var(--accent); color:var(--accent-ink); border-radius:3px; padding:0 .25rem; font-family:var(--font-mono); }
.frame-thumb .fdel { position:absolute; inset-block-start:2px; inset-inline-end:2px; inline-size:20px; block-size:20px;
  border:0; border-radius:4px; background:rgba(0,0,0,.55); color:#fff; font-size:.85rem; line-height:1; cursor:pointer; padding:0; }
.frame-thumb .fdel:focus-visible { outline:2px solid var(--accent); outline-offset:1px; }
.sample-strip { display:flex; gap:.5rem; flex-wrap:wrap; margin:.5rem 0; }
.sample-thumb { inline-size:76px; block-size:56px; object-fit:cover; border-radius:6px;
  border:2px solid transparent; cursor:pointer; padding:0; background:var(--bg-raised); }
.sample-thumb.active { border-color:var(--accent); }
.sample-thumb:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.dropzone { border:2px dashed var(--border-strong); border-radius:var(--radius); background:var(--bg-raised);
  padding:.8rem; text-align:center; cursor:pointer; transition:border-color .15s, background .15s; }
.dropzone.drag { border-color:var(--accent); background:var(--bg-secondary); }
.dropzone:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.chips { display:flex; flex-wrap:wrap; gap:.4rem; margin:.4rem 0; }
.chip { font:inherit; font-size:.78rem; padding:.3rem .6rem; border-radius:999px; min-block-size:32px;
  border:1px solid var(--border); background:var(--bg-raised); color:var(--color); cursor:pointer; }
.chip:hover { border-color:var(--accent); }
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
.controls-row { display:flex; flex-wrap:wrap; gap:.6rem; align-items:center; margin:.6rem 0; }
.stage-canvas { inline-size:100%; max-inline-size:100%; block-size:auto; border-radius:8px; border:1px solid var(--border);
  background:var(--bg-raised); display:block; }
.fallback { border:1px solid var(--warn); border-radius:var(--radius); background:var(--bg-raised); padding:1rem; }
.fallback code { background:var(--bg-secondary); padding:.05rem .3rem; border-radius:4px; }
.field-label { display:flex; flex-direction:column; gap:.25rem; font-size:.82rem; }
`;
