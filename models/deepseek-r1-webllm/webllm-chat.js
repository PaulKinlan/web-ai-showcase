// Front-end helpers shared by every DeepSeek-R1-Distill (WebLLM) page: the worker handshake +
// streaming plumbing, the honest WebGPU gate/fallback, the <think> reasoning-trace splitter (the
// whole point of an R1 model), and the widget CSS. All real inference happens in worker.js, which
// drives lib/webllm.js (MLC's WebGPU engine). We re-export the canonical webGPUAdapterAvailable()
// so pages gate on the exact helper the task mandates. The engine is parameterised by MLC model id
// so a page can run a second model for the multi-model compositions.

export { webGPUAdapterAvailable } from "/web-ai-showcase/lib/webllm.js";

const WORKER_URL = "/web-ai-showcase/models/deepseek-r1-webllm/worker.js";

export class WebLLMChatEngine {
  /** @param {string} [modelId] MLC build id — defaults to the page's primary R1 model. */
  constructor(modelId) {
    this.modelId = modelId || "DeepSeek-R1-Distill-Llama-8B-q4f16_1-MLC";
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.onProgress = null;
    this._loadWaiters = [];
    this._active = null;
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
    switch (msg.type) {
      case "progress":
        this.onProgress?.(msg.p);
        break;
      case "ready":
        this.ready = true;
        for (const w of this._loadWaiters) w.resolve();
        this._loadWaiters = [];
        break;
      case "first":
        if (this._active && this._active.id === msg.id) this._active.onFirstToken?.(msg.t);
        break;
      case "token":
        if (this._active && this._active.id === msg.id) this._active.onToken?.(msg.delta);
        break;
      case "done":
        if (this._active && this._active.id === msg.id) {
          this._active.resolve({
            text: msg.text,
            ms: msg.ms,
            ttft: msg.ttft,
            chunks: msg.chunks,
            stats: msg.stats,
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

  load(onProgress) {
    if (onProgress) this.onProgress = onProgress;
    if (this.ready) return Promise.resolve();
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load", modelId: this.modelId });
    });
  }

  chat(req, { onToken, onFirstToken } = {}) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._active = { id, onToken, onFirstToken, resolve, reject };
      this.worker.postMessage({ type: "run", id, req, modelId: this.modelId });
    });
  }

  stop() {
    this.worker.postMessage({ type: "stop" });
  }
}

/**
 * Split a DeepSeek-R1-Distill response into its reasoning trace and its final answer.
 *
 * R1-distill models are trained to "think out loud" first, then answer. Their chat template opens a
 * `<think>` block for the assistant, so the streamed text is one of these shapes:
 *   1. "<think>reasoning…</think>answer"   (explicit open + close)
 *   2. "reasoning…</think>answer"          (template opened <think>; model emits only the close)
 *   3. "<think>reasoning…"                 (still streaming, block not closed yet)
 *   4. "reasoning…"                        (still streaming, no close yet, no open emitted)
 *   5. "answer"                            (rare: no reasoning at all)
 *
 * Returns { reasoning, answer, thinking }: `thinking` is true while the reasoning block is still
 * open (no `</think>` seen yet), which the UI uses to show a live "reasoning…" state. Everything
 * before the first `</think>` (minus a leading `<think>`) is reasoning; everything after is answer.
 */
export function splitThink(text) {
  const s = String(text ?? "");
  const OPEN = "<think>";
  const CLOSE = "</think>";
  const openIdx = s.indexOf(OPEN);
  const closeIdx = s.indexOf(CLOSE);

  if (closeIdx !== -1) {
    const start = openIdx !== -1 && openIdx < closeIdx ? openIdx + OPEN.length : 0;
    return {
      reasoning: s.slice(start, closeIdx).trim(),
      answer: s.slice(closeIdx + CLOSE.length).trim(),
      thinking: false,
    };
  }
  // No </think> yet — the reasoning block is still open, so all content so far is reasoning.
  const start = openIdx !== -1 ? openIdx + OPEN.length : 0;
  return { reasoning: s.slice(start).trim(), answer: "", thinking: true };
}

export async function probeWebGPU() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) {
    return { ok: false, reason: "no-gpu" };
  }
  let adapter = null;
  try {
    adapter = await navigator.gpu.requestAdapter();
  } catch (e) {
    return { ok: false, reason: "adapter-error", detail: String(e?.message ?? e) };
  }
  if (!adapter) return { ok: false, reason: "no-adapter" };
  return { ok: true, shaderF16: adapter.features?.has?.("shader-f16") ?? false };
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

export function estimateTokens(text) {
  return Math.max(1, Math.round((text || "").length / 4));
}

/** The labelled needs-WebGPU state. Never a faked reply — the real reason + how to enable, plus the
 *  ~4.5 GB download so a visitor knows what running it for real would cost. */
export function webgpuFallbackHTML(probe, sizeLabel = "~4.5 GB", vram = "~5 GB") {
  const reasons = {
    "no-gpu": "This browser doesn't expose the WebGPU API at all.",
    "no-adapter":
      "WebGPU is present but no GPU adapter is available here (normal in headless Chrome, many VMs, or when the GPU is blocklisted).",
    "adapter-error": "Requesting a WebGPU adapter threw an error.",
  };
  const why = reasons[probe.reason] ?? "WebGPU isn't usable here.";
  return `
    <strong>DeepSeek-R1-Distill runs on WebLLM, which needs WebGPU — and it isn't available in this browser.</strong>
    <p class="muted" style="margin:.4rem 0">${why}${
    probe.detail ? " (" + escapeHTML(probe.detail) + ")" : ""
  }
    WebLLM has <em>no WASM fallback</em>: an 8B reasoning model that thinks out loud before answering needs a
    real GPU, and the weights are a <strong>${sizeLabel}</strong> download (cached after the first load). So
    this page won't fake a reasoning trace. To run it for real:</p>
    <ul class="muted" style="margin:.2rem 0">
      <li>Open in <strong>Chrome or Edge 113+</strong> on a machine with a capable GPU (${vram} VRAM).</li>
      <li>Check <code>chrome://gpu</code> — "WebGPU" should read <em>Hardware accelerated</em>.</li>
      <li>If it's blocklisted, enable <code>chrome://flags/#enable-unsafe-webgpu</code> and relaunch.</li>
      <li>The q4f16_1 build is fastest on GPUs that report the <code>shader-f16</code> feature (most modern ones).</li>
    </ul>`;
}

export const CHAT_CSS = `
.chat-wrap { display:flex; flex-direction:column; gap:.6rem; }
.composer { display:flex; gap:.5rem; align-items:flex-end; }
.composer textarea { flex:1 1 auto; field-sizing:content; width:auto; min-block-size:2.6lh; max-block-size:10lh; resize:vertical; }
.controls { display:flex; flex-wrap:wrap; gap:1rem 1.4rem; margin:.4rem 0 .2rem; }
.control { display:flex; flex-direction:column; gap:.2rem; font-size:.78rem; color:var(--muted); min-inline-size:9rem; }
.control .row { display:flex; align-items:center; gap:.5rem; }
.control input[type=range] { flex:1 1 auto; accent-color:var(--accent); }
.control b { color:var(--color); font-family:var(--font-mono); font-weight:600; min-inline-size:2.6rem; text-align:right; }
.sysbox { display:flex; flex-direction:column; gap:.25rem; font-size:.82rem; }
.sysbox textarea { field-sizing:content; width:100%; min-block-size:2.2lh; max-block-size:7lh; }
.fallback { border:1px solid var(--warn); border-radius:var(--radius); background:var(--bg-raised); padding:1rem; }
.fallback code { background:var(--bg-secondary); padding:.05rem .3rem; border-radius:4px; }
.inside-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.6rem; margin:.4rem 0 .8rem; }
.stat { border:1px solid var(--border); border-radius:8px; background:var(--bg-raised); padding:.5rem .6rem; }
.stat .k { font-size:.68rem; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
.stat .v { font-family:var(--font-mono); font-size:1.05rem; color:var(--color); }
.stat .v small { font-size:.7rem; color:var(--muted); }
.wire { font-family:var(--font-mono); font-size:.76rem; white-space:pre-wrap; word-break:break-word; max-block-size:20rem; overflow:auto; }
.chip { font:inherit; font-size:.78rem; padding:.2rem .6rem; border-radius:999px; border:1px solid var(--border); background:var(--bg-raised); color:var(--color); cursor:pointer; text-align:left; }
.chip:hover { border-color:var(--accent); }
.chip[aria-pressed=true] { border-color:var(--accent); background:var(--accent); color:var(--accent-ink); }
.chips { display:flex; flex-wrap:wrap; gap:.4rem; margin:.2rem 0; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.76rem; color:var(--muted); margin-top:.5rem; }
.readout b { color:var(--color); font-weight:600; }
.caret { display:inline-block; inline-size:.5rem; block-size:1em; vertical-align:text-bottom; background:currentColor; animation:blink 1s steps(2) infinite; }
@keyframes blink { 50% { opacity:0; } }
@media (prefers-reduced-motion: reduce) { .caret { animation:none; } }
/* Reasoning trace — a native <details> disclosure so it's collapsible, searchable and accessible. */
.reasoning { border:1px solid var(--border); border-radius:10px; background:var(--bg-raised); margin:.4rem 0; overflow:hidden; }
.reasoning > summary { cursor:pointer; padding:.5rem .7rem; font-size:.8rem; color:var(--muted); display:flex; align-items:center; gap:.5rem; list-style:none; }
.reasoning > summary::-webkit-details-marker { display:none; }
.reasoning > summary::before { content:"▶"; font-size:.6rem; transition:transform .15s; }
.reasoning[open] > summary::before { transform:rotate(90deg); }
@media (prefers-reduced-motion: reduce) { .reasoning > summary::before { transition:none; } }
.reasoning .badge { margin-inline-start:auto; font-family:var(--font-mono); font-size:.68rem; }
.reasoning .think-body { padding:.2rem .8rem .7rem; font-size:.82rem; line-height:1.6; color:var(--muted); white-space:pre-wrap; overflow-wrap:anywhere; border-block-start:1px dashed var(--border); max-block-size:24rem; overflow:auto; }
.answer { border:1px solid var(--border); border-inline-start:3px solid var(--accent); border-radius:8px; background:var(--bg-secondary); padding:.6rem .8rem; line-height:1.65; white-space:pre-wrap; overflow-wrap:anywhere; margin:.4rem 0; }
.answer.pending { color:var(--muted); font-style:italic; }
.think-live { display:inline-flex; align-items:center; gap:.35rem; color:var(--accent); font-size:.72rem; }
.think-dot { inline-size:.4rem; block-size:.4rem; border-radius:50%; background:currentColor; animation:pulse 1s ease-in-out infinite; }
@keyframes pulse { 0%,100% { opacity:.3; } 50% { opacity:1; } }
@media (prefers-reduced-motion: reduce) { .think-dot { animation:none; } }
`;
