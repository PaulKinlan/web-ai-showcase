// Front-end helpers shared by every SmolLM3-3B page: the worker handshake + streaming plumbing, the
// honest WebGPU gate/fallback, SmolLM3's /think · /no_think switch labelling, and the chat widget CSS.
// All real inference happens in worker.js, which drives the Transformers.js pipeline (ONNX / WebGPU).
//
// What makes SmolLM3 distinct here: it is a HYBRID reasoning model running on TRANSFORMERS.JS (not
// WebLLM) — its chat template exposes an enable_thinking switch (the /think · /no_think modes). In
// thinking mode it emits an explicit <think>…</think> chain-of-thought before the answer; the worker
// separates the reasoning stream from the answer stream so the UI shows the private trace distinctly.

export { hasWebGPU as webGPUAdapterAvailable } from "/web-ai-showcase/lib/webai.js";

const WORKER_URL = "/web-ai-showcase/models/smollm3/worker.js";

export class SmolLM3Engine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "webgpu";
    this.onProgress = null;
    this._loadWaiters = [];
    this._active = null; // { id, onToken, onReasoning, onFirstToken, onPrompt, resolve, reject }
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
        this.device = msg.device;
        for (const w of this._loadWaiters) w.resolve(msg.device);
        this._loadWaiters = [];
        break;
      case "prompt":
        if (this._active && this._active.id === msg.id) {
          this._active.onPrompt?.(msg.template, msg.thinking);
        }
        break;
      case "first":
        if (this._active && this._active.id === msg.id) this._active.onFirstToken?.(msg.t);
        break;
      case "token":
        if (this._active && this._active.id === msg.id) {
          if (msg.kind === "reasoning") this._active.onReasoning?.(msg.delta);
          else this._active.onToken?.(msg.delta);
        }
        break;
      case "done":
        if (this._active && this._active.id === msg.id) {
          this._active.resolve({
            text: msg.text,
            reasoning: msg.reasoning,
            ms: msg.ms,
            ttft: msg.ttft,
            chunks: msg.chunks,
            reasoningChunks: msg.reasoningChunks,
            thinking: msg.thinking,
            device: msg.device,
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

  /** Load the model (Transformers.js / ONNX on WebGPU, dtype q4f16). Resolves with the device string. */
  load(onProgress) {
    if (onProgress) this.onProgress = onProgress;
    if (this.ready) return Promise.resolve(this.device);
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load" });
    });
  }

  /**
   * Stream a chat completion. `messages` = array; `opts` = { thinking, temperature, topP, maxTokens }.
   * onToken(delta) fires per answer chunk; onReasoning(delta) per reasoning chunk; onFirstToken(ttftMs)
   * once; onPrompt(template, thinking) once with the exact templated prompt.
   */
  chat(messages, { onToken, onReasoning, onFirstToken, onPrompt, ...opts } = {}) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._active = { id, onToken, onReasoning, onFirstToken, onPrompt, resolve, reject };
      this.worker.postMessage({ type: "run", id, messages, opts });
    });
  }

  /** Cooperative interrupt — the Transformers.js stopping criteria ends the decode loop. */
  stop() {
    this.worker.postMessage({ type: "stop" });
  }
}

/** Detailed WebGPU probe used only to label the honest fallback. Returns { ok, reason?, detail? }. */
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

/** ~4 chars per token is the usual English rule of thumb — labelled as an estimate everywhere. */
export function estimateTokens(text) {
  return Math.max(1, Math.round((text || "").length / 4));
}

/** The labelled needs-WebGPU state. Never a faked reply — the real reason + how to enable, plus the
 *  ~2.1 GB download so a visitor knows what running a 3B model for real would cost. */
export function webgpuFallbackHTML(probe) {
  const reasons = {
    "no-gpu": "This browser doesn't expose the WebGPU API at all.",
    "no-adapter":
      "WebGPU is present but no GPU adapter is available here (normal in headless Chrome, many VMs, or when the GPU is blocklisted).",
    "adapter-error": "Requesting a WebGPU adapter threw an error.",
  };
  const why = reasons[probe.reason] ?? "WebGPU isn't usable here.";
  return `
    <strong>SmolLM3-3B needs WebGPU — and it isn't available in this browser.</strong>
    <p class="muted" style="margin:.4rem 0">${why}${
    probe.detail ? " (" + escapeHTML(probe.detail) + ")" : ""
  }
    Transformers.js can technically run on a WASM/CPU backend, but a <strong>3-billion-parameter</strong>
    model decoding token-by-token (and, in thinking mode, reasoning first) is impractical there — multiple
    gigabytes and minutes per reply — so this page requires a real GPU and won't fake a reply. The q4f16
    weights are a <strong>~2.1 GB</strong> download (cached after the first load). To run it for real:</p>
    <ul class="muted" style="margin:.2rem 0">
      <li>Open in <strong>Chrome or Edge 113+</strong> on a machine with a capable GPU and enough memory (~3 GB).</li>
      <li>Check <code>chrome://gpu</code> — "WebGPU" should read <em>Hardware accelerated</em>.</li>
      <li>If it's blocklisted, enable <code>chrome://flags/#enable-unsafe-webgpu</code> and relaunch.</li>
      <li>The q4f16 build is fastest on GPUs that report the <code>shader-f16</code> feature (most modern ones).</li>
    </ul>`;
}

export const CHAT_CSS = `
.chat-wrap { display:flex; flex-direction:column; gap:.6rem; }
.chat-log { display:flex; flex-direction:column; gap:.6rem; min-block-size:8rem; max-block-size:30rem;
  overflow-y:auto; padding:.4rem; border:1px solid var(--border); border-radius:8px; background:var(--bg-raised); }
.msg { display:flex; flex-direction:column; gap:.15rem; max-inline-size:92%; min-inline-size:0; }
.msg .who { font-family:var(--font-mono); font-size:.66rem; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); }
.msg .bubble { padding:.55rem .7rem; border-radius:10px; white-space:pre-wrap; line-height:1.6; overflow-wrap:anywhere; min-inline-size:0; }
.msg.user { align-self:flex-end; align-items:flex-end; }
.msg.user .bubble { background:var(--accent); color:var(--accent-ink); }
.msg.assistant { align-self:flex-start; inline-size:100%; }
.msg.assistant .bubble { background:var(--bg-secondary); border:1px solid var(--border); color:var(--color); }
.msg.system { align-self:center; max-inline-size:100%; }
.msg.system .bubble { background:transparent; border:1px dashed var(--border); color:var(--muted); font-size:.82rem; font-style:italic; }
.msg .caret { display:inline-block; inline-size:.5rem; block-size:1em; vertical-align:text-bottom;
  background:currentColor; animation:blink 1s steps(2) infinite; }
@keyframes blink { 50% { opacity:0; } }
@media (prefers-reduced-motion: reduce) { .msg .caret { animation:none; } }
/* SmolLM3 reasoning trace — the model's private "thinking" before it answers. */
.think { border:1px solid var(--border); border-left:3px solid var(--accent); border-radius:8px;
  background:var(--bg-raised); margin:0 0 .35rem; overflow:hidden; }
.think > summary { cursor:pointer; padding:.35rem .6rem; font-size:.74rem; font-family:var(--font-mono);
  letter-spacing:.04em; text-transform:uppercase; color:var(--muted); list-style:none; display:flex;
  align-items:center; gap:.4rem; min-block-size:2.4rem; }
.think > summary::-webkit-details-marker { display:none; }
.think > summary::before { content:"▸"; transition:transform .15s; }
.think[open] > summary::before { transform:rotate(90deg); }
@media (prefers-reduced-motion: reduce) { .think > summary::before { transition:none; } }
.think .think-body { padding:.1rem .7rem .5rem; font-size:.82rem; line-height:1.55; color:var(--muted);
  white-space:pre-wrap; overflow-wrap:anywhere; max-block-size:16rem; overflow:auto; font-style:italic; }
.think .spin { display:inline-block; inline-size:.6rem; block-size:.6rem; border-radius:50%;
  border:2px solid var(--accent); border-top-color:transparent; animation:spin .7s linear infinite; }
@keyframes spin { to { transform:rotate(360deg); } }
@media (prefers-reduced-motion: reduce) { .think .spin { animation:none; } }
.composer { display:flex; gap:.5rem; align-items:flex-end; }
.composer textarea { flex:1 1 auto; field-sizing:content; width:auto; min-inline-size:0; min-block-size:2.6lh; max-block-size:8lh; resize:vertical; }
.composer button { min-block-size:44px; }
.controls { display:flex; flex-wrap:wrap; gap:1rem 1.4rem; margin:.4rem 0 .2rem; }
.control { display:flex; flex-direction:column; gap:.2rem; font-size:.78rem; color:var(--muted); min-inline-size:9rem; }
.control .row { display:flex; align-items:center; gap:.5rem; }
.control input[type=range] { flex:1 1 auto; accent-color:var(--accent); }
.control b { color:var(--color); font-family:var(--font-mono); font-weight:600; min-inline-size:2.6rem; text-align:right; }
.thinktoggle { display:flex; align-items:center; gap:.5rem; flex-wrap:wrap; margin:.2rem 0 .4rem;
  padding:.5rem .65rem; border:1px solid var(--border); border-radius:8px; background:var(--bg-raised); }
.thinktoggle label { display:flex; align-items:center; gap:.45rem; font-size:.85rem; cursor:pointer; min-block-size:2.2rem; }
.thinktoggle input[type=checkbox] { inline-size:1.1rem; block-size:1.1rem; accent-color:var(--accent); }
.thinktoggle .hint { font-size:.76rem; color:var(--muted); }
.sysbox { display:flex; flex-direction:column; gap:.25rem; font-size:.82rem; }
.sysbox textarea { field-sizing:content; width:100%; min-block-size:2.2lh; max-block-size:7lh; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.76rem; color:var(--muted); margin-top:.5rem; }
.readout b { color:var(--color); font-weight:600; }
.fallback { border:1px solid var(--warn); border-radius:var(--radius); background:var(--bg-raised); padding:1rem; }
.fallback code { background:var(--bg-secondary); padding:.05rem .3rem; border-radius:4px; }
.inside-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:.6rem; margin:.4rem 0 .8rem; }
.stat { border:1px solid var(--border); border-radius:8px; background:var(--bg-raised); padding:.5rem .6rem; }
.stat .k { font-size:.68rem; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
.stat .v { font-family:var(--font-mono); font-size:1.05rem; color:var(--color); }
.stat .v small { font-size:.7rem; color:var(--muted); }
.wire { font-family:var(--font-mono); font-size:.76rem; white-space:pre-wrap; word-break:break-word; max-block-size:20rem; overflow:auto; }
.chip { font:inherit; font-size:.78rem; padding:.4rem .7rem; border-radius:999px; border:1px solid var(--border);
  background:var(--bg-raised); color:var(--color); cursor:pointer; min-block-size:40px; }
.chip:hover { border-color:var(--accent); }
.chips { display:flex; flex-wrap:wrap; gap:.4rem; margin:.2rem 0; }
`;
