// Front-end helpers shared by every OLMo-2 1B (WebLLM) page: the worker handshake + streaming
// plumbing, the honest WebGPU gate/fallback, and the chat widget CSS. All real inference happens in
// worker.js, which drives lib/webllm.js (MLC's WebGPU engine). We re-export the canonical
// webGPUAdapterAvailable() so pages gate on the exact helper the task mandates.
//
// What makes OLMo-2 distinct in this catalogue: it is the FULLY OPEN model. Most "open" LLMs release
// only the final weights. Ai2 (Allen Institute for AI) releases the whole recipe for OLMo-2 — the
// pre-training data (Dolma), the mid-training data (Dolmino), the training code (OLMo-core), every
// intermediate checkpoint, and the training logs — under Apache-2.0. That reproducibility is the
// story these pages tell, alongside a genuine, private, on-device chat.

export { webGPUAdapterAvailable } from "/web-ai-showcase/lib/webllm.js";

const WORKER_URL = "/web-ai-showcase/models/olmo-2-webllm/worker.js";

export class OlmoChatEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.onProgress = null;
    this._loadWaiters = [];
    this._active = null; // { id, onToken, onFirstToken, resolve, reject }
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
      this.worker.postMessage({ type: "load" });
    });
  }

  /**
   * Stream a chat completion. `req` = { messages, temperature, top_p, max_tokens }.
   * onToken(delta) fires per answer chunk; onFirstToken(ttftMs) once, at the first token.
   * Resolves with { ms, ttft, chunks, stats }.
   */
  chat(req, { onToken, onFirstToken } = {}) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._active = { id, onToken, onFirstToken, resolve, reject };
      this.worker.postMessage({ type: "run", id, req });
    });
  }

  /** Cooperative interrupt — WebLLM stops the decode loop and the current chat() resolves. */
  stop() {
    this.worker.postMessage({ type: "stop" });
  }
}

/**
 * OLMo-2 Instruct has a REAL system role (its chat template supports system/user/assistant turns), so
 * — unlike Gemma/Mistral — no system-prompt folding is needed. Build the messages array WebLLM
 * receives. Display THIS as the wire so the "see inside" surface is honest about exactly what the
 * model is given.
 */
export function buildMessages(system, history, userText) {
  const msgs = [];
  const sys = (system || "").trim();
  if (sys) msgs.push({ role: "system", content: sys });
  for (const m of history) msgs.push({ role: m.role, content: m.content });
  msgs.push({ role: "user", content: userText });
  return msgs;
}

/** Detailed WebGPU probe used only to label the honest fallback (the gate itself uses
 *  webGPUAdapterAvailable()). Returns { ok, reason?, detail?, shaderF16? }. */
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
 *  ~845 MB download so a visitor knows what running it for real would cost. */
export function webgpuFallbackHTML(probe) {
  const reasons = {
    "no-gpu": "This browser doesn't expose the WebGPU API at all.",
    "no-adapter":
      "WebGPU is present but no GPU adapter is available here (normal in headless Chrome, many VMs, or when the GPU is blocklisted).",
    "adapter-error": "Requesting a WebGPU adapter threw an error.",
  };
  const why = reasons[probe.reason] ?? "WebGPU isn't usable here.";
  return `
    <strong>OLMo-2 1B runs on WebLLM, which needs WebGPU — and it isn't available in this browser.</strong>
    <p class="muted" style="margin:.4rem 0">${why}${
    probe.detail ? " (" + escapeHTML(probe.detail) + ")" : ""
  }
    WebLLM has <em>no WASM fallback</em>: a 1B model decoding token-by-token needs a real GPU, and the
    weights are an <strong>~845 MB</strong> download (cached after the first load) needing roughly
    1.8 GB of GPU memory. So this page won't fake a reply. To run it for real:</p>
    <ul class="muted" style="margin:.2rem 0">
      <li>Open in <strong>Chrome or Edge 113+</strong> on a machine with a GPU (desktop, or a recent laptop/phone).</li>
      <li>Check <code>chrome://gpu</code> — "WebGPU" should read <em>Hardware accelerated</em>.</li>
      <li>If it's blocklisted, enable <code>chrome://flags/#enable-unsafe-webgpu</code> and relaunch.</li>
      <li>The q4f16_1 build is fastest on GPUs that report the <code>shader-f16</code> feature (most modern ones).</li>
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
.composer { display:flex; gap:.5rem; align-items:flex-end; }
.composer textarea { flex:1 1 auto; field-sizing:content; width:auto; min-inline-size:0; min-block-size:2.6lh; max-block-size:8lh; resize:vertical; }
.composer button { min-block-size:44px; }
.controls { display:flex; flex-wrap:wrap; gap:1rem 1.4rem; margin:.4rem 0 .2rem; }
.control { display:flex; flex-direction:column; gap:.2rem; font-size:.78rem; color:var(--muted); min-inline-size:9rem; }
.control .row { display:flex; align-items:center; gap:.5rem; }
.control input[type=range] { flex:1 1 auto; accent-color:var(--accent); }
.control b { color:var(--color); font-family:var(--font-mono); font-weight:600; min-inline-size:2.6rem; text-align:right; }
.sysbox { display:flex; flex-direction:column; gap:.25rem; font-size:.82rem; margin:.2rem 0 .4rem; }
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
.chip{font:inherit;font-size:.78rem;padding:.4rem .7rem;border-radius:999px;border:1px solid var(--border);background:var(--bg-raised);color:var(--color);cursor:pointer;min-block-size:2.4rem}
.chip:hover{border-color:var(--accent)}
/* Openness ledger — OLMo-2's distinguishing property: the whole recipe is released, not just weights. */
.ledger { display:grid; grid-template-columns:1fr auto auto; gap:.1rem .8rem; align-items:center;
  border:1px solid var(--border); border-radius:8px; overflow:hidden; font-size:.84rem; }
.ledger .lrow { display:contents; }
.ledger .lrow > * { padding:.4rem .6rem; border-block-start:1px solid var(--border); }
.ledger .lhead > * { border-block-start:0; font-family:var(--font-mono); font-size:.68rem; text-transform:uppercase;
  letter-spacing:.05em; color:var(--muted); background:var(--bg-raised); }
.ledger .lart { min-inline-size:0; overflow-wrap:anywhere; }
.ledger .lcell { text-align:center; font-family:var(--font-mono); font-weight:600; }
.ledger .yes { color:var(--good); }
.ledger .no { color:var(--muted); }
@media (max-width:520px){ .ledger{ font-size:.78rem; } .ledger .lrow > *{ padding:.35rem .4rem; } }
`;
