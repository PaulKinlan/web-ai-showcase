// Front-end helpers shared by every LFM2-350M page: the worker handshake + token-streaming plumbing,
// the WebGPU adapter probe (to choose the fast q4f16/WebGPU path or the verified fp32/WASM path), and
// the chat widget CSS. All real inference happens in worker.js, which drives the Transformers.js
// pipeline (ONNX). LFM2 is a plain instruct chat model — no hidden reasoning trace — so this is a
// straightforward streaming chat engine.

const WORKER_URL = "/web-ai-showcase/models/lfm2/worker.js";

/** Probe for a real WebGPU adapter. navigator.gpu merely existing is not enough. */
export async function webGPUAdapterAvailable() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) return false;
  try {
    return (await navigator.gpu.requestAdapter()) != null;
  } catch {
    return false;
  }
}

/** Decide the device + dtype pair for this machine. Both are real, verified code paths. */
export async function pickDeviceDtype() {
  if (await webGPUAdapterAvailable()) return { device: "webgpu", dtype: "q4f16", sizeMB: 260 };
  return { device: "wasm", dtype: "fp32", sizeMB: 1400 };
}

export class LFM2Engine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.dtype = "fp32";
    this.onProgress = null;
    this._loadWaiters = [];
    this._active = null;
    this._id = 0;
    this._loadOpts = null;
    this.worker.addEventListener("message", (e) => this._onMessage(e.data));
    this.worker.addEventListener(
      "error",
      (e) => this._rejectAll(new Error(e.message || "Worker failed to start")),
    );
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
        this.dtype = msg.dtype;
        for (const w of this._loadWaiters) w.resolve(msg.device);
        this._loadWaiters = [];
        break;
      case "prompt":
        if (this._active && this._active.id === msg.id) this._active.onPrompt?.(msg.template);
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
            device: msg.device,
            dtype: msg.dtype,
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

  /** Configure the device+dtype the worker should load with (from pickDeviceDtype). */
  configure(opts) {
    this._loadOpts = opts;
  }

  load(onProgress) {
    if (onProgress) this.onProgress = onProgress;
    if (this.ready) return Promise.resolve(this.device);
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load", opts: this._loadOpts });
    });
  }

  /**
   * Stream a chat completion. `messages` = array; `opts` = { temperature, topP, maxTokens, doSample }.
   * onToken(delta) fires per chunk; onFirstToken(ttftMs) once; onPrompt(template) once.
   */
  chat(messages, { onToken, onFirstToken, onPrompt, ...opts } = {}) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._active = { id, onToken, onFirstToken, onPrompt, resolve, reject };
      this.worker.postMessage({ type: "run", id, messages, opts: { ...this._loadOpts, ...opts } });
    });
  }

  stop() {
    this.worker.postMessage({ type: "stop" });
  }
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
.sysbox { display:flex; flex-direction:column; gap:.25rem; font-size:.82rem; }
.sysbox textarea { field-sizing:content; width:100%; min-block-size:2.2lh; max-block-size:7lh; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.76rem; color:var(--muted); margin-top:.5rem; }
.readout b { color:var(--color); font-weight:600; }
.devnote { font-size:.78rem; color:var(--muted); border:1px solid var(--border); border-radius:8px; background:var(--bg-raised); padding:.5rem .65rem; margin:.4rem 0; }
.devnote b { color:var(--color); }
.inside-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:.6rem; margin:.4rem 0 .8rem; }
.stat { border:1px solid var(--border); border-radius:8px; background:var(--bg-raised); padding:.5rem .6rem; }
.stat .k { font-size:.68rem; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
.stat .v { font-family:var(--font-mono); font-size:1.05rem; color:var(--color); }
.stat .v small { font-size:.7rem; color:var(--muted); }
.wire { font-family:var(--font-mono); font-size:.76rem; white-space:pre-wrap; word-break:break-word; max-block-size:20rem; overflow:auto; background:var(--bg-raised); border:1px solid var(--border); border-radius:8px; padding:.5rem; }
.chip { font:inherit; font-size:.78rem; padding:.4rem .7rem; border-radius:999px; border:1px solid var(--border);
  background:var(--bg-raised); color:var(--color); cursor:pointer; min-block-size:40px; }
.chip:hover { border-color:var(--accent); }
.chips { display:flex; flex-wrap:wrap; gap:.4rem; margin:.2rem 0; }
`;
