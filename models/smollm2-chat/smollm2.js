// Shared front-end helpers for the SmolLM2-360M-Instruct pages: the worker handshake, streaming
// plumbing, a real WebGPU probe (used only to label the backend), and the widget CSS. All real
// inference lives in worker.js. Unlike a WebGPU-only model, SmolLM2-360M has a genuine WASM path,
// so the worker auto-picks WebGPU (q4f16) when a real adapter exists and WebAssembly (q8) otherwise —
// the page auto-initialises either way (requiresWebGPU:false) and reports the backend actually used.

const WORKER_URL = "/web-ai-showcase/models/smollm2-chat/worker.js";

export class SmolLMEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = null;
    this.onProgress = null;
    this._loadWaiters = [];
    this._probeWaiters = [];
    this._active = null; // { id, kind, onToken, onPrompt, resolve, reject }
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
      case "probe-result":
        for (const w of this._probeWaiters) w.resolve(msg.gpu);
        this._probeWaiters = [];
        break;
      case "ready":
        this.ready = true;
        this.device = msg.device;
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
          this._active.resolve({ ms: msg.ms, tokens: msg.tokens, text: msg.text });
          this._active = null;
        }
        break;
      case "topk-result":
        if (this._active && this._active.id === msg.id) {
          this._active.resolve({ prompt: msg.prompt, tokens: msg.tokens });
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

  /** Load the model. The worker auto-picks webgpu (q4f16) or wasm (q8) unless opts forces one. */
  load(onProgress, opts = {}) {
    if (onProgress) this.onProgress = onProgress;
    if (this.ready) return Promise.resolve(this.device);
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load", device: opts.device, dtype: opts.dtype });
    });
  }

  /** Stream a chat completion. onToken(token, tMs) fires per token; onPrompt(template) once. */
  chat(messages, { onToken, onPrompt, ...opts } = {}) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._active = { id, onToken, onPrompt, resolve, reject };
      this.worker.postMessage({ type: "chat", id, messages, opts });
    });
  }

  /** Real top-k next-token distribution for the current context (one forward pass). */
  topk(messages, k = 12) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._active = { id, resolve, reject };
      this.worker.postMessage({ type: "topk", id, messages, k });
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

export const SMOLLM_CSS = `
.chat { display:flex; flex-direction:column; gap:.6rem; }
.transcript { display:flex; flex-direction:column; gap:.6rem; max-block-size:26rem; overflow-y:auto;
  padding:.4rem; border:1px solid var(--border); border-radius:var(--radius); background:var(--bg-raised); }
.bubble { padding:.55rem .8rem; border-radius:12px; max-inline-size:85%; white-space:pre-wrap; line-height:1.6; }
.bubble.user { align-self:flex-end; background:var(--accent); color:var(--accent-ink); border-bottom-right-radius:3px; }
.bubble.assistant { align-self:flex-start; background:var(--bg-secondary); border:1px solid var(--border); border-bottom-left-radius:3px; }
.bubble .role { display:block; font-family:var(--font-mono); font-size:.62rem; text-transform:uppercase;
  letter-spacing:.08em; opacity:.7; margin-block-end:.2rem; }
.bubble .caret { display:inline-block; inline-size:.5rem; block-size:1rem; background:var(--accent);
  vertical-align:text-bottom; animation:sblink 1s steps(2) infinite; }
@keyframes sblink { 50% { opacity:0; } }
@media (prefers-reduced-motion: reduce) { .bubble .caret { animation:none; } }
.composer { display:flex; gap:.5rem; align-items:flex-end; }
.composer textarea { flex:1 1 auto; resize:vertical; min-block-size:2.6rem; }
.controls { display:flex; flex-wrap:wrap; gap:1rem 1.4rem; align-items:center; margin:.4rem 0; }
.controls label { display:flex; flex-direction:column; gap:.15rem; font-size:.78rem; color:var(--muted); }
.controls label b { color:var(--color); font-family:var(--font-mono); }
.controls input[type=range] { inline-size:9rem; accent-color:var(--accent); }
.sysrow { display:flex; flex-direction:column; gap:.25rem; font-size:.82rem; margin-block-end:.4rem; }
.sysrow textarea { resize:vertical; min-block-size:2.4rem; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem;
  color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.tmpl { font-family:var(--font-mono); font-size:.78rem; white-space:pre-wrap; word-break:break-word; }
.topk { display:flex; flex-direction:column; gap:.3rem; margin-top:.5rem; }
.topk-row { display:grid; grid-template-columns:9rem 1fr 3.4rem; gap:.5rem; align-items:center; font-size:.8rem; }
.topk-tok { font-family:var(--font-mono); background:var(--bg-secondary); border:1px solid var(--border);
  border-radius:4px; padding:.1rem .35rem; white-space:pre; overflow:hidden; text-overflow:ellipsis; }
.topk-bar { block-size:.85rem; background:var(--accent); border-radius:3px; min-inline-size:2px; }
.topk-pct { font-family:var(--font-mono); color:var(--muted); text-align:end; }
.chip { font:inherit; font-size:.78rem; padding:.2rem .6rem; border-radius:999px; border:1px solid var(--border);
  background:var(--bg-raised); color:var(--color); cursor:pointer; }
.chip:hover { border-color:var(--accent); }
.chips { display:flex; flex-wrap:wrap; gap:.4rem; margin:.2rem 0; }
.wire { font-family:var(--font-mono); font-size:.76rem; white-space:pre-wrap; word-break:break-word;
  max-block-size:20rem; overflow:auto; padding:.5rem; border:1px solid var(--border); border-radius:8px;
  background:var(--bg-raised); }
.msg { display:flex; flex-direction:column; gap:.15rem; max-inline-size:100%; }
.msg .bubble { padding:.55rem .7rem; border-radius:10px; white-space:pre-wrap; line-height:1.6; overflow-wrap:anywhere; }
.msg.assistant .bubble { background:var(--bg-secondary); border:1px solid var(--border); color:var(--color); }
`;
