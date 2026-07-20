// Front-end helpers for the Qwen3-0.6B pages: the worker handshake, streaming plumbing, and shared
// widget CSS. Inference lives in worker.js (Transformers.js text-generation). Every page imports this so
// the chat loop, thinking-mode split, and "see inside" surface stay identical.
//
// Qwen3-0.6B (onnx-community/Qwen3-0.6B-ONNX) is the smallest Qwen3 — a hybrid reasoning model that can
// emit a <think>…</think> chain before its answer. The worker runs the q8 build on the universal
// WebAssembly/CPU path (~618 MB, no GPU); WebGPU is an optional accelerator with a WASM fallback.

const WORKER_URL = "/web-ai-showcase/models/qwen3-0-6b-wasm/worker.js";

export class QwenEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.onProgress = null;
    this._loadWaiters = [];
    this._active = null;
    this._id = 0;
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
          this._active.resolve({
            ms: msg.ms,
            tokens: msg.tokens,
            text: msg.text,
            think: msg.think,
            answer: msg.answer,
            firstTokenMs: msg.firstTokenMs,
            device: msg.device,
          });
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
        } else this._rejectAll(new Error(msg.message));
        break;
    }
  }
  load(onProgress) {
    if (onProgress) this.onProgress = onProgress;
    if (this.ready) return Promise.resolve(this.device);
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load" });
    });
  }
  /** Stream a chat completion. onToken(token,tMs) per token; onPrompt(template) once. opts.thinking toggles reasoning. */
  chat(messages, { onToken, onPrompt, ...opts } = {}) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._active = { id, onToken, onPrompt, resolve, reject };
      this.worker.postMessage({ type: "chat", id, messages, opts });
    });
  }
  topk(messages, k = 12, thinking = true) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._active = { id, resolve, reject };
      this.worker.postMessage({ type: "topk", id, messages, k, thinking });
    });
  }
  stop() {
    this.worker.postMessage({ type: "stop" });
  }
}

export function escapeHTML(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]),
  );
}

export const QWEN_CSS = `
.chat { display:flex; flex-direction:column; gap:.6rem; }
.transcript { display:flex; flex-direction:column; gap:.6rem; max-block-size:26rem; overflow-y:auto;
  padding:.4rem; border:1px solid var(--border); border-radius:var(--radius); background:var(--bg-raised); }
.bubble { padding:.55rem .8rem; border-radius:12px; max-inline-size:88%; min-inline-size:0; white-space:pre-wrap; overflow-wrap:anywhere; line-height:1.6; }
.bubble.user { align-self:flex-end; background:var(--accent); color:var(--accent-ink); border-bottom-right-radius:3px; }
.bubble.assistant { align-self:flex-start; background:var(--bg-secondary); border:1px solid var(--border); border-bottom-left-radius:3px; }
.bubble .role { display:block; font-family:var(--font-mono); font-size:.62rem; text-transform:uppercase; letter-spacing:.08em; opacity:.7; margin-block-end:.2rem; }
.bubble .caret { display:inline-block; inline-size:.5rem; block-size:1rem; background:var(--accent); vertical-align:text-bottom; animation:fblink 1s steps(2) infinite; }
@keyframes fblink { 50% { opacity:0; } }
@media (prefers-reduced-motion: reduce){ .bubble .caret { animation:none; } }
.think { border-inline-start:3px solid var(--border-strong); padding:.35rem .6rem; margin:.3rem 0; background:var(--bg-raised);
  border-radius:6px; font-size:.85rem; color:var(--muted); white-space:pre-wrap; overflow-wrap:anywhere; }
.think summary { cursor:pointer; font-family:var(--font-mono); font-size:.72rem; color:var(--muted); }
.composer { display:flex; gap:.5rem; align-items:flex-end; flex-wrap:wrap; }
.composer textarea { flex:1 1 14rem; min-inline-size:0; resize:vertical; min-block-size:2.6rem; }
.composer button { min-block-size:44px; }
.controls { display:flex; flex-wrap:wrap; gap:1rem 1.4rem; align-items:center; margin:.4rem 0; }
.controls label { display:flex; flex-direction:column; gap:.15rem; font-size:.78rem; color:var(--muted); }
.controls label b { color:var(--color); font-family:var(--font-mono); }
.controls input[type=range] { inline-size:9rem; max-inline-size:100%; accent-color:var(--accent); }
.chips { display:flex; flex-wrap:wrap; gap:.4rem; margin:.2rem 0 .4rem; }
.chip { font-size:.8rem; padding:.5rem .8rem; min-block-size:40px; border:1px solid var(--border); border-radius:999px; background:var(--bg-raised); color:var(--color); cursor:pointer; text-align:start; }
.chip:hover { border-color:var(--border-strong); }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem; color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.tmpl { font-family:var(--font-mono); font-size:.78rem; white-space:pre-wrap; word-break:break-word; background:var(--bg-raised); border:1px solid var(--border); border-radius:var(--radius); padding:.6rem; }
.topk { display:flex; flex-direction:column; gap:.3rem; margin-top:.5rem; }
.topk-row { display:grid; grid-template-columns:minmax(0,9rem) 1fr 3.4rem; gap:.5rem; align-items:center; font-size:.8rem; }
.topk-tok { font-family:var(--font-mono); background:var(--bg-secondary); border:1px solid var(--border); border-radius:4px; padding:.1rem .35rem; white-space:pre; overflow:hidden; text-overflow:ellipsis; }
.topk-bar { block-size:.85rem; background:var(--accent); border-radius:3px; min-inline-size:2px; }
.topk-pct { font-family:var(--font-mono); color:var(--muted); text-align:end; }
`;
