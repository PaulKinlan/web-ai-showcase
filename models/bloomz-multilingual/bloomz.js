// Front-end helpers for the BLOOMZ-560m pages: the worker handshake, streaming plumbing, and shared
// widget CSS. Inference lives in worker.js (Transformers.js text-generation). Every BLOOMZ page imports
// this so the prompt→completion loop and "see inside" surface stay identical.
//
// BLOOMZ-560m (Xenova/bloomz-560m) is BigScience's multitask-finetuned BLOOM — a DISTINCT decoder
// family (ALiBi, 250k multilingual vocab) that follows instructions across 46 languages zero-shot from
// a PLAIN prompt (no chat template). The worker runs the q8 legacy-layout build on the universal
// WebAssembly/CPU path (~350 MB); WebGPU is an optional accelerator with a WASM fallback.

const WORKER_URL = "/web-ai-showcase/models/bloomz-multilingual/worker.js";

export class BloomzEngine {
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

  /** Stream a completion for a plain prompt. onToken(token, tMs) per token; onPrompt(prompt) once. */
  complete(prompt, { onToken, onPrompt, ...opts } = {}) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._active = { id, onToken, onPrompt, resolve, reject };
      this.worker.postMessage({ type: "complete", id, prompt, opts });
    });
  }

  /** Real top-k next-token distribution for a prompt (one forward pass). */
  topk(prompt, k = 12) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._active = { id, resolve, reject };
      this.worker.postMessage({ type: "topk", id, prompt, k });
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

export const BLOOMZ_CSS = `
.io { display:flex; flex-direction:column; gap:.6rem; }
.prompt-box { inline-size:100%; min-block-size:4rem; resize:vertical; }
.completion { white-space:pre-wrap; overflow-wrap:anywhere; line-height:1.6; padding:.7rem .9rem;
  border:1px solid var(--border); border-radius:var(--radius); background:var(--bg-raised); min-block-size:2.5rem; }
.completion .prompt-echo { color:var(--muted); }
.completion .caret { display:inline-block; inline-size:.5rem; block-size:1rem; background:var(--accent);
  vertical-align:text-bottom; animation:fblink 1s steps(2) infinite; }
@keyframes fblink { 50% { opacity:0; } }
@media (prefers-reduced-motion: reduce) { .completion .caret { animation:none; } }
.composer { display:flex; gap:.5rem; align-items:center; flex-wrap:wrap; }
.composer button { min-block-size:44px; }
.controls { display:flex; flex-wrap:wrap; gap:1rem 1.4rem; align-items:center; margin:.4rem 0; }
.controls label { display:flex; flex-direction:column; gap:.15rem; font-size:.78rem; color:var(--muted); }
.controls label b { color:var(--color); font-family:var(--font-mono); }
.controls input[type=range] { inline-size:9rem; max-inline-size:100%; accent-color:var(--accent); }
.chips { display:flex; flex-wrap:wrap; gap:.4rem; margin:.2rem 0 .4rem; }
.chip { font-size:.8rem; padding:.5rem .8rem; min-block-size:40px; border:1px solid var(--border);
  border-radius:999px; background:var(--bg-raised); color:var(--color); cursor:pointer; text-align:start; }
.chip:hover { border-color:var(--border-strong); }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem;
  color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.tmpl { font-family:var(--font-mono); font-size:.78rem; white-space:pre-wrap; word-break:break-word;
  background:var(--bg-raised); border:1px solid var(--border); border-radius:var(--radius); padding:.6rem; }
.topk { display:flex; flex-direction:column; gap:.3rem; margin-top:.5rem; }
.topk-row { display:grid; grid-template-columns:minmax(0,9rem) 1fr 3.4rem; gap:.5rem; align-items:center; font-size:.8rem; }
.topk-tok { font-family:var(--font-mono); background:var(--bg-secondary); border:1px solid var(--border);
  border-radius:4px; padding:.1rem .35rem; white-space:pre; overflow:hidden; text-overflow:ellipsis; }
.topk-bar { block-size:.85rem; background:var(--accent); border-radius:3px; min-inline-size:2px; }
.topk-pct { font-family:var(--font-mono); color:var(--muted); text-align:end; }
.lang-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,220px),1fr)); gap:.8rem; margin:.6rem 0; }
`;
