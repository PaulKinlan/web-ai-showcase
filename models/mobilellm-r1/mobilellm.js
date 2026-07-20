// Front-end helpers for the MobileLLM-R1 360M pages: the worker handshake, streaming plumbing, and the
// shared widget CSS. Inference lives in worker.js (Transformers.js text-generation). Every page imports
// this so the chat loop and "see inside" surface stay identical.
//
// MobileLLM-R1-360M is Meta's 2025 on-device REASONING model (arXiv:2509.24945): a sub-billion model
// post-trained for math, code, and science that reasons step by step inside <think>…</think> and gives
// its final answer after. The ONNX build (onnx-community/MobileLLM-R1-360M-ONNX) uses Meta's
// `llama4_text` architecture, which the repo's shared Transformers.js 3.7.5 already registers — so this
// page runs on the universal WebAssembly/CPU path (dtype q4), no WebGPU required and no version pin.

const WORKER_URL = "/web-ai-showcase/models/mobilellm-r1/worker.js";

// The Llama-4 tokenizer carries a `<image>` placeholder token that this text-only checkpoint can emit
// spuriously on short/off-domain prompts; it isn't registered as a special token so skip_special_tokens
// leaves it in. It has no meaning here, so we strip that one placeholder for display (honest cleanup of
// a documented tokenizer artifact — never a rewrite of the model's actual words).
export function stripArtifacts(s) {
  return String(s).replaceAll("<image>", "");
}

export class MobileLLMEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm";
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
        } else {
          this._rejectAll(new Error(msg.message));
        }
        break;
    }
  }

  /** Load the model on the WASM/CPU path (dtype q4). Resolves with the device string. */
  load(onProgress) {
    if (onProgress) this.onProgress = onProgress;
    if (this.ready) return Promise.resolve(this.device);
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load" });
    });
  }

  /** Stream a completion. onToken(token, tMs) fires per token; onPrompt(template) once. */
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

// Render a reasoning transcript: the <think>…</think> trace is styled as a muted "reasoning" block, the
// rest as the answer. Pure display — the underlying text is the model's real output (artifact-stripped).
export function renderReasoning(raw) {
  const clean = stripArtifacts(raw);
  const m = clean.match(/^([\s\S]*?)<\/think>([\s\S]*)$/);
  if (!m) return escapeHTML(clean);
  const think = m[1].replace(/^<think>/, "").trim();
  const answer = m[2].trim();
  let html = "";
  if (think) {
    html += `<div class="think"><span class="think-label">reasoning</span>${
      escapeHTML(think)
    }</div>`;
  }
  if (answer) html += `<div class="answer">${escapeHTML(answer)}</div>`;
  return html || escapeHTML(clean);
}

export const MOBILELLM_CSS = `
.chat { display:flex; flex-direction:column; gap:.6rem; }
.transcript { display:flex; flex-direction:column; gap:.6rem; max-block-size:30rem; overflow-y:auto;
  padding:.4rem; border:1px solid var(--border); border-radius:var(--radius); background:var(--bg-raised); }
.bubble { padding:.55rem .8rem; border-radius:12px; max-inline-size:88%; min-inline-size:0;
  white-space:pre-wrap; overflow-wrap:anywhere; line-height:1.6; }
.bubble.user { align-self:flex-end; background:var(--accent); color:var(--accent-ink); border-bottom-right-radius:3px; }
.bubble.assistant { align-self:flex-start; background:var(--bg-secondary); border:1px solid var(--border); border-bottom-left-radius:3px; }
.bubble .role { display:block; font-family:var(--font-mono); font-size:.62rem; text-transform:uppercase;
  letter-spacing:.08em; opacity:.7; margin-block-end:.2rem; }
.bubble .caret { display:inline-block; inline-size:.5rem; block-size:1rem; background:var(--accent);
  vertical-align:text-bottom; animation:fblink 1s steps(2) infinite; }
@keyframes fblink { 50% { opacity:0; } }
@media (prefers-reduced-motion: reduce) { .bubble .caret { animation:none; } }
.think { border-inline-start:3px solid var(--border-strong); padding:.3rem 0 .3rem .6rem; margin-block-end:.5rem;
  color:var(--muted); font-size:.92em; white-space:pre-wrap; }
.think-label { display:block; font-family:var(--font-mono); font-size:.6rem; text-transform:uppercase;
  letter-spacing:.08em; opacity:.7; margin-block-end:.2rem; }
.answer { white-space:pre-wrap; }
.composer { display:flex; gap:.5rem; align-items:flex-end; flex-wrap:wrap; }
.composer textarea { flex:1 1 14rem; min-inline-size:0; resize:vertical; min-block-size:2.6rem; }
.composer button { min-block-size:44px; }
.controls { display:flex; flex-wrap:wrap; gap:1rem 1.4rem; align-items:center; margin:.4rem 0; }
.controls label { display:flex; flex-direction:column; gap:.15rem; font-size:.78rem; color:var(--muted); }
.controls label b { color:var(--color); font-family:var(--font-mono); }
.controls input[type=range] { inline-size:9rem; max-inline-size:100%; accent-color:var(--accent); }
.chips { display:flex; flex-wrap:wrap; gap:.4rem; margin:.2rem 0 .4rem; }
.chip { font-size:.8rem; padding:.5rem .8rem; min-block-size:40px; border:1px solid var(--border);
  border-radius:999px; background:var(--bg-raised); color:var(--color); cursor:pointer; text-align:start; }
.chip:hover { border-color:var(--border-strong); }
.sysrow { display:flex; flex-direction:column; gap:.25rem; font-size:.82rem; margin-block-end:.4rem; }
.sysrow textarea { resize:vertical; min-block-size:2.4rem; inline-size:100%; }
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
`;
