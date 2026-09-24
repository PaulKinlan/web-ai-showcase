// Front-end helpers for the CodeGen pages. Thin: owns the worker handshake + renderers.
// All inference (the pipeline decode loop) runs in worker.js, off the main thread.

// Must point at THIS family's worker: it loads Xenova/codegen-350M-multi. Pointing at the mono
// worker made every multi page decode with the mono checkpoint while its copy, catalogue entry and
// acceptance record claimed multi (bead web-ai-showcase-utx).
const WORKER_URL = "/web-ai-showcase/models/codegen-350m-multi/worker.js";

export class CodegenEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.device = "wasm";
    this.onProgress = null;
    this._loadWaiters = [];
    this._pending = new Map();
    this._streams = new Map();
    this._id = 0;
    this.worker.addEventListener("message", (e) => this._onMessage(e.data));
    this.worker.addEventListener("error", (e) => {
      const err = new Error(e.message || "Worker failed to start");
      for (const w of this._loadWaiters) w.reject(err);
      this._loadWaiters = [];
      for (const [, p] of this._pending) p.reject(err);
      this._pending.clear();
    });
  }

  _onMessage(msg) {
    if (msg.type === "progress") {
      this.onProgress?.(msg.p);
    } else if (msg.type === "ready") {
      this.device = msg.device;
      for (const w of this._loadWaiters) w.resolve(msg.device);
      this._loadWaiters = [];
    } else if (msg.type === "token") {
      this._streams.get(msg.id)?.(msg);
    } else if (msg.type === "done") {
      const p = this._pending.get(msg.id);
      if (p) {
        this._pending.delete(msg.id);
        this._streams.delete(msg.id);
        p.resolve(msg);
      }
    } else if (msg.type === "error") {
      if (msg.id != null && this._pending.has(msg.id)) {
        this._pending.get(msg.id).reject(new Error(msg.message));
        this._pending.delete(msg.id);
        this._streams.delete(msg.id);
      } else {
        const err = new Error(msg.message);
        for (const w of this._loadWaiters) w.reject(err);
        this._loadWaiters = [];
      }
    }
  }

  load(onProgress) {
    if (onProgress) this.onProgress = onProgress;
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load" });
    });
  }

  /** Stream a completion. opts: { maxNew, greedy, temperature, topK }. onToken(msg) per step. */
  generate(prompt, opts = {}, onToken) {
    const id = ++this._id;
    if (onToken) this._streams.set(id, onToken);
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "generate", id, prompt, opts });
    });
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** A token string for display: make leading spaces and newlines visible. */
export function showToken(t) {
  return escapeHTML(t).replace(/\n/g, "⏎").replace(/^ /, "␣");
}

export const CODEGEN_CSS = `
textarea.prompt, input.prompt { inline-size: 100%; font: inherit; padding: .6rem .7rem; border-radius: 8px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); }
textarea.prompt { min-block-size: 4.2rem; resize: vertical; font-family: var(--font-mono); font-size: .88rem; line-height: 1.55; }
.gen-out { font-size: .92rem; line-height: 1.6; font-family: var(--font-mono); border: 1px solid var(--border);
  border-radius: 8px; background: var(--bg-raised); padding: .7rem .8rem; margin-top: .6rem; min-block-size: 3rem; white-space: pre-wrap; }
.gen-out .prompt-span { color: var(--muted); }
.gen-out .new-span { color: var(--color); }
.controls { display: flex; flex-wrap: wrap; gap: 1rem 1.4rem; align-items: center; margin: .6rem 0; }
/* A native <select> has an intrinsic min-content width, so without these constraints a long option
   label widens the whole page: at a requested 360px viewport the practical pages rendered into 404px
   and the control sat ~84px outside its panel (bead web-ai-showcase-vtk). Constrain rather than clip:
   the picker keeps its native behaviour and the label wraps. */
.controls label { display: flex; align-items: center; gap: .5rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); flex-wrap: wrap; min-inline-size: 0; max-inline-size: 100%; }
.controls :is(input, select) { min-inline-size: 0; max-inline-size: 100%; font-size: 1rem; }
.controls :is(button, input, select) { min-block-size: 44px; }
.controls output { color: var(--color); font-weight: 600; min-inline-size: 2.2rem; }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 999px; overflow: hidden; }
.seg button { border: none; border-radius: 0; background: var(--bg-raised); color: var(--color); padding: .3rem .8rem; font-size: .8rem; }
.seg button[aria-pressed="true"] { background: var(--accent); color: var(--accent-ink); }
.sample-row { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0; }
.chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.chip:hover, .chip:focus-visible { border-color: var(--accent); }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.token-chain { display: flex; flex-wrap: wrap; gap: 2px; margin-top: .5rem; line-height: 1.9; }
.token-chain .t { font-family: var(--font-mono); font-size: .8rem; padding: .05rem .15rem; border-radius: 3px;
  background: color-mix(in srgb, var(--accent) 14%, transparent); }
@media (max-width: 560px){ .token-chain .t { font-size: .72rem; } }
`;
