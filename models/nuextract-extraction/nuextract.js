// Front-end helpers for the NuExtract structured-extraction pages. Keeps each page thin: worker
// handshake, streaming plumbing, and shared widget CSS. All generation lives in worker.js (off-thread).

const WORKER_URL = "/web-ai-showcase/models/nuextract-extraction/worker.js";

export class NuExtractEngine {
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
      const err = new Error(e.message || "Worker failed to start");
      for (const w of this._loadWaiters) w.reject(err);
      this._loadWaiters = [];
      if (this._active) {
        this._active.reject(err);
        this._active = null;
      }
    });
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
        if (this._active && this._active.id === msg.id) this._active.onPrompt?.(msg.prompt);
        break;
      case "token":
        if (this._active && this._active.id === msg.id) this._active.onToken?.(msg.token);
        break;
      case "result":
        if (this._active && this._active.id === msg.id) {
          this._active.resolve(msg);
          this._active = null;
        }
        break;
      case "error":
        if (this._active && this._active.id === msg.id) {
          this._active.reject(new Error(msg.message));
          this._active = null;
        } else {
          for (const w of this._loadWaiters) w.reject(new Error(msg.message));
          this._loadWaiters = [];
        }
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

  /** Extract structured JSON. onToken(token) streams; onPrompt(prompt) fires once.
   *  Resolves with { prompt, raw, parsed, valid, tokens, ms, device }. */
  extract(template, text, { maxTokens, onToken, onPrompt } = {}) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._active = { id, onToken, onPrompt, resolve, reject };
      this.worker.postMessage({ type: "run", id, template, text, maxTokens });
    });
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Pretty-print a value as indented JSON if possible, else return the raw string. */
export function prettyJSON(value) {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export const NUEXTRACT_CSS = `
.extract-grid { display:grid; gap:1rem; grid-template-columns:repeat(auto-fit, minmax(min(100%, 20rem), 1fr)); }
.field { display:flex; flex-direction:column; gap:.3rem; min-inline-size:0; }
.field > label { font-size:.82rem; font-weight:600; color:var(--muted); }
.field textarea { inline-size:100%; min-inline-size:0; resize:vertical; font-family:var(--font-mono);
  font-size:.85rem; line-height:1.5; min-block-size:8rem; box-sizing:border-box; }
.field textarea.text-in { font-family:var(--font-body); }
.chips { display:flex; flex-wrap:wrap; gap:.4rem; margin:.3rem 0 .5rem; }
.chip { font:inherit; font-size:.8rem; padding:.5rem .8rem; min-block-size:40px; border-radius:999px;
  border:1px solid var(--border); background:var(--bg-raised); color:var(--color); cursor:pointer; }
.chip:hover { border-color:var(--accent); }
.chip[aria-pressed="true"] { border-color:var(--accent); background:var(--bg-secondary); }
.actions { display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; margin:.6rem 0; }
.actions button { min-block-size:44px; }
.json-out { font-family:var(--font-mono); font-size:.9rem; line-height:1.5; white-space:pre-wrap;
  overflow-wrap:anywhere; padding:.8rem 1rem; border-radius:var(--radius); background:var(--bg-raised);
  border:1px solid var(--border); min-block-size:3rem; margin:.4rem 0; }
.json-out.valid { border-color:var(--accent); }
.json-out.invalid { border-color:var(--bad,#d33); }
.badge { display:inline-flex; align-items:center; gap:.35rem; font-size:.74rem; font-family:var(--font-mono);
  padding:.15rem .5rem; border-radius:999px; border:1px solid var(--border); }
.badge.ok { color:var(--good,#0a0); border-color:color-mix(in srgb,var(--good,#0a0) 45%,transparent); }
.badge.no { color:var(--bad,#d33); border-color:color-mix(in srgb,var(--bad,#d33) 45%,transparent); }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem;
  color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.tmpl { font-family:var(--font-mono); font-size:.78rem; white-space:pre-wrap; overflow-wrap:anywhere;
  background:var(--bg-raised); border:1px solid var(--border); border-radius:var(--radius); padding:.6rem;
  max-block-size:20rem; overflow-y:auto; }
.field-table { inline-size:100%; border-collapse:collapse; font-size:.85rem; margin-top:.4rem; }
.field-table th, .field-table td { text-align:left; padding:.3rem .5rem; border-bottom:1px solid var(--border);
  vertical-align:top; font-family:var(--font-mono); overflow-wrap:anywhere; }
.field-table th { color:var(--muted); font-weight:600; }
`;
