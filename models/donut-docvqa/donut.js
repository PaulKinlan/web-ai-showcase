// Shared front-end helpers for the Donut document-VQA pages. Owns the worker handshake, turns
// files/samples into data URLs, and streams the answer back to the page. All inference happens off the
// main thread in worker.js.

const WORKER_URL = "/web-ai-showcase/models/donut-docvqa/worker.js";

export class DonutEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this._loadWaiters = [];
    this._pending = new Map();
    this._id = 0;
    this.onProgress = null;
    this.device = "wasm";
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
      this.ready = true;
      this.device = msg.device;
      for (const w of this._loadWaiters) w.resolve(msg.device);
      this._loadWaiters = [];
    } else if (msg.type === "prompt") {
      this._pending.get(msg.id)?.onPrompt?.(msg.prompt);
    } else if (msg.type === "token") {
      this._pending.get(msg.id)?.onToken?.(msg);
    } else if (msg.type === "result") {
      const p = this._pending.get(msg.id);
      if (p) {
        this._pending.delete(msg.id);
        p.resolve(msg);
      }
    } else if (msg.type === "error") {
      if (msg.id != null && this._pending.has(msg.id)) {
        this._pending.get(msg.id).reject(new Error(msg.message));
        this._pending.delete(msg.id);
      } else {
        const err = new Error(msg.message);
        for (const w of this._loadWaiters) w.reject(err);
        this._loadWaiters = [];
      }
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

  /**
   * Ask a question about a document image. Streams the constructed prompt via onPrompt(str) and answer
   * tokens via onToken({token, t, i}); resolves with { id, answer, prompt, tokens, ms, device }.
   */
  ask(imageURL, question, { maxTokens = 64, onPrompt, onToken } = {}) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject, onPrompt, onToken });
      this.worker.postMessage({ type: "run", id, image: imageURL, question, maxTokens });
    });
  }
}

/** Read a File (upload or drop) into a data URL. */
export function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Shared inline styles for the Donut widgets (dropzone, doc preview, answer box, token trace). */
export const DONUT_CSS = `
.dropzone {
  border: 2px dashed var(--border-strong);
  border-radius: var(--radius);
  background: var(--bg-raised);
  padding: 1rem;
  text-align: center;
  cursor: pointer;
  transition: border-color .15s, background .15s;
}
.dropzone.drag { border-color: var(--accent); background: var(--bg-secondary); }
.dropzone:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.sample-strip { display: flex; gap: .5rem; flex-wrap: wrap; margin: .5rem 0; }
.sample-thumb {
  block-size: 68px; max-inline-size: 90px; object-fit: cover; object-position: top;
  border-radius: 6px; border: 2px solid transparent; cursor: pointer; padding: 2px; background: #fff;
}
.sample-thumb.active { border-color: var(--accent); }
.sample-thumb:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.doc-wrap { position: relative; display: inline-block; max-inline-size: 100%; }
.doc-img { max-inline-size: 100%; max-block-size: 360px; border-radius: 8px; display: block;
  background: #fff; border: 1px solid var(--border); }
.answer-box {
  font-family: var(--font-display, Georgia, serif); font-size: 1.3rem; line-height: 1.4;
  padding: .8rem 1rem; border-radius: var(--radius); background: var(--bg-raised);
  border: 1px solid var(--border); min-block-size: 2.4em; margin: .5rem 0; word-break: break-word;
}
.answer-box .cursor { display: inline-block; inline-size: .5ch; background: var(--accent);
  animation: dqblink 1s steps(2) infinite; }
@keyframes dqblink { 50% { opacity: 0; } }
.q-row { display: flex; gap: .5rem; flex-wrap: wrap; margin: .5rem 0; }
.q-row input[type="text"] { flex: 1 1 200px; font: inherit; padding: .4rem .5rem;
  border: 1px solid var(--border); border-radius: 6px; background: var(--bg-raised); color: var(--color); }
.preset-qs { display: flex; gap: .4rem; flex-wrap: wrap; margin: .4rem 0; }
.chip {
  font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer;
}
.chip:hover { border-color: var(--accent); }
.chip[aria-pressed="true"] { border-color: var(--accent); background: var(--bg-secondary); }
.readout {
  display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem;
}
.readout b { color: var(--color); font-weight: 600; }
.token-trace { display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .4rem; }
.token-trace .tok {
  font-family: var(--font-mono); font-size: .74rem; padding: .1rem .4rem; border-radius: 4px;
  background: var(--bg-secondary); border: 1px solid var(--border); white-space: pre;
}
.token-trace .tok small { color: var(--muted); margin-inline-start: .3rem; }
.field-row { display: flex; flex-wrap: wrap; gap: .6rem; align-items: start; margin: .6rem 0; }
.inside-table { inline-size: 100%; border-collapse: collapse; font-size: .82rem; margin-top: .5rem; }
.inside-table th, .inside-table td {
  text-align: left; padding: .3rem .5rem; border-bottom: 1px solid var(--border);
  font-family: var(--font-mono);
}
.inside-table th { color: var(--muted); font-weight: 600; }
.tmpl { font-family: var(--font-mono); font-size: .82rem; background: var(--bg-raised);
  border: 1px solid var(--border); border-radius: var(--radius); padding: .6rem .8rem;
  overflow-x: auto; white-space: pre-wrap; word-break: break-word; margin: .3rem 0; }
.qa-log { display: flex; flex-direction: column; gap: .5rem; margin: .5rem 0; }
.qa-item { border: 1px solid var(--border); border-radius: 8px; padding: .5rem .7rem; background: var(--bg-raised); }
.qa-item .q { font-size: .82rem; color: var(--muted); }
.qa-item .a { font-family: var(--font-display, Georgia, serif); font-size: 1.1rem; }
`;
