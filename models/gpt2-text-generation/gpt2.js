// Front-end helpers for the GPT-2 pages. Thin: owns the worker handshake + a couple of renderers.
// All inference (the decode loop, softmax, top-k) runs in worker.js, off the main thread.

const WORKER_URL = "/web-ai-showcase/models/gpt2-text-generation/worker.js";

export class GPT2Engine {
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
    } else if (msg.type === "done" || msg.type === "dist") {
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

  /** Next-token distribution for the current text → { topk:[{token,prob,logit}], ms, device } */
  distribution(text, topK = 12) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "distribution", id, text, topK });
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

/** Render a list of {token, prob, chosen?} as probability bars. onPick(token) optional (keyboard+click). */
export function renderDist(container, items, onPick) {
  const maxP = Math.max(1e-6, ...items.map((i) => i.prob));
  container.replaceChildren(...items.map((it, n) => {
    const row = document.createElement(onPick ? "button" : "div");
    row.className = "tok-row" + (it.chosen ? " chosen" : "");
    if (onPick) {
      row.type = "button";
      row.addEventListener("click", () => onPick(it.token, it));
    }
    const rank = document.createElement("span");
    rank.className = "tok-rank";
    rank.textContent = `#${n + 1}`;
    const tok = document.createElement("span");
    tok.className = "tok-str";
    tok.innerHTML = `<code>${showToken(it.token)}</code>`;
    const barWrap = document.createElement("span");
    barWrap.className = "tok-bar";
    const bar = document.createElement("span");
    bar.className = "tok-fill";
    bar.style.inlineSize = `${(it.prob / maxP) * 100}%`;
    barWrap.append(bar);
    const num = document.createElement("span");
    num.className = "tok-num";
    num.textContent = `${(it.prob * 100).toFixed(1)}%`;
    row.append(rank, tok, barWrap, num);
    return row;
  }));
}

export const GPT2_CSS = `
textarea.prompt, input.prompt { inline-size: 100%; font: inherit; padding: .6rem .7rem; border-radius: 8px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); }
textarea.prompt { min-block-size: 3.4rem; resize: vertical; }
.gen-out { font-size: 1.08rem; line-height: 1.7; border: 1px solid var(--border); border-radius: 8px;
  background: var(--bg-raised); padding: .7rem .8rem; margin-top: .6rem; min-block-size: 3rem; white-space: pre-wrap; }
.gen-out .prompt-span { color: var(--muted); }
.gen-out .new-span { color: var(--color); }
.gen-out .cursor { display:inline-block; inline-size:.5rem; }
.controls { display: flex; flex-wrap: wrap; gap: 1rem 1.4rem; align-items: center; margin: .6rem 0; }
.controls label { display: flex; align-items: center; gap: .5rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); }
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
.tok-list { display: flex; flex-direction: column; gap: .28rem; margin-top: .5rem; }
.tok-row { display: grid; grid-template-columns: 2.2rem 7rem 1fr 3.6rem; align-items: center; gap: .5rem;
  text-align: start; font: inherit; padding: .22rem .35rem; border-radius: 6px; border: 1px solid transparent; background: transparent; color: var(--color); }
button.tok-row { cursor: pointer; }
button.tok-row:hover, button.tok-row:focus-visible { border-color: var(--accent); background: var(--bg-raised); }
.tok-row.chosen { border-color: var(--good); background: color-mix(in srgb, var(--good) 12%, transparent); }
.tok-rank { font-family: var(--font-mono); font-size: .72rem; color: var(--muted); }
.tok-str code { font-size: .82rem; }
.tok-bar { block-size: .7rem; border-radius: 999px; background: var(--bg-secondary); border: 1px solid var(--border); overflow: hidden; }
.tok-fill { display: block; block-size: 100%; background: var(--accent); }
.tok-row.chosen .tok-fill { background: var(--good); }
.tok-num { font-family: var(--font-mono); font-size: .78rem; text-align: end; }
.trace { display: flex; flex-wrap: wrap; gap: 2px; margin-top: .5rem; }
.trace .t { font-family: var(--font-mono); font-size: .82rem; padding: .05rem .15rem; border-radius: 3px; }
.chain { font-size: 1.08rem; line-height: 1.9; }
.chain .t { border-radius: 3px; padding: 0 .1rem; }
.chain .picked { background: color-mix(in srgb, var(--accent) 18%, transparent); }
@media (max-width: 560px){ .tok-row { grid-template-columns: 1.8rem 5rem 1fr 3.2rem; } }
`;
