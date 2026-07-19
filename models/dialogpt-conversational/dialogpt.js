// Front-end helpers for the DialoGPT conversational pages. Thin: owns the worker handshake +
// streaming + a couple of renderers. All inference (the decode loop, softmax, top-k, the EOS-joined
// context) runs in worker.js, off the main thread.
//
// DialoGPT's defining mechanism: a multi-turn conversation is fed as ONE sequence with each turn
// separated by the end-of-text token (<|endoftext|>). The `turns` array we pass to the worker is that
// dialogue, in order (optional persona seed turns first, then alternating user/bot, then the new user
// message). The worker joins them with EOS and continues the sequence — its continuation up to the
// next EOS is the reply.

const WORKER_URL = "/web-ai-showcase/models/dialogpt-conversational/worker.js";

export class DialoGPTEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.device = "wasm";
    this.eosStr = "<|endoftext|>";
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
      if (msg.eosStr) this.eosStr = msg.eosStr;
      for (const w of this._loadWaiters) w.resolve(msg.device);
      this._loadWaiters = [];
    } else if (msg.type === "token") {
      this._streams.get(msg.id)?.(msg);
    } else if (msg.type === "done") {
      const p = this._pending.get(msg.id);
      if (p) {
        this._pending.delete(msg.id);
        this._streams.delete(msg.id);
        if (msg.eosStr) this.eosStr = msg.eosStr;
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

  /**
   * Generate the next bot reply for a dialogue.
   * @param {string[]} turns  Ordered turns (persona seeds + alternating user/bot + new user message).
   * @param {object} opts     { maxNew, greedy, temperature, topK }.
   * @param {function} onToken Per-token streaming callback (optional).
   * @returns {Promise<{reply,contextText,promptTokens,replyTokens,ms,device,endedOnEos}>}
   */
  chat(turns, opts = {}, onToken) {
    const id = ++this._id;
    if (onToken) this._streams.set(id, onToken);
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "chat", id, turns, opts });
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

/** Render the EOS-joined context as turn pills separated by a visible <|endoftext|> marker. */
export function renderContext(container, turns, eosStr = "<|endoftext|>") {
  const parts = turns.filter((t) => t != null && String(t).length > 0);
  container.replaceChildren();
  parts.forEach((t, i) => {
    const pill = document.createElement("span");
    pill.className = "ctx-turn";
    pill.textContent = String(t);
    container.append(pill);
    const eos = document.createElement("span");
    eos.className = "ctx-eos";
    eos.textContent = eosStr;
    eos.title = "end-of-text token — the turn separator DialoGPT is trained on";
    container.append(eos);
  });
}

/** Render top-k next-token candidates as probability bars (like the GPT-2 see-inside surface). */
export function renderDist(container, items) {
  const maxP = Math.max(1e-6, ...items.map((i) => i.prob));
  container.replaceChildren(...items.map((it, n) => {
    const row = document.createElement("div");
    row.className = "tok-row" + (it.chosen ? " chosen" : "");
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

export const DIALOGPT_CSS = `
.chat-log { display:flex; flex-direction:column; gap:.5rem; border:1px solid var(--border);
  border-radius:var(--radius); background:var(--bg-raised); padding:.8rem; min-block-size:180px;
  max-block-size:min(52vh, 460px); overflow-y:auto; }
.chat-log:empty::before { content:"Your conversation will appear here."; color:var(--muted); font-size:.92rem; }
.msg { max-inline-size:82%; padding:.5rem .75rem; border-radius:14px; line-height:1.5; white-space:pre-wrap;
  overflow-wrap:anywhere; min-inline-size:0; }
.msg.user { align-self:flex-end; background:var(--accent); color:var(--accent-ink); border-end-end-radius:4px; }
.msg.bot { align-self:flex-start; background:var(--bg-secondary); color:var(--color); border:1px solid var(--border);
  border-end-start-radius:4px; }
.msg.bot.thinking { color:var(--muted); font-style:italic; }
.msg .who { display:block; font-family:var(--font-mono); font-size:.62rem; text-transform:uppercase;
  letter-spacing:.05em; opacity:.7; margin-block-end:.15rem; }
.chat-input-row { display:flex; flex-wrap:wrap; gap:.5rem; align-items:end; margin-top:.7rem; }
.chat-input-row label { flex:1 1 240px; display:flex; flex-direction:column; gap:.3rem; font-size:.82rem; min-inline-size:0; }
.chat-input-row input[type="text"], .chat-input-row textarea { inline-size:100%; font:inherit; padding:.55rem .7rem;
  border-radius:8px; border:1px solid var(--border); background:var(--bg-raised); color:var(--color); }
.chat-actions { display:flex; gap:.5rem; flex-wrap:wrap; }
.controls { display:flex; flex-wrap:wrap; gap:1rem 1.4rem; align-items:center; margin:.7rem 0; }
.controls label { display:flex; align-items:center; gap:.5rem; font-family:var(--font-mono); font-size:.78rem; color:var(--muted); }
.controls output { color:var(--color); font-weight:600; min-inline-size:2.2rem; }
.seg { display:inline-flex; border:1px solid var(--border); border-radius:999px; overflow:hidden; }
.seg button { border:none; border-radius:0; background:var(--bg-raised); color:var(--color); padding:.35rem .85rem; font-size:.8rem; min-block-size:2.2rem; }
.seg button[aria-pressed="true"] { background:var(--accent); color:var(--accent-ink); }
.chip { font:inherit; font-size:.78rem; padding:.25rem .65rem; border-radius:999px; border:1px solid var(--border);
  background:var(--bg-raised); color:var(--color); cursor:pointer; min-block-size:2rem; }
.chip:hover, .chip:focus-visible { border-color:var(--accent); }
.sample-row { display:flex; flex-wrap:wrap; gap:.45rem; margin:.5rem 0; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem; color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.ctx-view { border:1px solid var(--border); border-radius:8px; background:var(--bg-secondary); padding:.6rem .7rem;
  line-height:2; overflow-wrap:anywhere; }
.ctx-turn { background:var(--bg-raised); border:1px solid var(--border); border-radius:6px; padding:.1rem .4rem; }
.ctx-eos { font-family:var(--font-mono); font-size:.72rem; color:var(--accent); font-weight:700; padding:0 .25rem;
  cursor:help; }
.tok-list { display:flex; flex-direction:column; gap:.28rem; margin-top:.5rem; }
.tok-row { display:grid; grid-template-columns:2.2rem 7rem 1fr 3.6rem; align-items:center; gap:.5rem; text-align:start;
  font:inherit; padding:.22rem .35rem; border-radius:6px; border:1px solid transparent; }
.tok-row.chosen { border-color:var(--good); background:color-mix(in srgb, var(--good) 12%, transparent); }
.tok-rank { font-family:var(--font-mono); font-size:.72rem; color:var(--muted); }
.tok-str code { font-size:.82rem; }
.tok-bar { block-size:.7rem; border-radius:999px; background:var(--bg-secondary); border:1px solid var(--border); overflow:hidden; }
.tok-fill { display:block; block-size:100%; background:var(--accent); }
.tok-row.chosen .tok-fill { background:var(--good); }
.tok-num { font-family:var(--font-mono); font-size:.78rem; text-align:end; }
@media (max-width:560px){ .tok-row { grid-template-columns:1.8rem 5rem 1fr 3.2rem; } .msg { max-inline-size:92%; } }
`;
