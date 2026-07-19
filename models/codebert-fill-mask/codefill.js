// Front-end helpers for the CodeBERTa code fill-mask pages. Thin: owns the worker handshake + renderers.
// All inference (masked-LM logits over code, softmax, tokenization view) lives in worker.js, off the
// main thread. The mask marker is the RoBERTa <mask> token (not BERT's [MASK]).

const WORKER_URL = "/web-ai-showcase/models/codebert-fill-mask/worker.js";
export const MASK = "<mask>";

export class CodeFillEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.onProgress = null;
    this._loadWaiters = [];
    this._pending = new Map();
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
      this.ready = true;
      this.device = msg.device;
      for (const w of this._loadWaiters) w.resolve(msg.device);
      this._loadWaiters = [];
    } else if (msg.type === "fill" || msg.type === "fillMany" || msg.type === "tokens") {
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

  _call(payload) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...payload, id });
    });
  }

  /** Fill one code snippet → { text, masks:[{pos, predictions:[{token,logit,prob}]}], maskCount, ms, device } */
  fill(text, topk = 8) {
    return this._call({ type: "fill", text, topk });
  }

  /** Fill a batch of snippets → { results:[{text,masks,maskCount}], ms, device } */
  fillMany(texts, topk = 8) {
    return this._call({ type: "fillMany", texts, topk });
  }

  /** Byte-level-BPE tokenization view → { tokens:[{id,raw,special,space,text}], count, device } */
  tokenize(text) {
    return this._call({ type: "tokenize", text });
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

export function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

/** Shannon entropy (bits) approximated from the returned top-k probs. Low = confident, high = hedging. */
export function topkEntropy(predictions) {
  let h = 0;
  for (const p of predictions) if (p.prob > 0) h -= p.prob * Math.log2(p.prob);
  return h;
}

/**
 * Render one mask's ranked code-token predictions as probability bars with the exact logit alongside.
 * `onPick` (optional) fires with the chosen token string when a row is activated (keyboard + click).
 */
export function renderPredictions(container, predictions, onPick) {
  const maxProb = Math.max(1e-6, ...predictions.map((p) => p.prob));
  container.replaceChildren(...predictions.map((p, i) => {
    const row = document.createElement(onPick ? "button" : "div");
    row.className = "pred-row";
    if (onPick) {
      row.type = "button";
      row.addEventListener("click", () => onPick(p.token, p));
    }
    const rank = document.createElement("span");
    rank.className = "pred-rank";
    rank.textContent = `#${i + 1}`;
    const tok = document.createElement("span");
    tok.className = "pred-tok";
    // Show a leading-space token visibly (e.g. "␣len") so code identifiers read correctly.
    tok.textContent = p.token.replace(/^ /, "␣");
    const barWrap = document.createElement("span");
    barWrap.className = "pred-bar";
    const bar = document.createElement("span");
    bar.className = "pred-fill";
    bar.style.inlineSize = `${(p.prob / maxProb) * 100}%`;
    barWrap.append(bar);
    const num = document.createElement("span");
    num.className = "pred-num";
    num.textContent = `${(p.prob * 100).toFixed(1)}%`;
    const logit = document.createElement("span");
    logit.className = "pred-logit";
    logit.textContent = `logit ${p.logit.toFixed(2)}`;
    row.append(rank, tok, barWrap, num, logit);
    return row;
  }));
}

/** Render the byte-level-BPE tokenization as chips, marking special tokens and the Ġ→␣ space marker. */
export function renderCodeTokens(container, tokens) {
  container.replaceChildren(...(tokens || []).map((t) => {
    const chip = document.createElement("span");
    chip.className = "ctok";
    chip.textContent = t.text === "" ? "∅" : t.text;
    if (t.special) chip.classList.add("ctok-special");
    if (t.space) chip.classList.add("ctok-space");
    chip.title = `id ${t.id}${t.space ? " · leading space (Ġ)" : ""}${t.special ? " · special token" : ""}`;
    return chip;
  }));
}

export const CODEFILL_CSS = `
.code-input { font-family:var(--font-mono); font-size:.85rem; inline-size:100%; box-sizing:border-box;
  padding:.6rem .7rem; border-radius:8px; border:1px solid var(--border); background:var(--bg-raised);
  color:var(--color); tab-size:2; white-space:pre; overflow-x:auto; }
.code-input:focus-visible { outline:2px solid var(--accent); outline-offset:1px; }
.filled-code { font-family:var(--font-mono); font-size:.85rem; white-space:pre; overflow-x:auto;
  border:1px solid var(--border); border-radius:8px; background:var(--bg-secondary); padding:.6rem .7rem;
  margin:.3rem 0; }
.filled-code .slot { color:var(--accent); font-weight:700; background:color-mix(in srgb, var(--accent) 14%, transparent);
  border-radius:3px; padding:0 .15em; }
.lang-picker { display:flex; flex-wrap:wrap; gap:.4rem; margin:.4rem 0; }
.lang-btn { font:inherit; font-size:.78rem; padding:.3rem .7rem; border-radius:999px; border:1px solid var(--border);
  background:var(--bg-raised); color:var(--color); cursor:pointer; min-block-size:2.1rem; }
.lang-btn:hover { border-color:var(--accent); }
.lang-btn:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.lang-btn[aria-pressed=true] { background:var(--accent); color:var(--accent-ink); border-color:var(--accent); }
.pred-list { display:flex; flex-direction:column; gap:.3rem; margin-top:.5rem; }
.pred-head { font-family:var(--font-display); font-size:1.05rem; margin:.6rem 0 .1rem; }
.pred-row { display:grid; grid-template-columns:2.2rem 8rem 1fr 3.4rem 5.2rem; align-items:center;
  gap:.5rem; text-align:start; font:inherit; padding:.25rem .35rem; border-radius:6px;
  border:1px solid transparent; background:transparent; color:var(--color); }
button.pred-row { cursor:pointer; }
button.pred-row:hover, button.pred-row:focus-visible { border-color:var(--accent); background:var(--bg-raised); }
.pred-rank { font-family:var(--font-mono); font-size:.74rem; color:var(--muted); }
.pred-tok { font-family:var(--font-mono); font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.pred-bar { block-size:.7rem; border-radius:999px; background:var(--bg-raised); border:1px solid var(--border); overflow:hidden; }
.pred-fill { display:block; block-size:100%; background:var(--accent); }
.pred-num { font-family:var(--font-mono); font-size:.8rem; text-align:end; }
.pred-logit { font-family:var(--font-mono); font-size:.72rem; color:var(--muted); text-align:end; }
@media (max-width: 560px) {
  .pred-row { grid-template-columns:1.8rem 5rem 1fr 3rem; }
  .pred-logit { display:none; }
}
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem; color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.ctok-wrap { display:flex; flex-wrap:wrap; gap:3px; margin-top:.4rem; min-inline-size:0; }
.ctok { font-family:var(--font-mono); font-size:.74rem; padding:.1rem .35rem; border-radius:4px;
  border:1px solid var(--border); background:var(--bg-raised); white-space:pre; word-break:break-all; }
.ctok-special { color:var(--muted); border-style:dashed; }
.ctok-space { color:var(--accent); }
.hint-mask { font-family:var(--font-mono); font-size:.78rem; color:var(--muted); margin-top:.3rem; }
.snip-chip { font:inherit; font-size:.78rem; padding:.2rem .6rem; border-radius:999px; border:1px solid var(--border);
  background:var(--bg-raised); color:var(--color); cursor:pointer; }
.snip-chip:hover, .snip-chip:focus-visible { border-color:var(--accent); }
`;
