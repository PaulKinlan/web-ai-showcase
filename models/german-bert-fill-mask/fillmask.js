// Front-end helpers for the German DistilBERT fill-mask pages. Thin: owns the worker handshake +
// renderers. All inference (masked-LM logits, softmax, candidate scoring, tokenisation) lives in the
// workers, off the main thread.

export const MASK = "[MASK]";

const GERMAN_WORKER_URL = "/web-ai-showcase/models/german-bert-fill-mask/worker.js";
const MBERT_WORKER_URL = "/web-ai-showcase/models/german-bert-fill-mask/mbert-worker.js";

// Shared worker-handshake base so the two engines don't duplicate the plumbing.
class FillMaskEngineBase {
  constructor(workerUrl) {
    this.worker = new Worker(workerUrl, { type: "module" });
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
    } else if (msg.type === "fill" || msg.type === "fillMany" || msg.type === "scores") {
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

  /** Fill one text → { text, masks:[{pos, predictions:[{token,logit,prob}]}], maskCount, tokens, ms, device } */
  fill(text, topk = 8) {
    return this._call({ type: "fill", text, topk });
  }
}

export class GermanFillMaskEngine extends FillMaskEngineBase {
  constructor() {
    super(GERMAN_WORKER_URL);
  }
  /** Fill a batch of texts → { results:[{text,masks,maskCount,tokens}], ms, device } */
  fillMany(texts, topk = 8) {
    return this._call({ type: "fillMany", texts, topk });
  }
  /** Score a fixed candidate set at the first mask → { text, scores:[{word,logit,prob}], ms, device } */
  scoreCandidates(text, candidates) {
    return this._call({ type: "scoreCandidates", text, candidates });
  }
}

export class MbertFillMaskEngine extends FillMaskEngineBase {
  constructor() {
    super(MBERT_WORKER_URL);
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

/** Shannon entropy (bits) of a full-ish distribution approximated from the returned top-k probs. */
export function topkEntropy(predictions) {
  let h = 0;
  for (const p of predictions) if (p.prob > 0) h -= p.prob * Math.log2(p.prob);
  return h;
}

/**
 * Render one mask's ranked predictions as probability bars with the exact logit alongside.
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
    tok.textContent = p.token;
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

/**
 * Render the WordPiece token strip — the "see inside" surface that shows how German splits into
 * subwords (compound words become several ## pieces). `tokens` come from the worker's tokenStrip().
 */
export function renderTokenStrip(container, tokens) {
  container.replaceChildren(...tokens.map((t) => {
    const chip = document.createElement("span");
    chip.className = "tok-chip";
    if (t.isMask) chip.classList.add("tok-mask");
    else if (t.isSpecial) chip.classList.add("tok-special");
    else if (t.cont) chip.classList.add("tok-cont");
    chip.textContent = t.isSpecial ? (t.id === undefined ? t.piece : t.piece || "·") : t.piece;
    if (t.cont) chip.title = "WordPiece continuation (##) — part of a split German word";
    return chip;
  }));
}

/** Render a candidate-scoring probe (P(word) at the mask across a fixed set). */
export function renderCandidates(container, scores) {
  const max = Math.max(1e-6, ...scores.map((s) => s.prob));
  container.replaceChildren(...scores.map((s) => {
    const row = document.createElement("div");
    row.className = "cand-row";
    const w = document.createElement("span");
    w.className = "cand-word";
    w.textContent = s.word;
    const barWrap = document.createElement("span");
    barWrap.className = "cand-bar";
    const bar = document.createElement("span");
    bar.className = "cand-fill";
    bar.style.inlineSize = `${(s.prob / max) * 100}%`;
    barWrap.append(bar);
    const num = document.createElement("span");
    num.className = "cand-num";
    num.textContent = s.logit == null ? "—" : `${(s.prob * 100).toFixed(1)}%`;
    row.append(w, barWrap, num);
    return row;
  }));
}

export const GERMAN_CSS = `
.mask-input { font: inherit; inline-size: 100%; padding: .6rem .7rem; border-radius: 8px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); }
.mask-input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.hint-mask { font-family: var(--font-mono); font-size: .78rem; color: var(--muted); margin-top: .3rem; }
.mask-chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer;
  display: inline-flex; align-items: center; gap: .35rem; }
.mask-chip:hover, .mask-chip:focus-visible { border-color: var(--accent); }
.pred-list { display: flex; flex-direction: column; gap: .3rem; margin-top: .5rem; }
.pred-head { font-family: var(--font-display); font-size: 1.05rem; margin: .6rem 0 .1rem; }
.pred-row { display: grid; grid-template-columns: 2.2rem 9rem 1fr 3.4rem 5.2rem; align-items: center;
  gap: .5rem; text-align: start; font: inherit; padding: .25rem .35rem; border-radius: 6px;
  border: 1px solid transparent; background: transparent; color: var(--color); }
button.pred-row { cursor: pointer; }
button.pred-row:hover, button.pred-row:focus-visible { border-color: var(--accent); background: var(--bg-raised); }
.pred-rank { font-family: var(--font-mono); font-size: .74rem; color: var(--muted); }
.pred-tok { font-family: var(--font-mono); font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pred-bar { block-size: .7rem; border-radius: 999px; background: var(--bg-raised);
  border: 1px solid var(--border); overflow: hidden; }
.pred-fill { display: block; block-size: 100%; background: var(--accent); }
.pred-num { font-family: var(--font-mono); font-size: .8rem; text-align: end; }
.pred-logit { font-family: var(--font-mono); font-size: .72rem; color: var(--muted); text-align: end; }
@media (max-width: 560px) {
  .pred-row { grid-template-columns: 1.8rem 6rem 1fr 3rem; }
  .pred-logit { display: none; }
}
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.filled-sentence { font-family: var(--font-display); font-size: 1.15rem; margin: .3rem 0 .2rem; line-height: 1.5; }
.filled-sentence .slot { color: var(--accent); font-weight: 700; border-bottom: 2px dotted var(--accent); }
.probe-grid { display: grid; gap: .8rem; margin-top: .6rem; }
@media (min-width: 620px) { .probe-grid.two { grid-template-columns: 1fr 1fr; } }
.probe-card { border: 1px solid var(--border); border-radius: 10px; background: var(--bg-raised); padding: .7rem .8rem; }
.probe-card h4 { margin: 0 0 .3rem; font-family: var(--font-mono); font-size: .8rem; font-weight: 600;
  display: flex; align-items: center; gap: .4rem; }
.lang-tag { font-family: var(--font-mono); font-size: .72rem; color: var(--muted); font-weight: 400; }
.tok-strip { display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .5rem; }
.tok-chip { font-family: var(--font-mono); font-size: .8rem; padding: .18rem .5rem; border-radius: 6px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); }
.tok-chip.tok-cont { border-style: dashed; color: var(--accent); }
.tok-chip.tok-cont::before { content: "##"; opacity: .5; }
.tok-chip.tok-mask { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); font-weight: 700; }
.tok-chip.tok-special { color: var(--muted); font-size: .72rem; opacity: .7; }
.cand-row { display: grid; grid-template-columns: 10rem 1fr 3.4rem; align-items: center; gap: .5rem; margin: .18rem 0; }
.cand-word { font-family: var(--font-mono); font-size: .82rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cand-bar { block-size: .65rem; border-radius: 999px; background: var(--bg-secondary); border: 1px solid var(--border); overflow: hidden; }
.cand-fill { display: block; block-size: 100%; background: var(--accent); }
.cand-num { font-family: var(--font-mono); font-size: .76rem; text-align: end; color: var(--muted); }
@media (max-width: 560px) { .cand-row { grid-template-columns: 7rem 1fr 3rem; } }
`;
