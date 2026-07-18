// Shared front-end helpers for the DistilBERT question-answering pages. Keeps each page thin: it owns
// the worker handshake and the renderers (highlighted answer span, top-k candidates, token start/end
// distribution strips). All inference lives in worker.js (off the main thread).

const WORKER_URL = "/web-ai-showcase/models/distilbert-qa/worker.js";

export class QAEngine {
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

  /** Answer → { question, context, answers:[{answer,score,start,end}], tokens, startProbs, endProbs,
   *  argStart, argEnd, startPeak, endPeak, ms, device }. */
  ask(question, context, topk = 5) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, question, context, topk });
    });
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

/** Render the context with the winning answer span wrapped in a <mark>, using char offsets. */
export function renderHighlighted(container, context, start, end) {
  if (start == null || end == null || end <= start) {
    container.textContent = context;
    return;
  }
  const before = context.slice(0, start);
  const span = context.slice(start, end);
  const after = context.slice(end);
  container.innerHTML = `${escapeHTML(before)}<mark class="qa-span">${escapeHTML(span)}</mark>${
    escapeHTML(after)
  }`;
}

/** Render the ranked candidate answers with their probabilities. */
export function renderCandidates(container, answers) {
  const max = Math.max(1e-6, ...answers.map((a) => a.score));
  container.replaceChildren(...answers.map((a, i) => {
    const row = document.createElement("div");
    row.className = "cand-row" + (i === 0 ? " top" : "");
    const t = a.score / max;
    row.innerHTML = `<span class="cand-rank">${i + 1}</span>` +
      `<span class="cand-text">${escapeHTML(a.answer || "(empty)")}</span>` +
      `<span class="cand-score">${(a.score * 100).toFixed(1)}%</span>` +
      `<span class="cand-bar"><span class="cand-fill" style="inline-size:${
        (t * 100).toFixed(1)
      }%"></span></span>`;
    return row;
  }));
}

/**
 * Render the token-level start/end probability strips over the context tokens. Each token is a cell
 * whose height-bar encodes P(start) (accent) and P(end) (a second colour). The argmax start/end get
 * a labelled outline so you can see the exact span the logits pick out.
 */
export function renderTokenStrip(container, tokens, startProbs, endProbs, argStart, argEnd) {
  const maxS = Math.max(1e-6, ...startProbs);
  const maxE = Math.max(1e-6, ...endProbs);
  container.replaceChildren(...tokens.map((tokRaw, i) => {
    const tokTxt = tokRaw.replace(/^##/, "");
    const cell = document.createElement("span");
    cell.className = "tok-cell";
    if (i === argStart) cell.classList.add("is-start");
    if (i === argEnd) cell.classList.add("is-end");
    const sH = (startProbs[i] / maxS) * 100;
    const eH = (endProbs[i] / maxE) * 100;
    cell.innerHTML =
      `<span class="tok-bars"><span class="tok-bar s" style="block-size:${
        sH.toFixed(0)
      }%"></span>` +
      `<span class="tok-bar e" style="block-size:${eH.toFixed(0)}%"></span></span>` +
      `<span class="tok-label">${escapeHTML(tokTxt)}</span>`;
    cell.title = `"${tokTxt}"  P(start)=${(startProbs[i] * 100).toFixed(1)}%  P(end)=${
      (endProbs[i] * 100).toFixed(1)
    }%`;
    return cell;
  }));
}

export const QA_CSS = `
.qa-context { line-height: 1.9; background: var(--bg-raised); border: 1px solid var(--border);
  border-radius: 8px; padding: .7rem .85rem; margin-top: .5rem; }
.qa-span { background: color-mix(in srgb, var(--accent) 32%, transparent); color: var(--color);
  border-radius: 4px; padding: .05rem .15rem; font-weight: 600; box-decoration-break: clone; }
.qa-answer { font-family: var(--font-display); font-size: 1.5rem; margin: .3rem 0; }
.qa-answer.empty { color: var(--muted); font-style: italic; font-size: 1.1rem; }
.cand-list { display: flex; flex-direction: column; gap: .4rem; margin-top: .5rem; }
.cand-row { display: grid; grid-template-columns: auto 1fr auto 120px; gap: .5rem; align-items: center;
  border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised); padding: .4rem .6rem; }
.cand-row.top { border-color: var(--accent); }
.cand-rank { font-family: var(--font-mono); color: var(--muted); font-size: .78rem; }
.cand-text { font-size: .92rem; }
.cand-score { font-family: var(--font-mono); font-size: .78rem; color: var(--muted); white-space: nowrap; }
.cand-bar { block-size: .45rem; background: var(--bg-secondary); border: 1px solid var(--border);
  border-radius: 999px; overflow: hidden; }
.cand-fill { display: block; block-size: 100%; background: var(--accent); }
.tok-strip { display: flex; flex-wrap: wrap; gap: 3px; margin-top: .5rem; align-items: flex-end; }
.tok-cell { display: flex; flex-direction: column; align-items: center; gap: 2px; padding: 2px;
  border-radius: 5px; border: 1px solid transparent; }
.tok-cell.is-start { border-color: var(--accent); }
.tok-cell.is-end { border-color: var(--good); }
.tok-bars { display: flex; align-items: flex-end; gap: 2px; block-size: 46px; }
.tok-bar { inline-size: 7px; border-radius: 2px 2px 0 0; min-block-size: 1px; }
.tok-bar.s { background: var(--accent); }
.tok-bar.e { background: var(--good); }
.tok-label { font-family: var(--font-mono); font-size: .66rem; color: var(--muted); max-inline-size: 7ch;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.tok-legend { display: flex; flex-wrap: wrap; gap: 1rem; font-size: .76rem; color: var(--muted);
  font-family: var(--font-mono); margin-top: .5rem; }
.tok-legend .swatch { display: inline-block; inline-size: .8rem; block-size: .8rem; border-radius: 3px;
  margin-inline-end: .3rem; vertical-align: -1px; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.chip:hover { border-color: var(--accent); }
.qa-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem;
  align-items: start; }
.unanswerable-note { border-inline-start: 4px solid var(--warn); background: var(--bg-raised);
  padding: .5rem .7rem; border-radius: 6px; margin-top: .5rem; font-size: .88rem; }
`;
