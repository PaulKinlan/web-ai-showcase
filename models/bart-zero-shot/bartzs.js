// Front-end helpers for the BART-large-MNLI zero-shot pages. Thin: owns the worker handshake and the
// renderers (ranked label scores, the 3-way NLI entailment breakdown, and the BART-vs-DeBERTa contrast).
// All inference lives in worker.js (off the main thread).

const WORKER_URL = "/web-ai-showcase/models/bart-zero-shot/worker.js";

export class BartZeroShotEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.cmpReady = false;
    this.device = "wasm";
    this.onProgress = null;
    this.onCmpProgress = null;
    this._loadWaiters = [];
    this._cmpLoadWaiters = [];
    this._pending = new Map();
    this._id = 0;
    this.worker.addEventListener("message", (e) => this._onMessage(e.data));
    this.worker.addEventListener("error", (e) => {
      const err = new Error(e.message || "Worker failed to start");
      for (const w of this._loadWaiters) w.reject(err);
      this._loadWaiters = [];
      for (const w of this._cmpLoadWaiters) w.reject(err);
      this._cmpLoadWaiters = [];
      for (const [, p] of this._pending) p.reject(err);
      this._pending.clear();
    });
  }

  _onMessage(msg) {
    if (msg.type === "progress") {
      this.onProgress?.(msg.p);
    } else if (msg.type === "cmpProgress") {
      this.onCmpProgress?.(msg.p);
    } else if (msg.type === "ready") {
      this.ready = true;
      this.device = msg.device;
      for (const w of this._loadWaiters) w.resolve(msg.device);
      this._loadWaiters = [];
    } else if (msg.type === "cmpReady") {
      this.cmpReady = true;
      for (const w of this._cmpLoadWaiters) w.resolve(msg.device);
      this._cmpLoadWaiters = [];
    } else if (msg.type === "result" || msg.type === "cmpResult") {
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
        for (const w of this._cmpLoadWaiters) w.reject(err);
        this._cmpLoadWaiters = [];
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

  loadCompare(onProgress) {
    if (onProgress) this.onCmpProgress = onProgress;
    if (this.cmpReady) return Promise.resolve(true);
    return new Promise((resolve, reject) => {
      this._cmpLoadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "loadCompare" });
    });
  }

  /** Classify → { text, multiLabel, scored:[{label,score}], nli:[{label,logits,probs}], classNames, ms, device }. */
  classify(text, labels, multiLabel = false) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, text, labels, multiLabel });
    });
  }

  /** Classify with one chosen backbone ("bart" | "deberta") → { which, scored, ms, device }. Loads it if needed. */
  classifyWith(which, text, labels, multiLabel = false) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "runWith", id, which, text, labels, multiLabel });
    });
  }

  /** Contrast → { text, multiLabel, bart:{scored,ms,device,params}, deberta:{scored,ms,device,params} }. */
  compare(text, labels, multiLabel = false) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "compare", id, text, labels, multiLabel });
    });
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Parse a comma- or newline-separated list of labels into a clean, de-duped list. */
export function parseLabels(text) {
  return [...new Set(text.split(/[,\n]/).map((s) => s.trim()).filter(Boolean))];
}

/** Render the ranked label scores as bars. `scored` is sorted descending by the worker/pipeline. */
export function renderScores(container, scored, multiLabel) {
  const max = Math.max(1e-6, ...scored.map((s) => s.score));
  container.replaceChildren(...scored.map((s, i) => {
    const row = document.createElement("div");
    row.className = "zs-row" + (i === 0 ? " top" : "");
    const w = (multiLabel ? s.score : s.score / max) * 100;
    row.innerHTML = `<div class="zs-head"><span class="zs-label">${escapeHTML(s.label)}</span>` +
      `<span class="zs-score">${(s.score * 100).toFixed(1)}%</span></div>` +
      `<div class="zs-bar"><span class="zs-fill" style="inline-size:${
        w.toFixed(1)
      }%"></span></div>`;
    return row;
  }));
}

/**
 * Render the 3-way NLI breakdown per label: for each label, a stacked bar of
 * contradiction / neutral / entailment probabilities. Entailment (the class that becomes the score) is
 * highlighted, so you can see *why* a label won — it's the one the model most entails.
 */
export function renderNLI(container, nli, classNames) {
  const entIdx = classNames.findIndex((c) => /entail/i.test(c));
  const colorFor = (name) =>
    /entail/i.test(name) ? "var(--good)" : /contradict/i.test(name) ? "var(--bad)" : "var(--muted)";
  container.replaceChildren(...nli.map((n) => {
    const row = document.createElement("div");
    row.className = "nli-row";
    const ent = entIdx >= 0 ? n.probs[entIdx] : 0;
    const seg = n.probs.map((p, i) =>
      `<span class="nli-seg" style="inline-size:${(p * 100).toFixed(1)}%;background:${
        colorFor(classNames[i])
      }" title="${escapeHTML(classNames[i])}: ${(p * 100).toFixed(1)}%"></span>`
    ).join("");
    const legend = classNames.map((c, i) =>
      `<span class="nli-k">${escapeHTML(c)} ${(n.probs[i] * 100).toFixed(0)}%</span>`
    ).join("");
    row.innerHTML = `<div class="nli-head"><span class="nli-label">${escapeHTML(n.label)}</span>` +
      `<span class="nli-ent">entailment ${(ent * 100).toFixed(1)}%</span></div>` +
      `<div class="nli-stack">${seg}</div><div class="nli-legend">${legend}</div>`;
    return row;
  }));
}

/**
 * Render the BART-vs-DeBERTa contrast: both models' ranked scores side by side, aligned by label, with
 * each model's top pick badged and any disagreement on the winner flagged.
 */
export function renderCompare(container, cmp) {
  const order = cmp.bart.scored.map((s) => s.label);
  const bMap = new Map(cmp.bart.scored.map((s) => [s.label, s.score]));
  const dMap = new Map(cmp.deberta.scored.map((s) => [s.label, s.score]));
  const bTop = cmp.bart.scored[0]?.label;
  const dTop = cmp.deberta.scored[0]?.label;
  const agree = bTop === dTop;

  const head = document.createElement("div");
  head.className = "cmp-head";
  head.innerHTML =
    `<div class="cmp-col-h"><span class="cmp-name">BART-large-MNLI</span><span class="cmp-meta">406M · ${cmp.bart.ms} ms · ${cmp.bart.device.toUpperCase()}</span></div>` +
    `<div class="cmp-col-h"><span class="cmp-name">DeBERTa-v3-xsmall</span><span class="cmp-meta">22M · ${cmp.deberta.ms} ms · ${cmp.deberta.device.toUpperCase()}</span></div>`;

  const rows = order.map((label) => {
    const b = bMap.get(label) ?? 0;
    const d = dMap.get(label) ?? 0;
    const bW = (cmp.multiLabel ? b : b / Math.max(1e-6, ...bMap.values())) * 100;
    const dW = (cmp.multiLabel ? d : d / Math.max(1e-6, ...dMap.values())) * 100;
    const row = document.createElement("div");
    row.className = "cmp-row";
    row.innerHTML = `<div class="cmp-cell${label === bTop ? " win" : ""}">` +
      `<div class="cmp-cl"><span>${escapeHTML(label)}${label === bTop ? " ★" : ""}</span><b>${
        (b * 100).toFixed(1)
      }%</b></div>` +
      `<div class="zs-bar"><span class="zs-fill" style="inline-size:${
        bW.toFixed(1)
      }%"></span></div></div>` +
      `<div class="cmp-cell${label === dTop ? " win" : ""}">` +
      `<div class="cmp-cl"><span>${escapeHTML(label)}${label === dTop ? " ★" : ""}</span><b>${
        (d * 100).toFixed(1)
      }%</b></div>` +
      `<div class="zs-bar"><span class="zs-fill deb" style="inline-size:${
        dW.toFixed(1)
      }%"></span></div></div>`;
    return row;
  });

  const verdict = document.createElement("p");
  verdict.className = "cmp-verdict " + (agree ? "ok" : "warn");
  verdict.textContent = agree
    ? `Both backbones agree: top label is “${bTop}”.`
    : `Disagreement — BART picks “${bTop}”, DeBERTa picks “${dTop}”. Same NLI trick, different backbone and training data.`;

  container.replaceChildren(head, ...rows, verdict);
}

export const BARTZS_CSS = `
.zs-list { display: flex; flex-direction: column; gap: .5rem; margin-top: .5rem; }
.zs-row { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised); padding: .45rem .7rem; }
.zs-row.top { border-color: var(--accent); border-inline-start: 4px solid var(--accent); }
.zs-head { display: flex; justify-content: space-between; align-items: baseline; gap: .5rem; }
.zs-label { font-size: .95rem; font-weight: 600; }
.zs-score { font-family: var(--font-mono); font-size: .78rem; color: var(--muted); }
.zs-bar { block-size: .5rem; background: var(--bg-secondary); border: 1px solid var(--border);
  border-radius: 999px; overflow: hidden; margin-top: .3rem; }
.zs-fill { display: block; block-size: 100%; background: var(--accent); }
.zs-fill.deb { background: color-mix(in srgb, var(--accent) 45%, var(--good)); }
.nli-list { display: flex; flex-direction: column; gap: .6rem; margin-top: .5rem; }
.nli-row { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised); padding: .5rem .7rem; }
.nli-head { display: flex; justify-content: space-between; align-items: baseline; gap: .5rem; }
.nli-label { font-weight: 600; font-size: .9rem; }
.nli-ent { font-family: var(--font-mono); font-size: .76rem; color: var(--good); }
.nli-stack { display: flex; block-size: .7rem; border-radius: 999px; overflow: hidden; margin-top: .35rem;
  border: 1px solid var(--border); }
.nli-seg { display: block; block-size: 100%; }
.nli-legend { display: flex; flex-wrap: wrap; gap: .8rem; font-family: var(--font-mono); font-size: .72rem;
  color: var(--muted); margin-top: .3rem; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.chip:hover, .chip:focus-visible { border-color: var(--accent); }
.cmp-grid { margin-top: .5rem; }
.cmp-head, .cmp-row { display: grid; grid-template-columns: 1fr 1fr; gap: .8rem; }
.cmp-row { margin-top: .5rem; }
.cmp-col-h { display: flex; flex-direction: column; gap: .1rem; padding-bottom: .3rem;
  border-bottom: 1px solid var(--border); }
.cmp-name { font-weight: 600; font-size: .9rem; }
.cmp-meta { font-family: var(--font-mono); font-size: .72rem; color: var(--muted); }
.cmp-cell { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised); padding: .4rem .6rem; }
.cmp-cell.win { border-color: var(--accent); border-inline-start: 4px solid var(--accent); }
.cmp-cl { display: flex; justify-content: space-between; align-items: baseline; gap: .5rem; font-size: .85rem; }
.cmp-cl b { font-family: var(--font-mono); font-size: .76rem; }
.cmp-verdict { margin-top: .7rem; font-size: .85rem; padding: .5rem .7rem; border-radius: 8px;
  border: 1px solid var(--border); background: var(--bg-raised); }
.cmp-verdict.ok { border-inline-start: 4px solid var(--good); }
.cmp-verdict.warn { border-inline-start: 4px solid var(--accent); }
@media (max-width: 560px) { .cmp-head, .cmp-row { grid-template-columns: 1fr; } }
`;
