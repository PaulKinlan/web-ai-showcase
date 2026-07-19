// Shared front-end helpers for the mDeBERTa multilingual zero-shot pages. Keeps each page thin: it owns
// the worker handshake and the renderers (ranked label scores, the 3-way NLI entailment breakdown), plus
// a set of multilingual sample texts. All inference lives in worker.js (off the main thread).

const WORKER_URL = "/web-ai-showcase/models/mdeberta-zero-shot/worker.js";

export class MZeroShotEngine {
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

  /**
   * Classify text against candidate labels.
   * @param {string} text
   * @param {string[]} labels
   * @param {boolean} multiLabel independent per-label scores if true, softmax competition if false
   * @param {string} [template] hypothesis template containing "{}", e.g. "This example is {}."
   * → { text, template, multiLabel, scored:[{label,score}], nli:[{label,hypothesis,logits,probs}],
   *     classNames, ms, device }
   */
  classify(text, labels, multiLabel = false, template = undefined) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, text, labels, multiLabel, template });
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
    // Single-label: softmax across labels (sum 1); multi-label: each independent.
    const w = (multiLabel ? s.score : s.score / max) * 100;
    row.innerHTML =
      `<div class="zs-head"><span class="zs-label" dir="auto">${escapeHTML(s.label)}</span>` +
      `<span class="zs-score">${(s.score * 100).toFixed(1)}%</span></div>` +
      `<div class="zs-bar"><span class="zs-fill" style="inline-size:${
        w.toFixed(1)
      }%"></span></div>`;
    return row;
  }));
}

/**
 * Render the 3-way NLI breakdown per label: for each label, a stacked bar of
 * entailment / neutral / contradiction probabilities. Entailment (the class that becomes the score) is
 * highlighted, and the exact hypothesis sentence the model scored is shown — so you can see *why* a
 * label won, in whatever language the template + label are written.
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
    const hyp = n.hypothesis
      ? `<p class="nli-hyp" dir="auto">hypothesis: “${escapeHTML(n.hypothesis)}”</p>`
      : "";
    row.innerHTML =
      `<div class="nli-head"><span class="nli-label" dir="auto">${escapeHTML(n.label)}</span>` +
      `<span class="nli-ent">entailment ${(ent * 100).toFixed(1)}%</span></div>` +
      hyp + `<div class="nli-stack">${seg}</div><div class="nli-legend">${legend}</div>`;
    return row;
  }));
}

/** Multilingual sample texts for the demos: {lang, flag, text, note}. */
export const LANG_SAMPLES = [
  {
    lang: "Spanish",
    flag: "🇪🇸",
    text: "El nuevo teléfono tiene una cámara increíble pero la batería dura muy poco.",
    note: "product review",
  },
  {
    lang: "French",
    flag: "🇫🇷",
    text: "Ma commande devait arriver lundi et nous sommes jeudi, toujours sans nouvelles.",
    note: "shipping complaint",
  },
  {
    lang: "German",
    flag: "🇩🇪",
    text: "Die Bundesregierung hat heute ein neues Klimaschutzgesetz beschlossen.",
    note: "news headline",
  },
  {
    lang: "Japanese",
    flag: "🇯🇵",
    text: "映画の結末が急すぎて、正直がっかりしました。",
    note: "movie review",
  },
  {
    lang: "Arabic",
    flag: "🇸🇦",
    text: "الفريق فاز بالمباراة في الدقيقة الأخيرة بهدف رائع.",
    note: "sports",
  },
  {
    lang: "Hindi",
    flag: "🇮🇳",
    text: "मुझे आज बहुत अच्छा लग रहा है, सब कुछ ठीक चल रहा है।",
    note: "emotion",
  },
];

export const MZEROSHOT_CSS = `
.zs-list { display: flex; flex-direction: column; gap: .5rem; margin-top: .5rem; }
.zs-row { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised); padding: .45rem .7rem; }
.zs-row.top { border-color: var(--accent); border-inline-start: 4px solid var(--accent); }
.zs-head { display: flex; justify-content: space-between; align-items: baseline; gap: .5rem; }
.zs-label { font-size: .95rem; font-weight: 600; overflow-wrap: anywhere; }
.zs-score { font-family: var(--font-mono); font-size: .78rem; color: var(--muted); white-space: nowrap; }
.zs-bar { block-size: .5rem; background: var(--bg-secondary); border: 1px solid var(--border);
  border-radius: 999px; overflow: hidden; margin-top: .3rem; }
.zs-fill { display: block; block-size: 100%; background: var(--accent); }
.nli-list { display: flex; flex-direction: column; gap: .6rem; margin-top: .5rem; }
.nli-row { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised); padding: .5rem .7rem; }
.nli-head { display: flex; justify-content: space-between; align-items: baseline; gap: .5rem; flex-wrap: wrap; }
.nli-label { font-weight: 600; font-size: .9rem; overflow-wrap: anywhere; }
.nli-ent { font-family: var(--font-mono); font-size: .76rem; color: var(--good); white-space: nowrap; }
.nli-hyp { font-size: .76rem; color: var(--muted); margin: .3rem 0 .1rem; overflow-wrap: anywhere; }
.nli-stack { display: flex; block-size: .7rem; border-radius: 999px; overflow: hidden; margin-top: .35rem;
  border: 1px solid var(--border); }
.nli-seg { display: block; block-size: 100%; }
.nli-legend { display: flex; flex-wrap: wrap; gap: .8rem; font-family: var(--font-mono); font-size: .72rem;
  color: var(--muted); margin-top: .3rem; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.chip { font: inherit; font-size: .78rem; padding: .25rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.chip:hover { border-color: var(--accent); }
.field { display: flex; flex-direction: column; gap: .25rem; font-size: .82rem; margin-top: .6rem; }
.field textarea, .field input[type=text] { font: inherit; padding: .45rem .55rem; border: 1px solid var(--border);
  border-radius: 6px; background: var(--bg-raised); color: var(--color); inline-size: 100%; box-sizing: border-box; }
.sample-row { display: flex; flex-wrap: wrap; gap: .4rem; margin: .4rem 0; }
`;
