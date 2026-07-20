// Shared front-end helpers for the multilingual mDeBERTa zero-shot pages. Keeps each page thin: owns the
// worker handshake and the renderers (ranked label scores, the 3-way NLI entailment breakdown). All
// inference lives in worker.js (off the main thread).

const WORKER_URL = "/web-ai-showcase/models/multilingual-zero-shot/worker.js";

// A hypothesis template in the text's own language calibrates the NLI better (XNLI aligns hypotheses
// across languages). Each sample carries a native-language template so the multilingual angle is real.
export const SAMPLES = [
  {
    lang: "German",
    code: "de",
    dir: "ltr",
    text: "Die Regierung hat heute ein neues Gesetz zur Klimapolitik verabschiedet.",
    labels: "Politik, Sport, Kochen, Technologie, Wirtschaft",
    template: "In diesem Text geht es um {}.",
  },
  {
    lang: "Spanish",
    code: "es",
    dir: "ltr",
    text: "Me encanta cocinar pasta fresca con tomate y albahaca del jardín.",
    labels: "política, deporte, cocina, tecnología, viajes",
    template: "Este ejemplo trata sobre {}.",
  },
  {
    lang: "French",
    code: "fr",
    dir: "ltr",
    text: "L'équipe a remporté la finale après une prolongation spectaculaire.",
    labels: "politique, sport, cuisine, technologie, musique",
    template: "Cet exemple parle de {}.",
  },
  {
    lang: "Chinese",
    code: "zh",
    dir: "ltr",
    text: "这家公司刚刚发布了一款新的人工智能芯片。",
    labels: "政治, 体育, 烹饪, 科技, 金融",
    template: "这个例子是关于{}的。",
  },
  {
    lang: "Arabic",
    code: "ar",
    dir: "rtl",
    text: "أعلن الفريق عن صفقة انتقال لاعب جديد قبل بداية الموسم.",
    labels: "سياسة, رياضة, طبخ, تكنولوجيا, اقتصاد",
    template: "هذا المثال يتحدث عن {}.",
  },
  {
    lang: "Hindi",
    code: "hi",
    dir: "ltr",
    text: "वैज्ञानिकों ने एक नई दूरबीन से दूर की आकाशगंगा की खोज की है।",
    labels: "राजनीति, खेल, खाना बनाना, विज्ञान, मनोरंजन",
    template: "यह उदाहरण {} के बारे में है।",
  },
];

export class ZeroShotEngine {
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

  /** Classify → { text, template, multiLabel, scored:[{label,score}], nli:[{label,logits,probs}], classNames, ms, device }. */
  classify(text, labels, multiLabel = false, template = "") {
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
  container.setAttribute("dir", "auto");
  container.replaceChildren(...scored.map((s, i) => {
    const row = document.createElement("div");
    row.className = "zs-row" + (i === 0 ? " top" : "");
    // In single-label mode scores sum to 1 (softmax across labels); in multi-label each is independent.
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
 * entailment / neutral / contradiction probabilities. Entailment (the class that becomes the score) is
 * highlighted, so you can see *why* a label won — it's the one the model most entails. Order-independent:
 * mDeBERTa's config lists entailment first, but we locate each class by name.
 */
export function renderNLI(container, nli, classNames) {
  const entIdx = classNames.findIndex((c) => /entail/i.test(c));
  const colorFor = (name) =>
    /entail/i.test(name) ? "var(--good)" : /contradict/i.test(name) ? "var(--bad)" : "var(--muted)";
  container.setAttribute("dir", "auto");
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

export function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

export const ZEROSHOT_CSS = `
.zs-list { display: flex; flex-direction: column; gap: .5rem; margin-top: .5rem; }
.zs-row { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised); padding: .45rem .7rem; }
.zs-row.top { border-color: var(--accent); border-inline-start: 4px solid var(--accent); }
.zs-head { display: flex; justify-content: space-between; align-items: baseline; gap: .5rem; }
.zs-label { font-size: .95rem; font-weight: 600; }
.zs-score { font-family: var(--font-mono); font-size: .78rem; color: var(--muted); }
.zs-bar { block-size: .5rem; background: var(--bg-secondary); border: 1px solid var(--border);
  border-radius: 999px; overflow: hidden; margin-top: .3rem; }
.zs-fill { display: block; block-size: 100%; background: var(--accent); }
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
.mzs-chip { font: inherit; font-size: .82rem; min-block-size: 2.2rem; padding: .35rem .7rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer;
  display: inline-flex; align-items: center; gap: .3rem; }
.mzs-chip:hover, .mzs-chip:focus-visible { border-color: var(--accent); outline: none; }
.mzs-chip .flag { font-family: var(--font-mono); font-size: .68rem; color: var(--muted); }
.mzs-field { display: flex; flex-direction: column; gap: .25rem; font-size: .82rem; margin-top: .6rem; }
.mzs-input { font: inherit; inline-size: 100%; padding: .5rem .6rem; border-radius: 8px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); }
.mzs-input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
textarea.mzs-input { min-block-size: 4.5rem; resize: vertical; }
`;
