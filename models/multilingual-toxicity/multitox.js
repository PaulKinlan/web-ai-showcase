// Shared front-end helpers for the multilingual toxicity / content-moderation pages. Each page stays
// thin: this module owns the worker handshake and the label renderers; all inference (the tokenizer +
// the sequence-classification forward pass + the softmax) runs in worker.js, off the main thread.
//
// Model: onnx-community/bert-multilingual-toxicity-classifier-ONNX — an ONNX export (transformers.js
// tagged) of textdetox/bert-multilingual-toxicity-classifier, itself a bert-base-multilingual-cased
// (mBERT) fine-tune from the TextDetox 2025 shared task. It is BINARY, SINGLE-LABEL: two logits →
// softmax → { neutral (idx 0), toxic (idx 1) } that sum to 1. That is the key contrast with the
// English Xenova/toxic-bert demo (six INDEPENDENT sigmoids). What makes THIS model distinct is
// breadth: one checkpoint moderates 15 languages — Arabic, Hindi, Chinese, Japanese, Hebrew, Amharic,
// Russian, Ukrainian, Tatar, German, Spanish, Italian, French, English (+ Hinglish).
//
// This is a defensive safety tool. It only scores text so a moderation queue can triage it; nothing
// leaves the device. Frame outputs soberly — a probability, not a verdict about a person.

const WORKER_URL = "/web-ai-showcase/models/multilingual-toxicity/worker.js";

// idx 0 = neutral / non-toxic, idx 1 = toxic (per the base model card's documented label order).
export const LABELS = ["neutral", "toxic"];

// The 15 languages the model was trained on, with the per-language test F1 the base card reports, so
// the UI can be honest that accuracy varies a lot by language (Arabic and German are the weakest).
export const LANGUAGES = [
  { code: "en", name: "English", f1: 0.90 },
  { code: "ru", name: "Russian", f1: 0.92 },
  { code: "uk", name: "Ukrainian", f1: 0.95 },
  { code: "de", name: "German", f1: 0.52 },
  { code: "es", name: "Spanish", f1: 0.73 },
  { code: "ar", name: "Arabic", f1: 0.51 },
  { code: "am", name: "Amharic", f1: 0.63 },
  { code: "hi", name: "Hindi", f1: 0.73 },
  { code: "zh", name: "Chinese", f1: 0.67 },
  { code: "it", name: "Italian", f1: 0.65 },
  { code: "fr", name: "French", f1: 0.91 },
  { code: "he", name: "Hebrew", f1: 0.87 },
  { code: "ja", name: "Japanese", f1: 0.86 },
  { code: "tt", name: "Tatar", f1: 0.62 },
];

// Right-to-left scripts among the supported languages — used to set dir="rtl" on the text field so
// Arabic and Hebrew render correctly.
export const RTL_CODES = new Set(["ar", "he"]);

export class MultiToxEngine {
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
    } else if (msg.type === "result" || msg.type === "triage" || msg.type === "gate") {
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

  /** Score one text → { text, pToxic, pNeutral, logits:[n,t], label, ms, device }. */
  classify(text) {
    return this._call({ type: "run", text });
  }

  /** Batch triage → { items:[{text, pToxic, flag}], ms, device }. */
  triage(texts, threshold) {
    return this._call({ type: "triage", texts, threshold });
  }

  /** Moderation gate for multi-model composition → { text, clean, pToxic, ms, device }. */
  gate(text, threshold) {
    return this._call({ type: "gate", text, threshold });
  }

  _call(payload) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...payload, id });
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

/**
 * Render the two-class probability bars (neutral + toxic). Because this is a SOFTMAX over two logits,
 * the two probabilities SUM TO 1 — the opposite of the English toxic-bert demo's independent sigmoids.
 * The toxic bar is highlighted when it lands at/above `threshold`.
 */
export function renderProbBars(container, pToxic, threshold) {
  const rows = [
    { label: "toxic", score: pToxic, flagged: pToxic >= threshold, tone: "bad" },
    { label: "neutral", score: 1 - pToxic, flagged: false, tone: "good" },
  ];
  container.replaceChildren(
    ...rows.map((l) => {
      const pct = (l.score * 100).toFixed(1);
      const row = document.createElement("div");
      row.className = "bar-row" + (l.flagged ? " bar-flag" : "");
      row.innerHTML = `
        <div class="bar-head">
          <span class="bar-label">${escapeHTML(l.label)}</span>
          <span class="bar-val">${pct}%${l.flagged ? " ⚑" : ""}</span>
        </div>
        <div class="bar-track" role="meter" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}"
             aria-label="${escapeHTML(l.label)}: ${pct} percent${l.flagged ? ", flagged" : ""}">
          <div class="bar-fill bar-${l.tone}" style="inline-size:${pct}%"></div>
        </div>`;
      return row;
    }),
  );
}

/** Overall verdict: FLAG if pToxic ≥ threshold, else ALLOW. */
export function verdict(pToxic, threshold) {
  return { flag: pToxic >= threshold, pToxic };
}

export const MULTITOX_CSS = `
.verdict-row { display: flex; align-items: baseline; gap: .8rem; flex-wrap: wrap; margin-top: .6rem; }
.verdict-label { font-family: var(--font-display); font-size: 1.7rem; }
.verdict-label.flag { color: var(--bad); }
.verdict-label.allow { color: var(--good); }
.verdict-sub { font-family: var(--font-mono); color: var(--muted); font-size: .85rem; }
.bars { display: flex; flex-direction: column; gap: .5rem; margin-top: .6rem; max-inline-size: 560px; }
.bar-head { display: flex; justify-content: space-between; gap: .5rem; font-size: .85rem; }
.bar-label { font-family: var(--font-body); text-transform: capitalize; }
.bar-val { font-family: var(--font-mono); color: var(--muted); white-space: nowrap; }
.bar-track { block-size: .7rem; background: var(--bg-raised); border: 1px solid var(--border);
  border-radius: 999px; overflow: hidden; margin-top: .15rem; }
.bar-fill { block-size: 100%; border-radius: 999px; transition: inline-size .3s ease; background: var(--muted); }
.bar-fill.bar-bad { background: var(--bad); }
.bar-fill.bar-good { background: var(--good); }
.bar-flag .bar-label { font-weight: 600; color: var(--bad); }
.thresh-wrap { display: flex; flex-direction: column; gap: .25rem; font-size: .82rem; max-inline-size: 560px; margin-top: .5rem; }
.thresh-wrap input[type=range] { accent-color: var(--accent); min-block-size: 1.6rem; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.sample-row { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0; }
.chip { font: inherit; font-size: .8rem; padding: .35rem .7rem; border-radius: 999px; min-block-size: 2.4rem;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; display: inline-flex; align-items: center; gap: .35rem; }
.chip:hover, .chip:focus-visible { border-color: var(--accent); }
.chip .chip-lang { font-family: var(--font-mono); color: var(--muted); font-size: .72rem; }
.chip[data-tone="toxic"] { border-inline-start: 3px solid var(--bad); }
.chip[data-tone="benign"] { border-inline-start: 3px solid var(--good); }
textarea.tox-in { inline-size: 100%; min-block-size: 4.5rem; resize: vertical; font: inherit;
  padding: .6rem .7rem; border-radius: 8px; border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); }
label.field { display: flex; flex-direction: column; gap: .25rem; font-size: .82rem; }
.inside-table { inline-size: 100%; border-collapse: collapse; font-size: .82rem; margin-top: .5rem; table-layout: fixed; }
.inside-table th, .inside-table td { text-align: start; padding: .3rem .5rem;
  border-bottom: 1px solid var(--border); font-family: var(--font-mono); overflow-wrap: anywhere; }
.inside-table th { color: var(--muted); font-weight: 600; }
.inside-table.glance th { inline-size: 34%; }
.queue-row { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised);
  padding: .5rem .7rem; display: flex; justify-content: space-between; gap: .6rem; align-items: center; flex-wrap: wrap; }
.queue-row.flag { border-inline-start: 4px solid var(--bad); }
.queue-row.allow { border-inline-start: 4px solid var(--good); }
.queue-text { flex: 1 1 auto; min-inline-size: 0; }
.queue-meta { font-family: var(--font-mono); font-size: .74rem; color: var(--muted); white-space: nowrap; text-align: end; }
.badge { font-family: var(--font-mono); font-size: .68rem; padding: .1rem .45rem; border-radius: 999px; border: 1px solid var(--border); }
.badge.flag { color: var(--bad); border-color: var(--bad); }
.badge.allow { color: var(--good); border-color: var(--good); }
.queue-list { display: flex; flex-direction: column; gap: .5rem; margin-top: .6rem; }
.lang-grid { display: flex; flex-wrap: wrap; gap: .35rem; margin-top: .5rem; }
.lang-pill { font-family: var(--font-mono); font-size: .72rem; padding: .2rem .5rem; border-radius: 999px;
  border: 1px solid var(--border); color: var(--muted); }
`;
