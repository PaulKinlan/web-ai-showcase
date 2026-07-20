// Shared front-end helpers for the Japanese WRIME emotion pages. Keeps each page thin: it owns the
// worker handshake and the renderers (8-emotion probability bars, a verdict pill, per-piece occlusion
// attribution, a SentencePiece token strip). All inference lives in worker.js (off the main thread).

const WORKER_URL = "/web-ai-showcase/models/japanese-wrime-emotion/worker.js";

// Plutchik's eight emotions, in the model's fixed id2label order. jp = the label the model was trained
// on; en = a plain gloss; a distinct hue per emotion (used only as a bar/​swatch fill, text stays in the
// design-system colour so contrast always passes in light + dark).
export const EMOTION_META = {
  joy: { jp: "喜び", en: "Joy", emoji: "😊", color: "#e0932f" },
  sadness: { jp: "悲しみ", en: "Sadness", emoji: "😢", color: "#4a7fb5" },
  anticipation: { jp: "期待", en: "Anticipation", emoji: "🤗", color: "#5f9a52" },
  surprise: { jp: "驚き", en: "Surprise", emoji: "😲", color: "#c471a8" },
  anger: { jp: "怒り", en: "Anger", emoji: "😠", color: "#cf5045" },
  fear: { jp: "恐れ", en: "Fear", emoji: "😨", color: "#7367b8" },
  disgust: { jp: "嫌悪", en: "Disgust", emoji: "🤢", color: "#849435" },
  trust: { jp: "信頼", en: "Trust", emoji: "🤝", color: "#2f9e97" },
};
export const CLASS_ORDER = [
  "joy",
  "sadness",
  "anticipation",
  "surprise",
  "anger",
  "fear",
  "disgust",
  "trust",
];

export class EmotionEngine {
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
    } else if (msg.type === "result" || msg.type === "attr" || msg.type === "many") {
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

  /** → { text, dist:{emotion:{prob,logit}}, label, tokens[], ms, device } */
  classify(text) {
    return this._call({ type: "run", text });
  }
  /** → { text, pieces[], wordStarts[], attributions[], dist, label, target, capped, ms, device } */
  attribute(text) {
    return this._call({ type: "attribute", text });
  }
  /** → { results:[{text,dist,label}], ms, device } */
  classifyMany(texts) {
    return this._call({ type: "classifyMany", texts });
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

/** Shannon entropy (bits) of the 8-class distribution — low = the model is sure, high = it's hedging. */
export function entropyOf(dist) {
  let h = 0;
  for (const c of CLASS_ORDER) {
    const p = dist[c].prob;
    if (p > 0) h -= p * Math.log2(p);
  }
  return h;
}

/** A compact verdict pill: emoji + Japanese label + English gloss + confidence. */
export function renderVerdict(container, dist, label) {
  const meta = EMOTION_META[label];
  const conf = dist[label].prob;
  container.className = "emo-verdict";
  container.style.borderColor = meta.color;
  container.innerHTML = `<span class="emo-emoji" aria-hidden="true">${meta.emoji}</span>` +
    `<span class="emo-vlabel">${meta.jp}<span class="emo-vgloss"> · ${meta.en}</span></span>` +
    `<span class="emo-vconf">${(conf * 100).toFixed(1)}%</span>`;
}

/**
 * Render the full 8-emotion distribution as labelled bars, sorted high→low (the "see inside" surface).
 * Each row shows the emoji, Japanese + English label, a hued bar, the probability and the raw logit.
 */
export function renderProbs(container, dist, label, { sort = true } = {}) {
  const order = sort ? [...CLASS_ORDER].sort((a, b) => dist[b].prob - dist[a].prob) : CLASS_ORDER;
  container.replaceChildren(...order.map((c) => {
    const meta = EMOTION_META[c];
    const pct = dist[c].prob * 100;
    const row = document.createElement("div");
    row.className = "emo-prob-row" + (c === label ? " is-top" : "");
    row.innerHTML =
      `<span class="emo-name"><span class="emo-swatch" style="background:${meta.color}" aria-hidden="true"></span>` +
      `<span aria-hidden="true">${meta.emoji}</span> ${meta.jp} <span class="emo-en">${meta.en}</span></span>` +
      `<span class="emo-track"><span class="emo-fill" style="inline-size:${
        pct.toFixed(1)
      }%;background:${meta.color}"></span></span>` +
      `<span class="emo-val">${pct.toFixed(1)}%</span>` +
      `<span class="emo-logit">logit ${dist[c].logit.toFixed(2)}</span>`;
    return row;
  }));
}

/**
 * Render per-piece occlusion attribution: each SentencePiece piece tinted by how much removing it moved
 * the winning emotion (in log-odds). Pieces that PUSH the verdict get the emotion's hue; pieces that
 * OPPOSE it get a neutral slate. Intensity encodes magnitude. Word-start pieces get a subtle separator.
 */
export function renderAttribution(container, pieces, wordStarts, attributions, target) {
  const max = Math.max(1e-6, ...attributions.map((a) => Math.abs(a)));
  const meta = EMOTION_META[target];
  container.replaceChildren(...pieces.map((piece, i) => {
    const a = attributions[i] || 0;
    const t = Math.min(1, Math.abs(a) / max);
    const span = document.createElement("span");
    span.className = "emo-attr-piece" + (wordStarts[i] ? " is-wordstart" : "");
    const hue = a >= 0 ? meta.color : "var(--muted)";
    span.style.background = `color-mix(in srgb, ${hue} ${(t * 60).toFixed(0)}%, transparent)`;
    span.textContent = piece || "·";
    const sign = a >= 0 ? "+" : "−";
    span.title = `"${piece}" ${sign}${
      Math.abs(a).toFixed(2)
    } log-odds toward ${meta.en.toLowerCase()}`;
    return span;
  }));
}

/** Render the SentencePiece token strip — how a spaceless Japanese sentence splits into subword pieces. */
export function renderTokenStrip(container, tokens) {
  container.replaceChildren(...tokens.map((t) => {
    const chip = document.createElement("span");
    chip.className = "emo-tok-chip";
    if (t.isSpecial) chip.classList.add("emo-tok-special");
    else if (t.wordStart) chip.classList.add("emo-tok-wordstart");
    chip.textContent = t.isSpecial ? t.piece : (t.piece || "·");
    if (t.wordStart && !t.isSpecial) {
      chip.title = "SentencePiece word-start (▁) — a token boundary the tokenizer chose";
    }
    return chip;
  }));
}

export const EMOTION_CSS = `
/* At-a-glance table: let long unbreakable code tokens / URLs wrap so the table never forces page overflow
   on narrow mobile (overflow-wrap:anywhere reduces min-content width; the page body must not scroll x). */
.inside-table { inline-size: 100%; max-inline-size: 100%; }
.inside-table th, .inside-table td { overflow-wrap: anywhere; word-break: break-word; }
.inside-table code, .inside-table a { overflow-wrap: anywhere; }
.emo-input { inline-size: 100%; font: inherit; padding: .5rem .6rem; border: 1px solid var(--border);
  border-radius: 8px; background: var(--bg-raised); color: var(--color); resize: vertical; }
.emo-input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.emo-verdict { display: inline-flex; align-items: center; gap: .5rem; border: 2px solid var(--border);
  border-radius: 999px; padding: .35rem .9rem; background: var(--bg-raised);
  font-family: var(--font-display); font-size: 1.15rem; margin-top: .3rem; }
.emo-emoji { font-size: 1.4rem; line-height: 1; }
.emo-vgloss { color: var(--muted); font-size: .9rem; }
.emo-vconf { font-family: var(--font-mono); font-size: .85rem; color: var(--muted); }
.emo-prob-list { display: flex; flex-direction: column; gap: .4rem; margin-top: .6rem; }
.emo-prob-row { display: grid; grid-template-columns: minmax(9rem, auto) 1fr 3.4rem 5.2rem; gap: .5rem;
  align-items: center; padding: .1rem .2rem; border-radius: 6px; }
.emo-prob-row.is-top { background: var(--bg-raised); }
.emo-name { font-size: .86rem; white-space: nowrap; display: flex; align-items: center; gap: .3rem;
  overflow: hidden; text-overflow: ellipsis; }
.emo-prob-row.is-top .emo-name { font-weight: 600; }
.emo-en { color: var(--muted); font-size: .74rem; }
.emo-swatch { inline-size: .55rem; block-size: .55rem; border-radius: 2px; flex: none; }
.emo-track { block-size: .7rem; background: var(--bg-secondary); border: 1px solid var(--border);
  border-radius: 999px; overflow: hidden; min-inline-size: 0; }
.emo-fill { display: block; block-size: 100%; border-radius: 999px; transition: inline-size .25s ease; }
.emo-val { font-family: var(--font-mono); font-size: .78rem; text-align: end; }
.emo-logit { font-family: var(--font-mono); font-size: .72rem; color: var(--muted); text-align: end; white-space: nowrap; }
@media (max-width: 560px) {
  .emo-prob-row { grid-template-columns: minmax(6rem, auto) 1fr 3rem; }
  .emo-logit { display: none; }
  .emo-en { display: none; }
}
.emo-readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.emo-readout b { color: var(--color); font-weight: 600; }
.emo-attr-strip { display: flex; flex-wrap: wrap; gap: 3px; margin-top: .5rem; line-height: 2.1; }
.emo-attr-piece { padding: .12rem .3rem; border-radius: 5px; font-size: .95rem; font-family: var(--font-mono); }
.emo-attr-piece.is-wordstart { border-inline-start: 2px solid var(--border); padding-inline-start: .35rem; }
.emo-tok-strip { display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .5rem; }
.emo-tok-chip { font-family: var(--font-mono); font-size: .82rem; padding: .18rem .5rem; border-radius: 6px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); }
.emo-tok-chip.emo-tok-wordstart { border-style: dashed; color: var(--accent); }
.emo-tok-chip.emo-tok-wordstart::before { content: "▁"; opacity: .5; }
.emo-tok-chip.emo-tok-special { color: var(--muted); font-size: .72rem; opacity: .7; }
.emo-chip { font: inherit; font-size: .82rem; padding: .3rem .8rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer;
  min-block-size: 44px; min-inline-size: 44px; display: inline-flex; align-items: center;
  justify-content: center; gap: .4rem; }
.emo-chip:hover, .emo-chip:focus-visible { border-color: var(--accent); }
.emo-chip .emo-chip-hint { font-family: var(--font-mono); color: var(--muted); font-size: .72rem; }
.emo-sample-row { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0; }
.emo-grid { display: grid; gap: .8rem; margin-top: .6rem; }
@media (min-width: 620px) { .emo-grid.two { grid-template-columns: 1fr 1fr; } }
.emo-card { border: 1px solid var(--border); border-radius: 10px; background: var(--bg-raised); padding: .7rem .8rem; }
.emo-card h4 { margin: 0 0 .3rem; font-family: var(--font-mono); font-size: .82rem; font-weight: 600; }
.emo-hint { font-family: var(--font-mono); font-size: .78rem; color: var(--muted); margin-top: .3rem; }
`;
