// Front-end helpers for the POS-tagging pages. Thin: owns the worker handshake, the FTB→universal tag
// mapping, and the renderers (inline colour-coded words, raw sub-token strip, a small structure view).
// All inference (token-classification + SentencePiece sub-word merge + char-offset mapping) lives in
// worker.js, off the main thread.
//
// Model: Xenova/french-camembert-postag-model (task: token-classification), WASM, q8. It is a genuine
// part-of-speech tagger — every token gets a grammatical role, NOT an entity type — which is what makes
// it DISTINCT from the NER page. It labels FRENCH text with the French TreeBank (FTB) tag set. We chose
// it honestly: at build time no English/universal POS tagger shipped a transformers.js/ONNX build
// (vblagoje, QCRI bert-base-multilingual-cased-pos-english and KoichiYasuoka UPOS models are all
// PyTorch-only), whereas this French tagger has a real ONNX build that runs in the browser today.

const WORKER_URL = "/web-ai-showcase/models/pos-tagging/worker.js";

// The 11 universal grammatical groups we fold the 30 FTB tags into — friendly English labels + palette.
export const GROUPS = {
  noun: { label: "Noun", desc: "a person, place, thing or idea (common noun)" },
  propn: { label: "Proper noun", desc: "a specific name — a person, place or brand" },
  verb: { label: "Verb", desc: "an action or state (any conjugated form)" },
  adj: { label: "Adjective", desc: "describes or qualifies a noun" },
  adv: { label: "Adverb", desc: "modifies a verb, adjective or another adverb" },
  det: { label: "Determiner", desc: "articles & possessives that introduce a noun" },
  adp: { label: "Preposition", desc: "relates a noun to the rest of the sentence" },
  pron: { label: "Pronoun", desc: "stands in for a noun (incl. clitic pronouns)" },
  conj: { label: "Conjunction", desc: "joins words, phrases or clauses" },
  punct: { label: "Punctuation", desc: "commas, full stops and other marks" },
  other: { label: "Other", desc: "interjection, foreign word, prefix, etc." },
};

// French TreeBank (FTB) tag → universal group. The raw FTB tag is kept for the "see inside" surface.
export const TAG_MAP = {
  NC: "noun", // nom commun
  NPP: "propn", // nom propre
  V: "verb",
  VIMP: "verb",
  VINF: "verb",
  VPP: "verb",
  VPR: "verb",
  VS: "verb",
  ADJ: "adj",
  ADJWH: "adj",
  ADV: "adv",
  ADVWH: "adv",
  DET: "det",
  DETWH: "det",
  P: "adp", // préposition
  "P+D": "adp", // préposition + article contracté (du, au)
  "P+PRO": "adp",
  PRO: "pron",
  PROREL: "pron",
  PROWH: "pron",
  CLO: "pron", // clitique objet
  CLR: "pron", // clitique réfléchi
  CLS: "pron", // clitique sujet
  CC: "conj", // conjonction de coordination
  CS: "conj", // conjonction de subordination
  PONCT: "punct",
  ET: "other", // mot étranger
  I: "other", // interjection
  PREF: "other",
  U: "other",
  O: "other",
};

export function groupOf(tag) {
  return TAG_MAP[tag] ?? "other";
}

export class PosEngine {
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
    } else if (msg.type === "tag" || msg.type === "tagMany") {
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

  /** Tag one text → { text, words:[{surface,tag,group,score,start,end}], tokens:[…raw sub-tokens], ms, device } */
  tag(text) {
    return this._call({ type: "tag", text });
  }

  /** Tag a batch → { results:[{text,words,tokens}], ms, device } */
  tagMany(texts) {
    return this._call({ type: "tagMany", texts });
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

/** Render the tagged sentence as inline colour-coded word chips, each with its universal group tag. */
export function renderTagged(container, words, { showTags = true } = {}) {
  container.replaceChildren(...words.map((w) => {
    const chip = document.createElement("span");
    chip.className = "pos-word pg-" + w.group;
    const word = document.createElement("span");
    word.className = "pos-surface";
    word.textContent = w.surface;
    chip.append(word);
    if (showTags && w.group !== "punct") {
      const tag = document.createElement("span");
      tag.className = "pos-tag";
      tag.textContent = GROUPS[w.group]?.label ?? w.group;
      chip.append(tag);
    }
    chip.title = `${GROUPS[w.group]?.label ?? w.group} · FTB tag ${w.tag} · ${
      (w.score * 100).toFixed(1)
    }% confidence`;
    return chip;
  }));
}

/** The "see inside" raw sub-token strip: every WordPiece/SentencePiece token with its FTB tag + score. */
export function renderTokenStrip(container, tokens) {
  container.replaceChildren(...tokens.map((t) => {
    const chip = document.createElement("span");
    chip.className = "tok-chip pg-" + groupOf(t.entity);
    const w = document.createElement("span");
    w.className = "tok-word";
    w.textContent = t.word.replace(/▁/g, "·"); // show the ▁ word-boundary marker legibly
    const tag = document.createElement("span");
    tag.className = "tok-tag";
    tag.textContent = t.entity;
    const sc = document.createElement("span");
    sc.className = "tok-score";
    sc.textContent = (t.score * 100).toFixed(0) + "%";
    chip.append(w, tag, sc);
    return chip;
  }));
}

/** A small structure view: the sentence as a coloured POS ribbon + a per-group count distribution. */
export function renderStructure(container, words) {
  const counts = {};
  for (const w of words) counts[w.group] = (counts[w.group] ?? 0) + 1;
  const ribbon = words.map((w) =>
    `<span class="ribbon-cell pg-${w.group}" title="${escapeHTML(w.surface)} — ${
      GROUPS[w.group]?.label ?? w.group
    }" style="flex:${Math.max(1, w.surface.length)}"></span>`
  ).join("");
  const total = words.length || 1;
  const bars = Object.keys(GROUPS)
    .filter((g) => counts[g])
    .sort((a, b) => counts[b] - counts[a])
    .map((g) => {
      const pct = (counts[g] / total) * 100;
      return `<div class="dist-row">
        <span class="dist-label"><span class="dist-sw pg-${g}"></span>${GROUPS[g].label}</span>
        <span class="dist-bar"><span class="dist-fill pg-${g}" style="inline-size:${
        pct.toFixed(0)
      }%"></span></span>
        <span class="dist-n">${counts[g]}</span>
      </div>`;
    }).join("");
  container.innerHTML =
    `<div class="pos-ribbon" role="img" aria-label="Sequence of part-of-speech tags across the sentence">${ribbon}</div>
     <div class="dist">${bars}</div>`;
}

/** Legend of the universal groups actually present (or all of them). */
export function renderLegend(container, groups = Object.keys(GROUPS)) {
  container.replaceChildren(...groups.map((g) => {
    const s = document.createElement("span");
    s.innerHTML = `<span class="sw pg-${g}"></span>${GROUPS[g].label}`;
    return s;
  }));
}

export const POS_CSS = `
.pos-input { font: inherit; inline-size: 100%; padding: .6rem .7rem; border-radius: 8px; min-block-size: 4.5rem;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); resize: vertical; }
.pos-input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.pos-chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.pos-chip:hover, .pos-chip:focus-visible { border-color: var(--accent); }
.pos-out { line-height: 2.6; font-size: 1.05rem; margin-top: .6rem; }
.pos-word { display: inline-flex; align-items: center; gap: .3rem; border-radius: 6px; padding: .1rem .25rem .1rem .4rem;
  margin: 0 .12rem; border: 1px solid color-mix(in srgb, var(--pg) 42%, transparent);
  background: color-mix(in srgb, var(--pg) 15%, transparent); }
.pos-surface { white-space: nowrap; }
.pos-tag { font-family: var(--font-mono); font-size: .58rem; font-weight: 700; letter-spacing: .03em;
  padding: .04rem .28rem; border-radius: 4px; background: var(--pg); color: var(--pg-ink); text-transform: uppercase; }
.pos-legend { display: flex; flex-wrap: wrap; gap: .55rem; font-size: .76rem; font-family: var(--font-mono);
  color: var(--muted); margin-top: .7rem; }
.pos-legend .sw, .dist-sw { display: inline-block; inline-size: .8rem; block-size: .8rem; border-radius: 3px;
  vertical-align: -1px; margin-inline-end: .3rem; background: var(--pg); }
.tok-strip { display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .5rem; }
.tok-chip { display: inline-flex; align-items: center; gap: .3rem; padding: .12rem .4rem; border-radius: 6px;
  border: 1px solid color-mix(in srgb, var(--pg) 42%, transparent); background: color-mix(in srgb, var(--pg) 12%, transparent); }
.tok-word { font-family: var(--font-mono); font-size: .8rem; }
.tok-tag { font-family: var(--font-mono); font-size: .6rem; font-weight: 700; padding: .02rem .22rem; border-radius: 3px;
  background: var(--pg); color: var(--pg-ink); }
.tok-score { font-family: var(--font-mono); font-size: .66rem; color: var(--muted); }
.pos-ribbon { display: flex; gap: 2px; block-size: 1.5rem; margin: .5rem 0 .9rem; border-radius: 6px; overflow: hidden;
  border: 1px solid var(--border); }
.ribbon-cell { background: var(--pg); min-inline-size: 3px; }
.dist { display: grid; gap: .3rem; }
.dist-row { display: grid; grid-template-columns: 8.5rem 1fr 2rem; align-items: center; gap: .5rem; font-size: .82rem; }
.dist-label { display: flex; align-items: center; }
.dist-bar { block-size: .7rem; background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 999px; overflow: hidden; }
.dist-fill { display: block; block-size: 100%; background: var(--pg); }
.dist-n { font-family: var(--font-mono); font-size: .78rem; color: var(--muted); text-align: end; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.gloss { color: var(--muted); font-size: .82rem; font-style: italic; margin: .2rem 0 0; }
.keyword-list, .pill-row { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .5rem; }
.kw-pill { font-family: var(--font-mono); font-size: .82rem; padding: .18rem .55rem; border-radius: 999px;
  border: 1px solid color-mix(in srgb, var(--pg) 45%, transparent); background: color-mix(in srgb, var(--pg) 14%, transparent); }
.madlib-input { font: inherit; font-size: .95rem; inline-size: 7rem; padding: .05rem .3rem; border-radius: 5px;
  border: 1px dashed var(--accent); background: var(--bg-raised); color: var(--accent); text-align: center; }
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: .6rem; margin-top: .6rem; }
.stat-card { border: 1px solid var(--border); border-radius: 10px; background: var(--bg-raised); padding: .6rem .8rem; }
.stat-card b { display: block; font-family: var(--font-display); font-size: 1.5rem; }
.stat-card span { font-size: .76rem; color: var(--muted); }

/* Universal-group palette — one class each, tuned for WCAG-AA legibility in light + dark. */
.pg-noun  { --pg: #4b3aff; --pg-ink: #fff; }
.pg-propn { --pg: #0e7490; --pg-ink: #fff; }
.pg-verb  { --pg: #1a7a34; --pg-ink: #fff; }
.pg-adj   { --pg: #a15c00; --pg-ink: #fff; }
.pg-adv   { --pg: #7c3aed; --pg-ink: #fff; }
.pg-det   { --pg: #475569; --pg-ink: #fff; }
.pg-adp   { --pg: #a3155f; --pg-ink: #fff; }
.pg-pron  { --pg: #0369a1; --pg-ink: #fff; }
.pg-conj  { --pg: #6d4c00; --pg-ink: #fff; }
.pg-punct { --pg: #57534e; --pg-ink: #fff; }
.pg-other { --pg: #78716c; --pg-ink: #fff; }
@media (prefers-color-scheme: dark) {
  .pg-noun  { --pg: #8ab4f8; --pg-ink: #0c1524; }
  .pg-propn { --pg: #4dd0c4; --pg-ink: #08201d; }
  .pg-verb  { --pg: #57c97a; --pg-ink: #06210f; }
  .pg-adj   { --pg: #e0a34f; --pg-ink: #241606; }
  .pg-adv   { --pg: #c4a2f5; --pg-ink: #180a2b; }
  .pg-det   { --pg: #a3b1c2; --pg-ink: #10161d; }
  .pg-adp   { --pg: #f08fb8; --pg-ink: #2b0715; }
  .pg-pron  { --pg: #7cc0f0; --pg-ink: #06192b; }
  .pg-conj  { --pg: #d9b866; --pg-ink: #241c06; }
  .pg-punct { --pg: #b8b0a6; --pg-ink: #14110d; }
  .pg-other { --pg: #c2bbb0; --pg-ink: #14120d; }
}
`;
