// Front-end helpers for the camembert-ner pages. Thin: owns the worker handshake, the entity-type
// metadata, and the renderers (inline highlighted entities, raw sub-token strip, entity summary).
// All inference (token-classification + SentencePiece sub-word merge + char-offset mapping + span
// aggregation) lives in worker.js, off the main thread.
//
// Model: Xenova/camembert-ner (task: token-classification), WASM, q8 — the ONNX export of
// Jean-Baptiste/camembert-ner (MIT). A genuine FRENCH named-entity recogniser fine-tuned from
// CamemBERT on WikiNER-fr (~170k sentences), labelling PER / ORG / LOC / MISC. It is DISTINCT from
// the built pos-tagging page (grammatical role of EVERY word) and from bert-ner / multilingual-ner
// (English / multilingual BERT NER): this is the French CamemBERT NER family.

const WORKER_URL = "/web-ai-showcase/models/camembert-ner/worker.js";

// The four WikiNER entity types — friendly English labels + descriptions. The raw I- label is kept
// for the "see inside" surface.
export const TYPES = {
  PER: { label: "Person", desc: "a person's name — real or fictional" },
  ORG: { label: "Organisation", desc: "a company, institution, team or agency" },
  LOC: { label: "Location", desc: "a place — city, country, region, landmark" },
  MISC: {
    label: "Miscellaneous",
    desc: "a named entity that is none of the above (events, works…)",
  },
};

export class NerEngine {
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

  /** Tag one text → { text, words:[{surface,entity,type,score,start,end}], tokens:[…raw], entities:[{type,text,start,end,score}], ms, device } */
  tag(text) {
    return this._call({ type: "tag", text });
  }

  /** Tag a batch → { results:[{text,words,tokens,entities}], ms, device } */
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

/** Render the text with entity words highlighted inline by type; non-entity words stay plain. */
export function renderTagged(container, words, { showTags = true } = {}) {
  container.replaceChildren(...words.map((w) => {
    if (!w.type) {
      const plain = document.createElement("span");
      plain.className = "ner-plain";
      plain.textContent = w.surface;
      return plain;
    }
    const chip = document.createElement("span");
    chip.className = "ner-word pg-" + w.type;
    const word = document.createElement("span");
    word.className = "ner-surface";
    word.textContent = w.surface;
    chip.append(word);
    if (showTags) {
      const tag = document.createElement("span");
      tag.className = "ner-tag";
      tag.textContent = TYPES[w.type]?.label ?? w.type;
      chip.append(tag);
    }
    chip.title = `${TYPES[w.type]?.label ?? w.type} · raw label ${w.entity} · ${
      (w.score * 100).toFixed(1)
    }% confidence`;
    return chip;
  }));
}

/** The "see inside" raw sub-token strip: every SentencePiece token with its raw label + score. */
export function renderTokenStrip(container, tokens) {
  container.replaceChildren(...tokens.map((t) => {
    const type = t.entity?.startsWith("I-") ? t.entity.slice(2) : "O";
    const chip = document.createElement("span");
    chip.className = "tok-chip pg-" + type;
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

/** Entity summary: one card per detected span (type chip, exact text, confidence) + per-type counts. */
export function renderEntities(container, entities) {
  if (!entities.length) {
    container.innerHTML = '<p class="muted">No named entities detected in this text.</p>';
    return;
  }
  const counts = {};
  for (const e of entities) counts[e.type] = (counts[e.type] ?? 0) + 1;
  const summary = Object.keys(TYPES)
    .filter((t) => counts[t])
    .map((t) =>
      `<span class="sw-row"><span class="sw pg-${t}"></span>${TYPES[t].label} × ${counts[t]}</span>`
    ).join("");
  const cards = entities.map((e) =>
    `<span class="ent-card pg-${e.type}" title="${TYPES[e.type]?.label ?? e.type} · ${
      (e.score * 100).toFixed(1)
    }% confidence"><span class="ent-text">${escapeHTML(e.text)}</span><span class="ent-type">${
      TYPES[e.type]?.label ?? e.type
    }</span><span class="ent-score">${(e.score * 100).toFixed(0)}%</span></span>`
  ).join("");
  container.innerHTML =
    `<div class="ent-summary">${summary}</div><div class="ent-list">${cards}</div>`;
}

/** Per-type distribution bars over the detected entity mentions. */
export function renderDistribution(container, entities) {
  const counts = {};
  for (const e of entities) counts[e.type] = (counts[e.type] ?? 0) + 1;
  const total = entities.length || 1;
  const bars = Object.keys(TYPES)
    .filter((t) => counts[t])
    .sort((a, b) => counts[b] - counts[a])
    .map((t) => {
      const pct = (counts[t] / total) * 100;
      return `<div class="dist-row">
        <span class="dist-label"><span class="dist-sw pg-${t}"></span>${TYPES[t].label}</span>
        <span class="dist-bar"><span class="dist-fill pg-${t}" style="inline-size:${
        pct.toFixed(0)
      }%"></span></span>
        <span class="dist-n">${counts[t]}</span>
      </div>`;
    }).join("");
  container.innerHTML = bars || '<p class="muted">No entities yet.</p>';
}

/** Legend of the entity types actually present (or all of them). */
export function renderLegend(container, types = Object.keys(TYPES)) {
  container.replaceChildren(...types.map((t) => {
    const s = document.createElement("span");
    s.innerHTML = `<span class="sw pg-${t}"></span>${TYPES[t].label}`;
    return s;
  }));
}

export const NER_CSS = `
.ner-input { font: inherit; inline-size: 100%; padding: .6rem .7rem; border-radius: 8px; min-block-size: 4.5rem;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); resize: vertical; }
.ner-input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.ner-chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.ner-chip:hover, .ner-chip:focus-visible { border-color: var(--accent); }
.ner-out { line-height: 2.6; font-size: 1.05rem; margin-top: .6rem; }
.ner-plain { margin: 0 .12rem; }
.ner-word { display: inline-flex; align-items: center; gap: .3rem; border-radius: 6px; padding: .1rem .25rem .1rem .4rem;
  margin: 0 .12rem; border: 1px solid color-mix(in srgb, var(--pg) 42%, transparent);
  background: color-mix(in srgb, var(--pg) 15%, transparent); }
.ner-surface { white-space: nowrap; }
.ner-tag { font-family: var(--font-mono); font-size: .58rem; font-weight: 700; letter-spacing: .03em;
  padding: .04rem .28rem; border-radius: 4px; background: var(--pg); color: var(--pg-ink); text-transform: uppercase; }
.ner-legend { display: flex; flex-wrap: wrap; gap: .55rem; font-size: .76rem; font-family: var(--font-mono);
  color: var(--muted); margin-top: .7rem; }
.ner-legend .sw, .dist-sw, .sw { display: inline-block; inline-size: .8rem; block-size: .8rem; border-radius: 3px;
  vertical-align: -1px; margin-inline-end: .3rem; background: var(--pg); }
.tok-strip { display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .5rem; }
.tok-chip { display: inline-flex; align-items: center; gap: .3rem; padding: .12rem .4rem; border-radius: 6px;
  border: 1px solid color-mix(in srgb, var(--pg) 42%, transparent); background: color-mix(in srgb, var(--pg) 12%, transparent); }
.tok-word { font-family: var(--font-mono); font-size: .8rem; }
.tok-tag { font-family: var(--font-mono); font-size: .6rem; font-weight: 700; padding: .02rem .22rem; border-radius: 3px;
  background: var(--pg); color: var(--pg-ink); }
.tok-score { font-family: var(--font-mono); font-size: .66rem; color: var(--muted); }
.ent-summary { display: flex; flex-wrap: wrap; gap: .7rem; font-size: .8rem; font-family: var(--font-mono);
  color: var(--muted); margin: .4rem 0 .6rem; }
.sw-row { display: inline-flex; align-items: center; }
.ent-list { display: flex; flex-wrap: wrap; gap: .4rem; }
.ent-card { display: inline-flex; align-items: center; gap: .4rem; border-radius: 8px; padding: .25rem .5rem;
  border: 1px solid color-mix(in srgb, var(--pg) 45%, transparent);
  background: color-mix(in srgb, var(--pg) 13%, transparent); }
.ent-text { font-weight: 600; }
.ent-type { font-family: var(--font-mono); font-size: .62rem; font-weight: 700; text-transform: uppercase;
  padding: .05rem .3rem; border-radius: 4px; background: var(--pg); color: var(--pg-ink); }
.ent-score { font-family: var(--font-mono); font-size: .72rem; color: var(--muted); }
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
.pill-row { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .5rem; }
.kw-pill { font-family: var(--font-mono); font-size: .82rem; padding: .18rem .55rem; border-radius: 999px;
  border: 1px solid color-mix(in srgb, var(--pg) 45%, transparent); background: color-mix(in srgb, var(--pg) 14%, transparent); }
.remix-input { font: inherit; font-size: .95rem; inline-size: 9rem; padding: .05rem .3rem; border-radius: 5px;
  border: 1px dashed var(--accent); background: var(--bg-raised); color: var(--accent); text-align: center; }
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: .6rem; margin-top: .6rem; }
.stat-card { border: 1px solid var(--border); border-radius: 10px; background: var(--bg-raised); padding: .6rem .8rem; }
.stat-card b { display: block; font-family: var(--font-display); font-size: 1.5rem; }
.stat-card span { font-size: .76rem; color: var(--muted); }
.redact-out { line-height: 2; font-size: 1rem; margin-top: .6rem; padding: .7rem .8rem; border-radius: 8px;
  border: 1px solid var(--border); background: var(--bg-raised); }
.redact-mark { border-radius: 4px; padding: .05rem .3rem; background: var(--pg); color: var(--pg-ink);
  font-family: var(--font-mono); font-size: .8rem; }
.json-out { font-family: var(--font-mono); font-size: .78rem; white-space: pre-wrap; word-break: break-word;
  border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised); padding: .7rem .8rem;
  max-block-size: 16rem; overflow: auto; }

/* Entity-type palette — one class each, tuned for WCAG-AA legibility in light + dark. */
.pg-PER  { --pg: #4b3aff; --pg-ink: #fff; }
.pg-ORG  { --pg: #1a7a34; --pg-ink: #fff; }
.pg-LOC  { --pg: #0e7490; --pg-ink: #fff; }
.pg-MISC { --pg: #a15c00; --pg-ink: #fff; }
.pg-O    { --pg: #78716c; --pg-ink: #fff; }
@media (prefers-color-scheme: dark) {
  .pg-PER  { --pg: #8ab4f8; --pg-ink: #0c1524; }
  .pg-ORG  { --pg: #57c97a; --pg-ink: #06210f; }
  .pg-LOC  { --pg: #4dd0c4; --pg-ink: #08201d; }
  .pg-MISC { --pg: #e0a34f; --pg-ink: #241606; }
  .pg-O    { --pg: #c2bbb0; --pg-ink: #14120d; }
}
`;
