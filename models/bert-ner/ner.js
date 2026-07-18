// Front-end helpers for the BERT NER pages. Thin: owns the worker handshake + renderers.
// All inference (token-classification, BIO merge, char-offset mapping) lives in worker.js.

const WORKER_URL = "/web-ai-showcase/models/bert-ner/worker.js";

// The four CoNLL-2003 entity classes this model emits, with accessible labels + palette roles.
export const ENTITY_TYPES = {
  PER: { label: "Person", css: "per" },
  ORG: { label: "Organization", css: "org" },
  LOC: { label: "Location", css: "loc" },
  MISC: { label: "Misc", css: "misc" },
};

export class NEREngine {
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
    } else if (msg.type === "ner" || msg.type === "nerMany") {
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

  /** Tag one text → { text, tokens:[{entity,score,word,start,end}], spans:[{type,text,start,end,score}], ms, device } */
  ner(text) {
    return this._call({ type: "ner", text });
  }

  /** Tag a batch → { results:[{text,tokens,spans}], ms, device } */
  nerMany(texts) {
    return this._call({ type: "nerMany", texts });
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
 * Render text with entity spans highlighted inline. `spans` must be non-overlapping and sorted by start.
 * Each highlight is a labelled <mark> with an accessible name and confidence in the tooltip.
 */
export function renderHighlighted(container, text, spans) {
  const frag = document.createDocumentFragment();
  let cursor = 0;
  const ordered = spans.filter((s) => s.start != null && s.end != null).sort((a, b) =>
    a.start - b.start
  );
  for (const s of ordered) {
    if (s.start < cursor) continue; // skip any overlap defensively
    if (s.start > cursor) frag.append(document.createTextNode(text.slice(cursor, s.start)));
    const mark = document.createElement("mark");
    mark.className = "ent ent-" + (ENTITY_TYPES[s.type]?.css ?? "misc");
    mark.textContent = text.slice(s.start, s.end);
    const tag = document.createElement("span");
    tag.className = "ent-tag";
    tag.textContent = s.type;
    mark.append(tag);
    mark.setAttribute(
      "title",
      `${ENTITY_TYPES[s.type]?.label ?? s.type} · ${(s.score * 100).toFixed(1)}% confidence`,
    );
    frag.append(mark);
    cursor = s.end;
  }
  if (cursor < text.length) frag.append(document.createTextNode(text.slice(cursor)));
  container.replaceChildren(frag);
}

/** Render the raw per-token BIO tags as a legible strip — the "see inside" surface. */
export function renderTokens(container, tokens) {
  container.replaceChildren(...tokens.map((t) => {
    const type = t.entity.slice(2);
    const chip = document.createElement("span");
    chip.className = "tok-chip ent-" + (ENTITY_TYPES[type]?.css ?? "misc");
    const w = document.createElement("span");
    w.className = "tok-word";
    w.textContent = t.word;
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

export const NER_CSS = `
.ner-input { font: inherit; inline-size: 100%; padding: .6rem .7rem; border-radius: 8px; min-block-size: 4.5rem;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); resize: vertical; }
.ner-input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.ner-chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.ner-chip:hover, .ner-chip:focus-visible { border-color: var(--accent); }
.ner-out { line-height: 2.3; font-size: 1.05rem; margin-top: .5rem; }
.ent { border-radius: 5px; padding: .08rem .2rem .08rem .3rem; white-space: nowrap;
  border: 1px solid transparent; }
.ent-tag { font-family: var(--font-mono); font-size: .6rem; font-weight: 700; vertical-align: .12em;
  margin-inline-start: .28rem; padding: .02rem .22rem; border-radius: 4px; letter-spacing: .04em; }
/* Distinct, WCAG-AA-legible entity colours in light + dark, using surface tints not raw hex text. */
.ent-per  { background: color-mix(in srgb, #4b3aff 18%, transparent); border-color: color-mix(in srgb, #4b3aff 45%, transparent); }
.ent-per  .ent-tag { background: #4b3aff; color: #fff; }
.ent-org  { background: color-mix(in srgb, #1a8a3a 18%, transparent); border-color: color-mix(in srgb, #1a8a3a 45%, transparent); }
.ent-org  .ent-tag { background: #1a8a3a; color: #fff; }
.ent-loc  { background: color-mix(in srgb, #c0392b 18%, transparent); border-color: color-mix(in srgb, #c0392b 45%, transparent); }
.ent-loc  .ent-tag { background: #c0392b; color: #fff; }
.ent-misc { background: color-mix(in srgb, #8a6d1a 20%, transparent); border-color: color-mix(in srgb, #8a6d1a 45%, transparent); }
.ent-misc .ent-tag { background: #8a6d1a; color: #fff; }
@media (prefers-color-scheme: dark) {
  .ent-per  { background: color-mix(in srgb, #8ab4f8 26%, transparent); border-color: color-mix(in srgb, #8ab4f8 55%, transparent); }
  .ent-per  .ent-tag { background: #8ab4f8; color: #0c1524; }
  .ent-org  { background: color-mix(in srgb, #57c97a 26%, transparent); border-color: color-mix(in srgb, #57c97a 55%, transparent); }
  .ent-org  .ent-tag { background: #57c97a; color: #0c1524; }
  .ent-loc  { background: color-mix(in srgb, #e06c75 26%, transparent); border-color: color-mix(in srgb, #e06c75 55%, transparent); }
  .ent-loc  .ent-tag { background: #e06c75; color: #12080a; }
  .ent-misc { background: color-mix(in srgb, #e0c56c 26%, transparent); border-color: color-mix(in srgb, #e0c56c 55%, transparent); }
  .ent-misc .ent-tag { background: #e0c56c; color: #14100a; }
}
.ent-legend { display: flex; flex-wrap: wrap; gap: .6rem; font-size: .76rem; font-family: var(--font-mono);
  color: var(--muted); margin-top: .7rem; }
.ent-legend .sw { display: inline-block; inline-size: .8rem; block-size: .8rem; border-radius: 3px; vertical-align: -1px; margin-inline-end: .3rem; }
.tok-strip { display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .5rem; }
.tok-chip { display: inline-flex; align-items: center; gap: .3rem; padding: .12rem .4rem; border-radius: 6px;
  border: 1px solid transparent; }
.tok-word { font-family: var(--font-mono); font-size: .8rem; }
.tok-tag { font-family: var(--font-mono); font-size: .62rem; font-weight: 700; padding: .02rem .2rem; border-radius: 3px; }
.tok-per  .tok-tag, .ent-per.tok-chip  .tok-tag { background: #4b3aff; color: #fff; }
.tok-score { font-family: var(--font-mono); font-size: .66rem; color: var(--muted); }
.contact-grid { display: grid; gap: .5rem; margin-top: .5rem; }
.contact-card { border: 1px solid var(--border); border-radius: 10px; background: var(--bg-raised); padding: .6rem .8rem; }
.contact-card h4 { margin: 0 0 .3rem; font-family: var(--font-mono); font-size: .74rem; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
.contact-list { display: flex; flex-wrap: wrap; gap: .35rem; }
.contact-pill { font-family: var(--font-mono); font-size: .82rem; padding: .15rem .5rem; border-radius: 999px; border: 1px solid var(--border); }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.graph-wrap { margin-top: .6rem; border: 1px solid var(--border); border-radius: 10px; background: var(--bg-raised); overflow: hidden; }
.graph-wrap svg { display: block; inline-size: 100%; block-size: auto; }
`;
