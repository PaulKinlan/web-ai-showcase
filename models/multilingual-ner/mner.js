// Front-end helpers for the Multilingual NER pages. Thin: owns the worker handshake + renderers +
// a small cross-lingual embedding client for the multi-model rung. All NER inference (token-
// classification, BIO merge, char-offset mapping) lives in worker.js; the embedding link demo talks
// to the already-built multilingual-embeddings worker.

const WORKER_URL = "/web-ai-showcase/models/multilingual-ner/worker.js";
const MLING_EMBED_WORKER = "/web-ai-showcase/models/multilingual-embeddings/worker.js";

// The four HRL entity classes this model emits, with accessible labels + palette roles.
export const ENTITY_TYPES = {
  PER: { label: "Person", css: "per" },
  ORG: { label: "Organization", css: "org" },
  LOC: { label: "Location", css: "loc" },
  DATE: { label: "Date", css: "date" },
};

// The ten HRL languages, each with a self-contained sample sentence and its English gloss.
export const SAMPLES = [
  {
    lang: "French",
    code: "fr",
    dir: "ltr",
    text: "Emmanuel Macron a reçu Angela Merkel à l'Élysée, à Paris, le 14 juillet 2023.",
  },
  {
    lang: "German",
    code: "de",
    dir: "ltr",
    text: "Die Deutsche Bank hat ihren Sitz in Frankfurt und wurde 1870 gegründet.",
  },
  {
    lang: "Spanish",
    code: "es",
    dir: "ltr",
    text: "Lionel Messi jugó en el Barcelona antes de fichar por el Inter Miami en 2023.",
  },
  {
    lang: "Italian",
    code: "it",
    dir: "ltr",
    text: "Giorgia Meloni ha incontrato il Papa in Vaticano lunedì scorso.",
  },
  {
    lang: "Portuguese",
    code: "pt",
    dir: "ltr",
    text: "A Petrobras anunciou em Brasília novos investimentos para 2025.",
  },
  {
    lang: "Dutch",
    code: "nl",
    dir: "ltr",
    text: "Max Verstappen won de Grand Prix in Zandvoort namens Red Bull Racing.",
  },
  { lang: "Chinese", code: "zh", dir: "ltr", text: "马云在杭州创立了阿里巴巴，公司成立于1999年。" },
  {
    lang: "Arabic",
    code: "ar",
    dir: "rtl",
    text: "التقى الملك سلمان بالرئيس الفرنسي في الرياض يوم الثلاثاء.",
  },
];

export class MNEREngine {
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

/** Minimal client for the built multilingual-embeddings worker (cross-lingual linking, multi-model rung). */
export class MlingEmbedClient {
  constructor() {
    this.worker = new Worker(MLING_EMBED_WORKER, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.onProgress = null;
    this._loadWaiters = [];
    this._pending = new Map();
    this._id = 0;
    this.worker.addEventListener("message", (e) => this._onMessage(e.data));
    this.worker.addEventListener("error", (e) => {
      const err = new Error(e.message || "Embedding worker failed to start");
      for (const w of this._loadWaiters) w.reject(err);
      this._loadWaiters = [];
      for (const [, p] of this._pending) p.reject(err);
      this._pending.clear();
    });
  }
  _onMessage(msg) {
    if (msg.type === "progress") this.onProgress?.(msg.p);
    else if (msg.type === "ready") {
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
  embed(texts) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, texts });
    });
  }
}

/** Cosine similarity for unit vectors (the embedding worker returns L2-normalized vectors). */
export function cosine(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
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
 * Each highlight is a semantic <mark> with an accessible name + confidence tooltip. dir="auto" keeps
 * RTL scripts (Arabic) laid out correctly. (modern-web-guidance highlight-text-ranges: prefer semantic
 * HTML over the CSS Custom Highlight API for meaningful, labelled ranges.)
 */
export function renderHighlighted(container, text, spans) {
  container.setAttribute("dir", "auto");
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

/** Group merged spans by entity type into a structured, deduped list (practical extraction). */
export function groupEntities(spans) {
  const groups = { PER: [], ORG: [], LOC: [], DATE: [] };
  for (const s of spans) {
    const t = s.text.trim();
    if (!t) continue;
    const bucket = groups[s.type] || (groups[s.type] = []);
    const found = bucket.find((e) => e.text.toLowerCase() === t.toLowerCase());
    if (found) found.count++;
    else bucket.push({ text: t, count: 1, score: s.score });
  }
  return groups;
}

/** The four-way legend markup, shared across pages. */
export const LEGEND_HTML = `
  <span><span class="sw ent-per"></span>PER — person</span>
  <span><span class="sw ent-org"></span>ORG — organization</span>
  <span><span class="sw ent-loc"></span>LOC — location</span>
  <span><span class="sw ent-date"></span>DATE — date</span>`;

export const MNER_CSS = `
.inside-table { border-collapse: collapse; inline-size: 100%; font-size: .9rem; }
.inside-table th, .inside-table td { border: 1px solid var(--border); padding: .4rem .55rem; text-align: start; vertical-align: top; }
.inside-table th { color: var(--muted); font-weight: 600; white-space: nowrap; inline-size: 30%; }
.ner-input { font: inherit; inline-size: 100%; padding: .6rem .7rem; border-radius: 8px; min-block-size: 4.5rem;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); resize: vertical; }
.ner-input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.ner-chip { font: inherit; font-size: .82rem; min-block-size: 2.2rem; padding: .35rem .7rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; display: inline-flex; align-items: center; gap: .3rem; }
.ner-chip:hover, .ner-chip:focus-visible { border-color: var(--accent); outline: none; }
.ner-chip .flag { font-family: var(--font-mono); font-size: .68rem; color: var(--muted); }
.ner-out { line-height: 2.4; font-size: 1.05rem; margin-top: .5rem; }
.ent { border-radius: 5px; padding: .08rem .2rem .08rem .3rem; border: 1px solid transparent; }
.ent-tag { font-family: var(--font-mono); font-size: .6rem; font-weight: 700; vertical-align: .12em;
  margin-inline-start: .28rem; padding: .02rem .22rem; border-radius: 4px; letter-spacing: .04em; }
.ent-per  { background: color-mix(in srgb, #4b3aff 18%, transparent); border-color: color-mix(in srgb, #4b3aff 45%, transparent); }
.ent-per  .ent-tag { background: #4b3aff; color: #fff; }
.ent-org  { background: color-mix(in srgb, #1a8a3a 18%, transparent); border-color: color-mix(in srgb, #1a8a3a 45%, transparent); }
.ent-org  .ent-tag { background: #1a8a3a; color: #fff; }
.ent-loc  { background: color-mix(in srgb, #c0392b 18%, transparent); border-color: color-mix(in srgb, #c0392b 45%, transparent); }
.ent-loc  .ent-tag { background: #c0392b; color: #fff; }
.ent-date { background: color-mix(in srgb, #7b2fbf 18%, transparent); border-color: color-mix(in srgb, #7b2fbf 45%, transparent); }
.ent-date .ent-tag { background: #7b2fbf; color: #fff; }
@media (prefers-color-scheme: dark) {
  .ent-per  { background: color-mix(in srgb, #8ab4f8 26%, transparent); border-color: color-mix(in srgb, #8ab4f8 55%, transparent); }
  .ent-per  .ent-tag { background: #8ab4f8; color: #0c1524; }
  .ent-org  { background: color-mix(in srgb, #57c97a 26%, transparent); border-color: color-mix(in srgb, #57c97a 55%, transparent); }
  .ent-org  .ent-tag { background: #57c97a; color: #0c1524; }
  .ent-loc  { background: color-mix(in srgb, #e06c75 26%, transparent); border-color: color-mix(in srgb, #e06c75 55%, transparent); }
  .ent-loc  .ent-tag { background: #e06c75; color: #12080a; }
  .ent-date { background: color-mix(in srgb, #c79bf0 26%, transparent); border-color: color-mix(in srgb, #c79bf0 55%, transparent); }
  .ent-date .ent-tag { background: #c79bf0; color: #140a1e; }
}
.ent-legend { display: flex; flex-wrap: wrap; gap: .6rem; font-size: .76rem; font-family: var(--font-mono);
  color: var(--muted); margin-top: .7rem; }
.ent-legend .sw { display: inline-block; inline-size: .8rem; block-size: .8rem; border-radius: 3px; vertical-align: -1px; margin-inline-end: .3rem; }
.ent-legend .sw.ent-per { background: #4b3aff; } .ent-legend .sw.ent-org { background: #1a8a3a; }
.ent-legend .sw.ent-loc { background: #c0392b; } .ent-legend .sw.ent-date { background: #7b2fbf; }
.tok-strip { display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .5rem; }
.tok-chip { display: inline-flex; align-items: center; gap: .3rem; padding: .12rem .4rem; border-radius: 6px; border: 1px solid transparent; }
.tok-word { font-family: var(--font-mono); font-size: .8rem; }
.tok-tag { font-family: var(--font-mono); font-size: .62rem; font-weight: 700; padding: .02rem .2rem; border-radius: 3px; }
.tok-score { font-family: var(--font-mono); font-size: .66rem; color: var(--muted); }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.entity-groups { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: .7rem; margin-top: .6rem; }
.entity-card { border: 1px solid var(--border); border-radius: 10px; background: var(--bg-raised); padding: .6rem .8rem; }
.entity-card h4 { margin: 0 0 .4rem; font-family: var(--font-mono); font-size: .74rem; text-transform: uppercase; letter-spacing: .05em; display: flex; align-items: center; gap: .35rem; }
.entity-card h4 .sw { display: inline-block; inline-size: .8rem; block-size: .8rem; border-radius: 3px; }
.entity-list { display: flex; flex-wrap: wrap; gap: .35rem; }
.entity-pill { font-size: .84rem; padding: .18rem .55rem; border-radius: 999px; border: 1px solid var(--border); background: var(--background); }
.entity-pill .n { color: var(--muted); font-family: var(--font-mono); font-size: .72rem; margin-inline-start: .25rem; }
.redact-out { line-height: 2.2; font-size: 1.02rem; margin-top: .5rem; }
.redact { background: var(--color); color: var(--color); border-radius: 4px; padding: 0 .35rem; font-family: var(--font-mono); font-size: .82rem; user-select: none; }
.redact::after { content: attr(data-type); color: var(--background); font-size: .62rem; font-weight: 700; }
.link-cluster { border: 1px solid var(--border); border-radius: 10px; background: var(--bg-raised); padding: .6rem .8rem; margin-top: .6rem; }
.link-cluster h4 { margin: 0 0 .35rem; font-size: .95rem; }
.link-mention { display: inline-flex; align-items: center; gap: .35rem; font-size: .84rem; padding: .16rem .5rem; border-radius: 999px; border: 1px solid var(--border); background: var(--background); margin: .15rem .25rem .15rem 0; }
.link-mention .flag { font-family: var(--font-mono); font-size: .68rem; color: var(--muted); }
.link-mention .cos { font-family: var(--font-mono); font-size: .68rem; color: var(--accent); }
`;
