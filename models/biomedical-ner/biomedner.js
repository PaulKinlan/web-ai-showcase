// Front-end helpers for the Biomedical NER pages. Thin: owns the worker handshake + renderers.
// All inference (token-classification, BIO merge, char-offset mapping) lives in worker.js.
//
// The model emits 40+ fine-grained biomedical entity types. Colour-coding 40 hues would be illegible,
// so we map every type to one of EIGHT clinically-meaningful GROUPS (condition, anatomy, procedure,
// medication, measurement, temporal, demographic, descriptor), each with a WCAG-AA colour in light +
// dark. The full raw type is always shown in the tag + tooltip, so no information is lost.

const WORKER_URL = "/web-ai-showcase/models/biomedical-ner/worker.js";

// Group definitions: label + palette role (css). Order = legend order.
export const ENTITY_GROUPS = {
  condition: { label: "Condition / finding", css: "condition" },
  anatomy: { label: "Anatomy / structure", css: "anatomy" },
  procedure: { label: "Procedure / event", css: "procedure" },
  medication: { label: "Medication / dose", css: "medication" },
  measurement: { label: "Measurement / value", css: "measurement" },
  temporal: { label: "Time / date", css: "temporal" },
  demographic: { label: "Person / history", css: "demographic" },
  descriptor: { label: "Descriptor / other", css: "descriptor" },
};

// Every biomedical-ner-all type → a group. Unknown/new types fall back to "descriptor".
const TYPE_GROUP = {
  Disease_disorder: "condition",
  Sign_symptom: "condition",
  Severity: "condition",
  Outcome: "condition",
  Biological_structure: "anatomy",
  Nonbiological_location: "anatomy",
  Biological_attribute: "anatomy",
  Diagnostic_procedure: "procedure",
  Therapeutic_procedure: "procedure",
  Clinical_event: "procedure",
  Activity: "procedure",
  Administration: "procedure",
  Medication: "medication",
  Dosage: "medication",
  Lab_value: "measurement",
  Quantitative_concept: "measurement",
  Qualitative_concept: "measurement",
  Area: "measurement",
  Distance: "measurement",
  Mass: "measurement",
  Volume: "measurement",
  Height: "measurement",
  Weight: "measurement",
  Frequency: "measurement",
  Duration: "measurement",
  Color: "measurement",
  Shape: "measurement",
  Texture: "measurement",
  Date: "temporal",
  Time: "temporal",
  Age: "demographic",
  Sex: "demographic",
  Subject: "demographic",
  Personal_background: "demographic",
  Family_history: "demographic",
  History: "demographic",
  Occupation: "demographic",
  Coreference: "demographic",
  Detailed_description: "descriptor",
  Other_entity: "descriptor",
  Other_event: "descriptor",
};

export function groupOf(type) {
  return TYPE_GROUP[type] || "descriptor";
}

/** Human-readable form of a raw type ("Disease_disorder" → "Disease disorder"). */
export function prettyType(type) {
  return String(type).replace(/_/g, " ");
}

export class BiomedNEREngine {
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
 * Each highlight is a labelled <mark> with an accessible name (raw type + confidence) in the tooltip.
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
    const g = groupOf(s.type);
    const mark = document.createElement("mark");
    mark.className = "ent ent-" + g;
    mark.textContent = text.slice(s.start, s.end);
    const tag = document.createElement("span");
    tag.className = "ent-tag";
    tag.textContent = prettyType(s.type);
    mark.append(tag);
    mark.setAttribute(
      "title",
      `${prettyType(s.type)} · ${ENTITY_GROUPS[g].label} · ${
        (s.score * 100).toFixed(1)
      }% confidence`,
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
    chip.className = "tok-chip ent-" + groupOf(type);
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

/** Render the group legend (only the groups actually present, or all if `all`). */
export function renderLegend(container, presentGroups = null) {
  const groups = presentGroups
    ? Object.keys(ENTITY_GROUPS).filter((g) => presentGroups.has(g))
    : Object.keys(ENTITY_GROUPS);
  container.replaceChildren(...groups.map((g) => {
    const item = document.createElement("span");
    const sw = document.createElement("span");
    sw.className = "sw ent-" + g;
    item.append(sw, document.createTextNode(ENTITY_GROUPS[g].label));
    return item;
  }));
}

export const BIOMED_CSS = `
.ner-input { font: inherit; inline-size: 100%; padding: .6rem .7rem; border-radius: 8px; min-block-size: 5.5rem;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); resize: vertical; }
.ner-input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.ner-chip { font: inherit; font-size: .78rem; padding: .3rem .7rem; border-radius: 999px; min-block-size: 32px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.ner-chip:hover, .ner-chip:focus-visible { border-color: var(--accent); }
.ner-out { line-height: 2.4; font-size: 1.05rem; margin-top: .5rem; overflow-wrap: anywhere; }
.ent { border-radius: 5px; padding: .08rem .2rem .08rem .3rem;
  border: 1px solid transparent; white-space: normal; }
.ent-tag { font-family: var(--font-mono); font-size: .58rem; font-weight: 700; vertical-align: .12em;
  margin-inline-start: .28rem; padding: .04rem .24rem; border-radius: 4px; letter-spacing: .02em; white-space: nowrap; }
/* Eight clinically-grouped, WCAG-AA-legible colours (surface tints, not raw hex text) in light + dark. */
.ent-condition   { background: color-mix(in srgb, #c0392b 16%, transparent); border-color: color-mix(in srgb, #c0392b 45%, transparent); }
.ent-condition   .ent-tag { background: #c0392b; color: #fff; }
.ent-anatomy     { background: color-mix(in srgb, #0e7c86 16%, transparent); border-color: color-mix(in srgb, #0e7c86 45%, transparent); }
.ent-anatomy     .ent-tag { background: #0e7c86; color: #fff; }
.ent-procedure   { background: color-mix(in srgb, #4b3aff 16%, transparent); border-color: color-mix(in srgb, #4b3aff 45%, transparent); }
.ent-procedure   .ent-tag { background: #4b3aff; color: #fff; }
.ent-medication  { background: color-mix(in srgb, #9b1d9b 16%, transparent); border-color: color-mix(in srgb, #9b1d9b 45%, transparent); }
.ent-medication  .ent-tag { background: #9b1d9b; color: #fff; }
.ent-measurement { background: color-mix(in srgb, #8a6d1a 20%, transparent); border-color: color-mix(in srgb, #8a6d1a 45%, transparent); }
.ent-measurement .ent-tag { background: #8a6d1a; color: #fff; }
.ent-temporal    { background: color-mix(in srgb, #1a8a3a 16%, transparent); border-color: color-mix(in srgb, #1a8a3a 45%, transparent); }
.ent-temporal    .ent-tag { background: #1a8a3a; color: #fff; }
.ent-demographic { background: color-mix(in srgb, #b0357a 16%, transparent); border-color: color-mix(in srgb, #b0357a 45%, transparent); }
.ent-demographic .ent-tag { background: #b0357a; color: #fff; }
.ent-descriptor  { background: color-mix(in srgb, #5a6472 18%, transparent); border-color: color-mix(in srgb, #5a6472 45%, transparent); }
.ent-descriptor  .ent-tag { background: #5a6472; color: #fff; }
@media (prefers-color-scheme: dark) {
  .ent-condition   { background: color-mix(in srgb, #e06c75 26%, transparent); border-color: color-mix(in srgb, #e06c75 55%, transparent); }
  .ent-condition   .ent-tag { background: #e06c75; color: #12080a; }
  .ent-anatomy     { background: color-mix(in srgb, #4fd0d8 26%, transparent); border-color: color-mix(in srgb, #4fd0d8 55%, transparent); }
  .ent-anatomy     .ent-tag { background: #4fd0d8; color: #06131a; }
  .ent-procedure   { background: color-mix(in srgb, #8ab4f8 26%, transparent); border-color: color-mix(in srgb, #8ab4f8 55%, transparent); }
  .ent-procedure   .ent-tag { background: #8ab4f8; color: #0c1524; }
  .ent-medication  { background: color-mix(in srgb, #e089e0 26%, transparent); border-color: color-mix(in srgb, #e089e0 55%, transparent); }
  .ent-medication  .ent-tag { background: #e089e0; color: #1a061a; }
  .ent-measurement { background: color-mix(in srgb, #e0c56c 26%, transparent); border-color: color-mix(in srgb, #e0c56c 55%, transparent); }
  .ent-measurement .ent-tag { background: #e0c56c; color: #14100a; }
  .ent-temporal    { background: color-mix(in srgb, #57c97a 26%, transparent); border-color: color-mix(in srgb, #57c97a 55%, transparent); }
  .ent-temporal    .ent-tag { background: #57c97a; color: #06140a; }
  .ent-demographic { background: color-mix(in srgb, #f08cc0 26%, transparent); border-color: color-mix(in srgb, #f08cc0 55%, transparent); }
  .ent-demographic .ent-tag { background: #f08cc0; color: #1a0812; }
  .ent-descriptor  { background: color-mix(in srgb, #aab4c2 26%, transparent); border-color: color-mix(in srgb, #aab4c2 55%, transparent); }
  .ent-descriptor  .ent-tag { background: #aab4c2; color: #0c1017; }
}
.ent-legend { display: flex; flex-wrap: wrap; gap: .6rem; font-size: .76rem; font-family: var(--font-mono);
  color: var(--muted); margin-top: .7rem; }
.ent-legend .sw { display: inline-block; inline-size: .9rem; block-size: .9rem; border-radius: 3px; vertical-align: -2px; margin-inline-end: .35rem; }
.tok-strip { display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .5rem; }
.tok-chip { display: inline-flex; align-items: center; gap: .3rem; padding: .12rem .4rem; border-radius: 6px;
  border: 1px solid transparent; }
.tok-word { font-family: var(--font-mono); font-size: .8rem; }
.tok-tag { font-family: var(--font-mono); font-size: .6rem; font-weight: 700; padding: .02rem .2rem; border-radius: 3px; background: color-mix(in srgb, currentColor 12%, transparent); }
.tok-score { font-family: var(--font-mono); font-size: .66rem; color: var(--muted); }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.count-grid { display: grid; gap: .4rem; margin-top: .6rem; }
.count-row { display: grid; grid-template-columns: 9rem 1fr 2.4rem; align-items: center; gap: .5rem; }
.count-label { font-family: var(--font-mono); font-size: .76rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.count-bar { block-size: .7rem; border-radius: 999px; background: var(--bg-raised); border: 1px solid var(--border); overflow: hidden; }
.count-fill { display: block; block-size: 100%; }
.count-num { font-family: var(--font-mono); font-size: .78rem; text-align: end; }
@media (max-width: 560px) {
  .count-row { grid-template-columns: 7rem 1fr 2.2rem; }
}
`;
