// Front-end helpers for the PII detection & redaction pages. Thin: owns the worker handshake, the
// 60-category → colour-group palette, and the renderers. All inference (token-classification, BIO merge,
// char-offset mapping, span trimming) lives in worker.js.

const WORKER_URL = "/web-ai-showcase/models/pii-detection-redaction/worker.js";

// The model emits 60 fine-grained PII categories. Sixty legend swatches would be unreadable, so we colour by
// nine intuitive GROUPS (the precise category is still shown as the tag on each highlight). Any category not
// listed falls back to "other".
export const GROUPS = {
  person: { label: "Name / identity", color: "#6a5cff" },
  contact: { label: "Contact", color: "#0f8bd9" },
  location: { label: "Address", color: "#2aa775" },
  financial: { label: "Financial", color: "#c9781a" },
  crypto: { label: "Crypto", color: "#b8860b" },
  secret: { label: "Secret / credential", color: "#c0392b" },
  digital: { label: "Digital / device", color: "#8e44ad" },
  work: { label: "Employment", color: "#4b6584" },
  temporal: { label: "Date / time", color: "#6b7280" },
  other: { label: "Other", color: "#7f8c8d" },
};

const TYPE_GROUP = {
  PREFIX: "person",
  FIRSTNAME: "person",
  MIDDLENAME: "person",
  LASTNAME: "person",
  FULLNAME: "person",
  NAME: "person",
  DISPLAYNAME: "person",
  SUFFIX: "person",
  USERNAME: "person",
  ACCOUNTNAME: "person",
  GENDER: "person",
  SEX: "person",
  SEXTYPE: "person",
  EMAIL: "contact",
  PHONE_NUMBER: "contact",
  PHONEIMEI: "contact",
  BUILDINGNUMBER: "location",
  CITY: "location",
  COUNTY: "location",
  STATE: "location",
  STREET: "location",
  STREETADDRESS: "location",
  SECONDARYADDRESS: "location",
  ZIPCODE: "location",
  NEARBYGPSCOORDINATE: "location",
  ORDINALDIRECTION: "location",
  ACCOUNTNUMBER: "financial",
  AMOUNT: "financial",
  BIC: "financial",
  IBAN: "financial",
  CREDITCARDNUMBER: "financial",
  CREDITCARDISSUER: "financial",
  MASKEDNUMBER: "financial",
  CURRENCY: "financial",
  CURRENCYCODE: "financial",
  CURRENCYNAME: "financial",
  CURRENCYSYMBOL: "financial",
  BITCOINADDRESS: "crypto",
  ETHEREUMADDRESS: "crypto",
  LITECOINADDRESS: "crypto",
  PASSWORD: "secret",
  PIN: "secret",
  SSN: "secret",
  CREDITCARDCVV: "secret",
  IP: "digital",
  IPV4: "digital",
  IPV6: "digital",
  MAC: "digital",
  URL: "digital",
  USERAGENT: "digital",
  VEHICLEVIN: "digital",
  VEHICLEVRM: "digital",
  COMPANY_NAME: "work",
  JOBAREA: "work",
  JOBDESCRIPTOR: "work",
  JOBTITLE: "work",
  JOBTYPE: "work",
  DATE: "temporal",
  TIME: "temporal",
  NUMBER: "other",
};

export const groupOf = (type) => TYPE_GROUP[type] ?? "other";

export function debounce(fn, ms) {
  let t;
  return (...a) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...a), ms);
  };
}

export class PIIEngine {
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
    if (msg.type === "progress") this.onProgress?.(msg.p);
    else if (msg.type === "ready") {
      this.ready = true;
      this.device = msg.device;
      for (const w of this._loadWaiters) w.resolve(msg.device);
      this._loadWaiters = [];
    } else if (msg.type === "detect") {
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
        for (const w of this._loadWaiters) w.reject(new Error(msg.message));
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
  detect(text) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "detect", id, text });
    });
  }
}

/**
 * Render text with PII spans highlighted inline, coloured by group. `spans` must be non-overlapping.
 * Each highlight is a labelled <mark> whose tooltip carries the precise category + confidence.
 */
export function renderHighlighted(container, text, spans) {
  const frag = document.createDocumentFragment();
  let cursor = 0;
  const ordered = spans.filter((s) => s.start != null && s.end != null).sort((a, b) =>
    a.start - b.start
  );
  for (const s of ordered) {
    if (s.start < cursor) continue; // defensive overlap skip
    if (s.start > cursor) frag.append(document.createTextNode(text.slice(cursor, s.start)));
    const g = groupOf(s.type);
    const mark = document.createElement("mark");
    mark.className = "pii pii-" + g;
    mark.textContent = text.slice(s.start, s.end);
    const tag = document.createElement("span");
    tag.className = "pii-tag";
    tag.textContent = s.type;
    mark.append(tag);
    mark.setAttribute(
      "title",
      `${s.type} · ${GROUPS[g].label} · ${(s.score * 100).toFixed(1)}% confidence`,
    );
    frag.append(mark);
    cursor = s.end;
  }
  if (cursor < text.length) frag.append(document.createTextNode(text.slice(cursor)));
  container.replaceChildren(frag);
}

/** Build the redacted string: every PII span replaced by a [CATEGORY] placeholder. */
export function redactText(text, spans) {
  const ordered = spans.filter((s) => s.start != null && s.end != null).sort((a, b) =>
    a.start - b.start
  );
  let out = "";
  let cursor = 0;
  for (const s of ordered) {
    if (s.start < cursor) continue;
    out += text.slice(cursor, s.start) + "[" + s.type + "]";
    cursor = s.end;
  }
  out += text.slice(cursor);
  return out;
}

/** Render the redacted output with the placeholders visually marked. */
export function renderRedacted(container, text, spans) {
  const frag = document.createDocumentFragment();
  let cursor = 0;
  const ordered = spans.filter((s) => s.start != null && s.end != null).sort((a, b) =>
    a.start - b.start
  );
  for (const s of ordered) {
    if (s.start < cursor) continue;
    if (s.start > cursor) frag.append(document.createTextNode(text.slice(cursor, s.start)));
    const ph = document.createElement("span");
    ph.className = "pii-redacted pii-" + groupOf(s.type);
    ph.textContent = "[" + s.type + "]";
    frag.append(ph);
    cursor = s.end;
  }
  if (cursor < text.length) frag.append(document.createTextNode(text.slice(cursor)));
  container.replaceChildren(frag);
}

/** Render the raw per-token BIO tags as a legible strip — the "see inside" surface. */
export function renderTokens(container, tokens) {
  container.replaceChildren(...tokens.map((t) => {
    const type = t.entity.replace(/^[BI]-/, "");
    const chip = document.createElement("span");
    chip.className = "tok-chip pii-" + groupOf(type);
    const w = document.createElement("span");
    w.className = "tok-word";
    w.textContent = String(t.word ?? "").replace(/^[▁Ġ]/, "");
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

export const PII_CSS = `
.pii-input { font: inherit; inline-size: 100%; padding: .6rem .7rem; border-radius: 8px; min-block-size: 7rem;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); resize: vertical; }
.pii-input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.pii-chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.pii-chip:hover, .pii-chip:focus-visible { border-color: var(--accent); }
.pii-toolbar { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; margin: .6rem 0 .2rem; }
.pii-switch { display: inline-flex; align-items: center; gap: .4rem; font-size: .85rem; cursor: pointer; }
.pii-out { line-height: 2.4; font-size: 1.05rem; margin-top: .5rem; white-space: pre-wrap; word-break: break-word; }
.pii { border-radius: 5px; padding: .08rem .2rem .08rem .3rem; border: 1px solid transparent; }
.pii-tag { font-family: var(--font-mono); font-size: .58rem; font-weight: 700; vertical-align: .12em;
  margin-inline-start: .28rem; padding: .02rem .22rem; border-radius: 4px; letter-spacing: .03em; }
.pii-redacted { font-family: var(--font-mono); font-size: .82rem; font-weight: 700; padding: .05rem .35rem;
  border-radius: 5px; color: #fff; }
/* colour-group tints for highlights + solid fills for tags/redactions, generated from GROUPS in JS below. */
.pii-legend { display: flex; flex-wrap: wrap; gap: .6rem; font-size: .74rem; font-family: var(--font-mono);
  color: var(--muted); margin-top: .7rem; }
.pii-legend span { display: inline-flex; align-items: center; }
.pii-legend .sw { display: inline-block; inline-size: .8rem; block-size: .8rem; border-radius: 3px;
  margin-inline-end: .3rem; }
.pii-legend .n { opacity: .7; margin-inline-start: .25rem; }
.tok-strip { display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .5rem; max-block-size: 15rem; overflow: auto; }
.tok-chip { display: inline-flex; align-items: center; gap: .3rem; padding: .12rem .4rem; border-radius: 6px;
  border: 1px solid transparent; }
.tok-word { font-family: var(--font-mono); font-size: .8rem; }
.tok-tag { font-family: var(--font-mono); font-size: .6rem; font-weight: 700; padding: .02rem .2rem; border-radius: 3px; color: #fff; }
.tok-score { font-family: var(--font-mono); font-size: .66rem; color: var(--muted); }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.pii-copy { font: inherit; font-size: .8rem; padding: .25rem .7rem; border-radius: 7px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.pii-copy:hover { border-color: var(--accent); }
`;

/** Build per-group colour CSS from GROUPS so the palette lives in one place. */
export function groupStyleCSS() {
  let css = "";
  for (const [g, { color }] of Object.entries(GROUPS)) {
    css +=
      `.pii-${g} { background: color-mix(in srgb, ${color} 18%, transparent); border-color: color-mix(in srgb, ${color} 48%, transparent); }\n`;
    css +=
      `.pii-${g} > .pii-tag, .tok-chip.pii-${g} .tok-tag, .pii-redacted.pii-${g} { background: ${color}; color: #fff; }\n`;
    css +=
      `@media (prefers-color-scheme: dark) { .pii-${g} { background: color-mix(in srgb, ${color} 32%, transparent); border-color: color-mix(in srgb, ${color} 60%, transparent); } }\n`;
  }
  return css;
}
