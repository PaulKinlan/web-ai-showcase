// Front-end helpers for the Stanford de-identifier pages. Thin: owns the worker handshake + the
// renderers (highlight / redact / surrogate replacement + copy). All inference (token-classification,
// offset mapping, flat-label merging) lives in worker.js, off the main thread.
//
// Highlighting uses the CSS Custom Highlight API when available (Chrome 105+, Safari 17.2+, Firefox
// 135+) — ranges over the live text node, no innerHTML with user text. Fallback wraps spans in
// <mark> via DOM construction (still no innerHTML). Redaction and surrogate replacement assemble
// output strings and render them with textContent, so model-returned text can never inject markup.

const WORKER_URL = "/web-ai-showcase/models/stanford-deidentifier-base/worker.js";

// The seven PHI categories + O, with accessible labels, chip classes and highlight colors.
export const ENTITY_META = {
  PATIENT: { label: "Patient", css: "patient", color: "#c2444a" },
  HCW: { label: "Healthcare worker", css: "hcw", color: "#b55b1f" },
  HOSPITAL: { label: "Hospital / institution", css: "hospital", color: "#0f7b7b" },
  DATE: { label: "Date", css: "date", color: "#3a5bd9" },
  ID: { label: "ID / record number", css: "id", color: "#7a4fc4" },
  PHONE: { label: "Phone / fax", css: "phone", color: "#2e7d46" },
  VENDOR: { label: "Vendor / product", css: "vendor", color: "#8a6d1a" },
};

export const LEGEND_HTML = Object.entries(ENTITY_META)
  .map(([k, m]) => `<span class="ent-chip ${m.css}">${m.label}</span>`)
  .join("");

export const DEIDENT_CSS = `
.phl { position: relative; }
textarea.phl-input, input.phl-input { inline-size: 100%; font: inherit; padding: .6rem .7rem;
  border-radius: 8px; border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); }
textarea.phl-input { min-block-size: 5rem; resize: vertical; }
.sample-row { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0; }
.chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.chip:hover, .chip:focus-visible { border-color: var(--accent); }
.controls { display: flex; flex-wrap: wrap; gap: 1rem 1.4rem; align-items: center; margin: .6rem 0; }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 999px; overflow: hidden; }
.seg button { border: none; border-radius: 0; background: var(--bg-raised); color: var(--color);
  padding: .34rem .8rem; font-size: .8rem; cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--accent); color: var(--accent-ink); }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .78rem;
  color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.doc { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised);
  padding: .8rem .9rem; margin-top: .6rem; white-space: pre-wrap; line-height: 1.65; min-block-size: 4rem; }
.ent-legend { display: flex; flex-wrap: wrap; gap: .4rem .7rem; margin: .6rem 0; font-size: .78rem; }
.ent-chip { padding: .12rem .5rem; border-radius: 999px; border: 1px solid var(--border); }
.ent-chip.patient { background: color-mix(in srgb, #c2444a 16%, transparent); color: var(--color); }
.ent-chip.hcw { background: color-mix(in srgb, #b55b1f 16%, transparent); color: var(--color); }
.ent-chip.hospital { background: color-mix(in srgb, #0f7b7b 16%, transparent); color: var(--color); }
.ent-chip.date { background: color-mix(in srgb, #3a5bd9 16%, transparent); color: var(--color); }
.ent-chip.id { background: color-mix(in srgb, #7a4fc4 16%, transparent); color: var(--color); }
.ent-chip.phone { background: color-mix(in srgb, #2e7d46 16%, transparent); color: var(--color); }
.ent-chip.vendor { background: color-mix(in srgb, #8a6d1a 16%, transparent); color: var(--color); }
mark.phi { border-radius: 3px; padding: 0 .1em; }
mark.phi.patient { background: color-mix(in srgb, #c2444a 28%, transparent); text-decoration: underline #c2444a 2px; }
mark.phi.hcw { background: color-mix(in srgb, #b55b1f 28%, transparent); text-decoration: underline #b55b1f 2px; }
mark.phi.hospital { background: color-mix(in srgb, #0f7b7b 28%, transparent); text-decoration: underline #0f7b7b 2px; }
mark.phi.date { background: color-mix(in srgb, #3a5bd9 28%, transparent); text-decoration: underline #3a5bd9 2px; }
mark.phi.id { background: color-mix(in srgb, #7a4fc4 28%, transparent); text-decoration: underline #7a4fc4 2px; }
mark.phi.phone { background: color-mix(in srgb, #2e7d46 28%, transparent); text-decoration: underline #2e7d46 2px; }
mark.phi.vendor { background: color-mix(in srgb, #8a6d1a 28%, transparent); text-decoration: underline #8a6d1a 2px; }
.token-chain { display: flex; flex-wrap: wrap; gap: 3px; margin-top: .6rem; line-height: 2; }
.token-chain .t { font-family: var(--font-mono); font-size: .78rem; padding: .05rem .2rem;
  border-radius: 3px; border: 1px solid var(--border); background: var(--bg-raised); }
.token-chain .t b { font-weight: 600; }
.copy-btn { font: inherit; font-size: .8rem; padding: .34rem .8rem; border-radius: 8px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.copy-btn:hover, .copy-btn:focus-visible { border-color: var(--accent); }
.counts { display: grid; grid-template-columns: repeat(auto-fit, minmax(8.5rem, 1fr)); gap: .45rem;
  margin: .6rem 0; }
.counts .c { border: 1px solid var(--border); border-radius: 8px; padding: .35rem .55rem;
  background: var(--bg-raised); font-size: .82rem; }
.counts .c b { font-size: 1.05rem; display: inline-block; min-inline-size: 1.6rem; }
.legend-note { font-size: .78rem; color: var(--muted); }
@media (max-width: 560px){ .doc { font-size: .92rem; } .token-chain .t { font-size: .72rem; } }
`;

// -- engine -------------------------------------------------------------------------------------

export class DeidentEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
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
      this.device = msg.device;
      for (const w of this._loadWaiters) w.resolve(msg.device);
      this._loadWaiters = [];
    } else if (msg.type === "result" || msg.type === "many") {
      const p = this._pending.get(msg.id);
      if (p) { this._pending.delete(msg.id); p.resolve(msg); }
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
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load" });
    });
  }

  deidentify(text) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "deidentify", id, text });
    });
  }

  deidentifyMany(texts) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "deidentifyMany", id, texts });
    });
  }
}

// -- renderers -----------------------------------------------------------------------------------

// Build an ordered list of {start,end} replacement runs (spans with replacement text), then render
// into a <pre class="doc"> via textContent assembly — never innerHTML with user text.
function assemble(container, text, runs) {
  container.replaceChildren();
  const frag = document.createDocumentFragment();
  let cursor = 0;
  for (const r of runs) {
    if (r.start > cursor) frag.append(text.slice(cursor, r.start));
    const span = document.createElement("span");
    span.textContent = r.text;
    span.className = r.cls ?? "";
    frag.append(span);
    cursor = r.end;
  }
  if (cursor < text.length) frag.append(text.slice(cursor));
  container.append(frag);
}

/** Highlight view: CSS Custom Highlight API when available, <mark> fallback otherwise. */
export function renderHighlight(container, text, spans, { interactive = false } = {}) {
  const out = container.ownerDocument.createElement("pre");
  out.className = "doc";
  out.setAttribute("tabindex", "0");
  container.replaceChildren();
  container.append(out);

  if (interactive && typeof CSS !== "undefined" && CSS.highlights) {
    const textNode = document.createTextNode(text);
    out.append(textNode);
    const groups = {};
    for (const s of spans) (groups[s.type] ??= []).push(s);
    CSS.highlights.clear();
    for (const [type, ss] of Object.entries(groups)) {
      const ranges = ss
        .filter((s) => s.start != null && s.end != null)
        .map((s) => {
          const r = new Range();
          r.setStart(textNode, Math.min(s.start, text.length));
          r.setEnd(textNode, Math.min(s.end, text.length));
          return r;
        });
      if (ranges.length) {
        const h = new Highlight(...ranges);
        h.priority = 0;
        CSS.highlights.set(`phi-${type}`, h);
      }
    }
    return;
  }

  // Fallback: wrap spans in <mark class="phi ...">. Ranges on a single text node; clear first.
  out.replaceChildren();
  const textNode = document.createTextNode(text);
  out.append(textNode);
  const runs = spans
    .filter((s) => s.start != null && s.end != null)
    .map((s) => ({ start: s.start, end: s.end, type: s.type }))
    .sort((a, b) => a.start - b.start);
  // Merge overlapping runs of the same type (shouldn't happen; guard anyway)
  const frag = document.createDocumentFragment();
  let cursor = 0;
  for (const r of runs) {
    if (r.start > cursor) frag.append(text.slice(cursor, r.start));
    const m = document.createElement("mark");
    m.className = `phi ${ENTITY_META[r.type]?.css ?? "id"}`;
    m.textContent = text.slice(r.start, r.end);
    if (interactive) m.title = `${ENTITY_META[r.type]?.label ?? r.type}`;
    frag.append(m);
    cursor = Math.max(cursor, r.end);
  }
  if (cursor < text.length) frag.append(text.slice(cursor));
  out.replaceChildren();
  out.append(frag);
}

/** Redact view: replace each span with [REDACTED] or [CATEGORY]. */
export function renderRedacted(container, text, spans, style = "redact") {
  const runs = spans
    .filter((s) => s.start != null && s.end != null)
    .sort((a, b) => a.start - b.start)
    .map((s) => ({
      start: s.start,
      end: s.end,
      text: style === "category" ? `[${ENTITY_META[s.type]?.label.toUpperCase() ?? s.type}]` : "[REDACTED]",
      cls: "phi " + (ENTITY_META[s.type]?.css ?? "id"),
    }));
  assemble(container, text, runs);
}

/** Hide-in-plain-sight view: replace each span with a realistic synthetic surrogate. */
export function renderSurrogates(container, text, spans, seed = 1) {
  const runs = spans
    .filter((s) => s.start != null && s.end != null)
    .sort((a, b) => a.start - b.start)
    .map((s, i) => ({
      start: s.start,
      end: s.end,
      text: surrogateFor(s.type, seed, i),
      cls: "phi " + (ENTITY_META[s.type]?.css ?? "id"),
    }));
  assemble(container, text, runs);
  return runs.map((r) => r.text);
}

// -- synthetic surrogates (clearly fictional, never real patient data) ----------------------------

const SURROGATES = {
  PATIENT: ["Jamie Rivera", "Alex Chen", "Morgan Lee", "Sam Patel", "Casey Nguyen", "Jordan Kim", "Taylor Brooks", "Riley Morgan"],
  HCW: ["Dr. Alan Reyes", "Dr. Priya Nair", "Dr. Evan Cole", "Dr. Lena Novak", "Dr. Omar Haddad", "Dr. Grace Lindholm"],
  HOSPITAL: ["St. Mary's General", "Riverside Medical Center", "Northgate Hospital", "Mercy Regional Medical", "Lakeview Health Center", "Cedar Hill Medical"],
  DATE: ["02/14/2023", "June 9, 2023", "11/03/2022", "August 22, 2023", "05/30/2022", "December 4, 2023"],
  ID: ["MRN 441089", "ID 7392041", "MRN 118573", "ID 903226", "MRN 552647", "ID 318408"],
  PHONE: ["(555) 014-2277", "(555) 010-8843", "(555) 019-5512", "(555) 016-9038"],
  VENDOR: ["Acme Diagnostics", "MedSystems Ltd.", "Aurora Health Software", "Vertex Medical Tools", "Nimbus Records Inc."],
};

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function surrogateFor(type, seed, i) {
  const pool = SURROGATES[type];
  if (!pool) return "[REDACTED]";
  const rnd = mulberry32((seed * 2654435761) ^ (i * 97) ^ 0x9e3779b9);
  return pool[Math.floor(rnd() * pool.length)];
}

// -- copy helper ----------------------------------------------------------------------------------

export async function copyText(text, button, live) {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.append(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    if (live) live.textContent = "Copied to clipboard.";
    if (button) button.textContent = "Copied ✓";
    setTimeout(() => { if (button) button.textContent = "Copy"; }, 1600);
  } catch (err) {
    if (live) live.textContent = "Copy failed — select and copy manually.";
    if (button) button.textContent = "Copy failed";
  }
}
