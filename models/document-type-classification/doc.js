// Document-specific helpers for the DiT (RVL-CDIP) document-image-classification pages. The heavy
// lifting — the worker handshake, file→dataURL, probability-bar rendering and the shared widget CSS —
// lives in lib/classify-ui.js (the DiT worker speaks the identical protocol). This module only adds the
// document-domain knowledge: pretty labels, per-type descriptions, and the "which question to ask an
// OCR/VQA model next" routing table used by the multi-model demo.

export const DOC_WORKER_URL = "/web-ai-showcase/models/document-type-classification/worker.js";

/** RVL-CDIP labels are lower-case ("scientific report"); present them Title-Cased. */
export function prettyDocType(label) {
  return String(label)
    .split(" ")
    .map((w) => (w ? w[0].toUpperCase() + w.slice(1) : w))
    .join(" ");
}

// The 16 RVL-CDIP classes, each with a one-line "what this type looks like" note (the cues DiT keys on)
// and the single most useful question to route to a document-VQA model once the type is known. Order
// matches DiT's id2label indices 0–15.
export const DOC_TYPES = {
  "letter": {
    blurb: "Letterhead, date, inside address, salutation and signature.",
    ask: "Who is the letter addressed to?",
  },
  "form": {
    blurb: "Labelled fields, rule lines, boxes and checkboxes to fill in.",
    ask: "What is the title of the form?",
  },
  "email": {
    blurb: "From / To / Subject / Date header block above a message body.",
    ask: "What is the subject?",
  },
  "handwritten": {
    blurb: "Cursive or printed handwriting rather than typeset text.",
    ask: "What does this note say?",
  },
  "advertisement": {
    blurb: "Large display type, product imagery, a marketing pitch.",
    ask: "What product is advertised?",
  },
  "scientific report": {
    blurb: "Technical body text, tables and figures; report layout.",
    ask: "What is the title?",
  },
  "scientific publication": {
    blurb: "Two-column journal layout with abstract and references.",
    ask: "What is the title of the paper?",
  },
  "specification": {
    blurb: "Dense structured technical spec — clauses, part numbers.",
    ask: "What is being specified?",
  },
  "file folder": {
    blurb: "A folder label / tab divider rather than page content.",
    ask: "What is the folder labelled?",
  },
  "news article": {
    blurb: "Headline, byline and columns of newspaper body text.",
    ask: "What is the headline?",
  },
  "budget": {
    blurb: "Grid of line items with figures — a costing / budget table.",
    ask: "What is the total budget?",
  },
  "invoice": {
    blurb: "Bill-to block, itemised table, subtotal / tax / total.",
    ask: "What is the total amount?",
  },
  "presentation": {
    blurb: "Slide-like layout: a title and a few large bullet points.",
    ask: "What is the slide title?",
  },
  "questionnaire": {
    blurb: "Numbered questions with response options or blank lines.",
    ask: "What is the first question?",
  },
  "resume": {
    blurb: "Name header, contact line, Experience / Education sections.",
    ask: "What is the person's name?",
  },
  "memo": {
    blurb: "MEMORANDUM heading with a To / From / Date / Re block.",
    ask: "What is the subject of the memo?",
  },
};

/** The routing question the multi-model demo hands to Donut once DiT names the type. */
export function questionForType(label) {
  return DOC_TYPES[label]?.ask ?? "What is this document about?";
}

/** Document-specific styling on top of CLASSIFY_CSS: the 16-class distribution grid + page preview. */
export const DOC_CSS = `
.preview-img.doc { background: #fff; border: 1px solid var(--border); }
.dist-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: .3rem .9rem; margin-top: .5rem; }
.dist-row { display: grid; grid-template-columns: 1fr auto; gap: .5rem; align-items: center; min-inline-size: 0; }
.dist-row .dl { font-size: .78rem; overflow-wrap: anywhere; min-inline-size: 0; }
.dist-row .dv { font-family: var(--font-mono); font-size: .72rem; color: var(--muted); white-space: nowrap; }
.dist-bar { grid-column: 1 / -1; block-size: .4rem; background: var(--bg-raised); border: 1px solid var(--border); border-radius: 999px; overflow: hidden; }
.dist-bar > i { display: block; block-size: 100%; background: var(--muted); border-radius: 999px; }
.dist-row.win .dist-bar > i { background: var(--accent); }
.dist-row.win .dl { font-weight: 600; }
.doc-result { display: flex; gap: .6rem; align-items: center; margin: .4rem 0; }
.doc-result svg { flex: 0 0 auto; color: var(--accent); }
.doc-result .rt { font-family: var(--font-body); font-size: 1.05rem; font-weight: 600; }
.doc-result .rb { font-size: .8rem; color: var(--muted); }
.route-box { border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-raised); padding: .7rem .8rem; margin-top: .5rem; font-size: .88rem; }
.caption-box { border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-raised); padding: .7rem .8rem; min-block-size: 2.4rem; font-size: .95rem; }
.caption-box .cursor { background: var(--accent); display: inline-block; inline-size: .5rem; animation: blink 1s steps(2) infinite; }
@keyframes blink { 50% { opacity: 0; } }
`;

/** Inline document glyph (SVG, not emoji) for the result header. */
export const DOC_ICON =
  `<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 3v4a1 1 0 0 0 1 1h4"/><path d="M17 21H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7l5 5v11a2 2 0 0 1-2 2Z"/><path d="M9 9h1M9 13h6M9 17h6"/></svg>`;

/** Render the full 16-class distribution (label + probability, sorted). `all` = [{label, prob}]. */
export function renderDistribution(container, all) {
  const top = all[0]?.label;
  container.replaceChildren(
    ...all.map((it) => {
      const pct = (it.prob * 100).toFixed(1);
      const row = document.createElement("div");
      row.className = "dist-row" + (it.label === top ? " win" : "");
      row.innerHTML =
        `<span class="dl">${prettyDocType(it.label)}</span><span class="dv">${pct}%</span>` +
        `<span class="dist-bar"><i style="inline-size:${pct}%"></i></span>`;
      return row;
    }),
  );
}
