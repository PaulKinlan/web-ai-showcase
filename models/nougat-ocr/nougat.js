// Shared front-end helpers for the Nougat academic-OCR pages. Owns the worker handshake, turns
// files/samples into data URLs, streams transcribed tokens back to the page, and renders Nougat's
// Markdown+LaTeX output into readable HTML (headings, lists, LaTeX tables, bold/italic, and math shown
// as preserved LaTeX). All inference happens off the main thread in worker.js.

const WORKER_URL = "/web-ai-showcase/models/nougat-ocr/worker.js";

export class NougatEngine {
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
    } else if (msg.type === "token") {
      this._pending.get(msg.id)?.onToken?.(msg);
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

  /**
   * Transcribe a whole document-page image to Markdown+LaTeX. Streams tokens via onToken({token, t, i});
   * resolves with { id, markdown, tokens, firstT, ms, device } when the page finishes.
   */
  transcribe(imageURL, maxTokens = 300, onToken) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject, onToken });
      this.worker.postMessage({ type: "run", id, image: imageURL, maxTokens });
    });
  }
}

/** Read a File (upload or drop) into a data URL. */
export function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// ── Tiny, dependency-free Markdown + LaTeX renderer ──────────────────────────
// Nougat emits GitHub-flavoured Markdown with embedded LaTeX: inline \( … \) and display \[ … \] math,
// and TABLES as \begin{tabular} … \end{tabular}. We keep the math as PRESERVED LaTeX in a highlighted
// box (the whole point of Nougat is that the math survives the transcription verbatim, ready to paste
// into any LaTeX/Markdown renderer), reconstruct tables as real HTML grids, and render the surrounding
// structure natively. No external library, so it works offline. Placeholders use @-delimited sentinels
// that never appear in Nougat output.

/** Count the math spans (inline + display) in a Nougat markdown string. */
export function countMath(md) {
  const display = (md.match(/\\\[[\s\S]*?\\\]|\$\$[\s\S]*?\$\$/g) || []).length;
  const inline = (md.match(/\\\([\s\S]*?\\\)|(?<![$\\])\$[^$\n]+?\$(?!\$)/g) || []).length;
  return { display, inline, total: display + inline };
}

function renderInline(text) {
  const math = [];
  let s = text.replace(/\\\(([\s\S]*?)\\\)|(?<![$\\])\$([^$\n]+?)\$(?!\$)/g, (_, a, b) => {
    math.push(a ?? b);
    return "@@MTH" + (math.length - 1) + "@@";
  });
  s = escapeHTML(s);
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
  s = s.replace(/@@MTH(\d+)@@/g, (_, i) => (
    '<span class="math math-inline">' + escapeHTML(math[+i].trim()) + "</span>"
  ));
  return s;
}

function cleanCell(c) {
  return c
    .replace(/\\hline|\\cline\{[^}]*\}/g, "")
    .replace(/\\%/g, "%").replace(/\\&/g, "&").replace(/\\\$/g, "$").replace(/\\_/g, "_")
    .replace(/\\(text|mathrm|mathbf|textbf|textit|emph)\{([^}]*)\}/g, "$2")
    .replace(/\$([^$]*)\$/g, "$1")
    .trim();
}

/** Parse the FIRST \begin{tabular}…\end{tabular} into { rows:[[cell,…],…], cols }, or null. */
export function parseLatexTable(md) {
  const m = md.match(/\\begin\{tabular\}\s*(?:\{[^}]*\})?([\s\S]*?)\\end\{tabular\}/);
  if (!m) return null;
  const body = m[1].replace(/^\s*\{[^}]*\}/, "");
  const rows = body
    .split(/\\\\/)
    .map((r) => r.replace(/\\hline/g, "").trim())
    .filter((r) => r.length)
    .map((r) => r.split("&").map(cleanCell));
  const kept = rows.filter((r) => r.some((c) => c.length));
  if (!kept.length) return null;
  return { rows: kept, cols: Math.max(...kept.map((r) => r.length)) };
}

function tableToHtml(tbl) {
  const [header, ...body] = tbl.rows;
  let t = '<div class="md-table-wrap"><table class="md-table"><thead><tr>';
  t += header.map((c) => "<th>" + renderInline(c) + "</th>").join("");
  t += "</tr></thead><tbody>";
  for (const r of body) {
    t += "<tr>" + r.map((c) => "<td>" + renderInline(c) + "</td>").join("") + "</tr>";
  }
  t += "</tbody></table></div>";
  return t;
}

/** A Nougat LaTeX table → CSV (RFC-4180 quoting). Header row included; markdown emphasis stripped. */
export function latexTableToCsv(tbl) {
  const plain = (c) => c.replace(/\*\*([^*]+)\*\*/g, "$1").replace(/\*([^*]+)\*/g, "$1");
  return tbl.rows.map((r) =>
    r.map((c) => {
      const v = plain(c);
      return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
    }).join(",")
  ).join("\n");
}

/** Render a Nougat Markdown+LaTeX string to safe HTML. */
export function renderMarkdown(md) {
  if (!md) return '<p class="muted">No text.</p>';
  const out = [];
  // Pull LaTeX tables and display-math out first (as @-sentinels on their own lines) so they survive
  // the line loop intact.
  const tables = [];
  const withTables = md.replace(
    /(?:\\begin\{table\}[\s\S]*?)?\\begin\{tabular\}[\s\S]*?\\end\{tabular\}(?:[\s\S]*?\\end\{table\})?/g,
    (block) => {
      const tbl = parseLatexTable(block);
      if (!tbl) return block;
      tables.push(tableToHtml(tbl));
      return "\n@@TBL" + (tables.length - 1) + "@@\n";
    },
  );
  const blocks = [];
  const protectedMd = withTables.replace(/\\\[([\s\S]*?)\\\]|\$\$([\s\S]*?)\$\$/g, (_, a, b) => {
    blocks.push(a ?? b);
    return "\n@@DSP" + (blocks.length - 1) + "@@\n";
  });
  const lines = protectedMd.split(/\r?\n/);
  let i = 0;
  let para = [];
  const flushPara = () => {
    if (para.length) {
      out.push("<p>" + renderInline(para.join(" ")) + "</p>");
      para = [];
    }
  };
  while (i < lines.length) {
    const line = lines[i];
    const tmatch = line.match(/^@@TBL(\d+)@@$/);
    if (tmatch) {
      flushPara();
      out.push(tables[+tmatch[1]]);
      i++;
      continue;
    }
    const bmatch = line.match(/^@@DSP(\d+)@@$/);
    if (bmatch) {
      flushPara();
      out.push(
        '<div class="math math-display">' + escapeHTML(blocks[+bmatch[1]].trim()) + "</div>",
      );
      i++;
      continue;
    }
    if (!line.trim()) {
      flushPara();
      i++;
      continue;
    }
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) {
      flushPara();
      const lvl = Math.min(6, h[1].length);
      out.push("<h" + lvl + ' class="md-h">' + renderInline(h[2]) + "</h" + lvl + ">");
      i++;
      continue;
    }
    // Markdown pipe table (header + separator row of dashes).
    if (
      /\|/.test(line) && i + 1 < lines.length &&
      /^\s*\|?[\s:|-]+\|?\s*$/.test(lines[i + 1]) && /-/.test(lines[i + 1])
    ) {
      flushPara();
      const header = splitRow(line);
      i += 2;
      const rows = [];
      while (i < lines.length && /\|/.test(lines[i]) && lines[i].trim()) {
        rows.push(splitRow(lines[i]));
        i++;
      }
      let t = '<div class="md-table-wrap"><table class="md-table"><thead><tr>';
      t += header.map((c) => "<th>" + renderInline(c) + "</th>").join("");
      t += "</tr></thead><tbody>";
      for (const r of rows) {
        t += "<tr>" + r.map((c) => "<td>" + renderInline(c) + "</td>").join("") + "</tr>";
      }
      t += "</tbody></table></div>";
      out.push(t);
      continue;
    }
    const li = line.match(/^\s*[-*+]\s+(.*)$/);
    if (li) {
      flushPara();
      const items = [];
      while (i < lines.length) {
        const m = lines[i].match(/^\s*[-*+]\s+(.*)$/);
        if (!m) break;
        items.push("<li>" + renderInline(m[1]) + "</li>");
        i++;
      }
      out.push('<ul class="md-list">' + items.join("") + "</ul>");
      continue;
    }
    para.push(line.trim());
    i++;
  }
  flushPara();
  return out.join("\n");
}

function splitRow(line) {
  return line.replace(/^\s*\|/, "").replace(/\|\s*$/, "").split("|").map((c) => c.trim());
}

/** Shared inline styles for the Nougat widgets (dropzone, page preview, rendered markdown, math, trace). */
export const NOUGAT_CSS = `
.dropzone {
  border: 2px dashed var(--border-strong); border-radius: var(--radius); background: var(--bg-raised);
  padding: 1rem; text-align: center; cursor: pointer; transition: border-color .15s, background .15s;
}
.dropzone.drag { border-color: var(--accent); background: var(--bg-secondary); }
.dropzone:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.sample-strip { display: flex; gap: .5rem; flex-wrap: wrap; margin: .5rem 0; }
.sample-thumb {
  block-size: 88px; max-inline-size: 130px; object-fit: cover; object-position: top; border-radius: 6px;
  border: 2px solid transparent; cursor: pointer; padding: 2px; background: #fff;
}
.sample-thumb.active { border-color: var(--accent); }
.sample-thumb:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.doc-wrap { max-inline-size: 100%; margin-top: .4rem; }
.doc-img {
  max-inline-size: 100%; block-size: auto; border-radius: 8px; display: block;
  background: #fff; border: 1px solid var(--border);
}
.field-row { display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start; margin: .8rem 0; }
.field-col { flex: 1 1 320px; min-inline-size: 0; }
.status.ok { color: var(--ok, #15803d); }
.status.err { color: var(--err, #b91c1c); }
.seg {
  display: inline-flex; border: 1px solid var(--border); border-radius: 999px; overflow: hidden; margin: .4rem 0;
}
.seg button {
  font: inherit; font-size: .8rem; padding: .28rem .8rem; border: 0; background: var(--bg-raised);
  color: var(--color); cursor: pointer; min-block-size: 36px;
}
.seg button[aria-pressed="true"] { background: var(--accent); color: #fff; }
.seg button:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.rendered {
  padding: .9rem 1.1rem; border-radius: var(--radius); background: var(--bg-raised);
  border: 1px solid var(--border); min-block-size: 4em; overflow-wrap: anywhere; line-height: 1.5;
}
.rendered .md-h { font-family: var(--font-display, Georgia, serif); margin: .7rem 0 .35rem; line-height: 1.25; }
.rendered h1.md-h { font-size: 1.35rem; } .rendered h2.md-h { font-size: 1.15rem; }
.rendered h3.md-h, .rendered h4.md-h, .rendered h5.md-h, .rendered h6.md-h { font-size: 1rem; color: var(--muted); }
.rendered p { margin: .5rem 0; }
.rendered .md-list { margin: .4rem 0 .4rem 1.1rem; }
.math {
  font-family: var(--font-mono); background: color-mix(in srgb, var(--accent) 12%, transparent);
  border: 1px solid color-mix(in srgb, var(--accent) 35%, transparent); border-radius: 6px;
}
.math-inline { padding: 0 .28em; font-size: .92em; white-space: pre-wrap; }
.math-display {
  display: block; padding: .6rem .9rem; margin: .6rem 0; white-space: pre-wrap; overflow-x: auto;
  font-size: .92rem;
}
.md-table-wrap { overflow-x: auto; margin: .6rem 0; }
.md-table { border-collapse: collapse; font-size: .85rem; inline-size: 100%; }
.md-table th, .md-table td { border: 1px solid var(--border); padding: .3rem .55rem; text-align: left; }
.md-table th { background: var(--bg-secondary); font-weight: 600; }
.raw-md {
  font-family: var(--font-mono); font-size: .8rem; line-height: 1.45; white-space: pre-wrap;
  word-break: break-word; padding: .9rem 1.1rem; border-radius: var(--radius); background: var(--bg-raised);
  border: 1px solid var(--border); min-block-size: 4em; margin: 0;
}
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .78rem;
  color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.token-trace {
  display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .4rem; max-block-size: 200px; overflow-y: auto;
}
.token-trace .tok {
  font-family: var(--font-mono); font-size: .72rem; padding: .1rem .4rem; border-radius: 4px;
  background: var(--bg-secondary); border: 1px solid var(--border); white-space: pre;
}
.token-trace .tok small { color: var(--muted); margin-inline-start: .3rem; }
.chip {
  font: inherit; font-size: .78rem; padding: .3rem .7rem; border-radius: 999px; min-block-size: 36px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer;
}
.chip:hover { border-color: var(--accent); }
.inside-table { inline-size: 100%; border-collapse: collapse; font-size: .82rem; margin-top: .5rem; }
.inside-table th, .inside-table td { text-align: left; padding: .3rem .5rem; border-bottom: 1px solid var(--border); }
.inside-table th { color: var(--muted); font-weight: 600; white-space: nowrap; }
`;
