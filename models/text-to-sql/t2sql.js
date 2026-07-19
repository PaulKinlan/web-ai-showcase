// Front-end helpers for the Text-to-SQL pages. Keeps each page thin: it owns the worker handshake,
// streaming, prompt composition, and the render helpers. All inference lives in worker.js (off-thread).
//
// The model is SCHEMA-CONDITIONED: buildPrompt() is the single source of truth for the exact string fed
// to the encoder, so what the "see inside" surface shows is byte-for-byte what actually runs.

const WORKER_URL = "/web-ai-showcase/models/text-to-sql/worker.js";

export class SqlEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.onProgress = null;
    this._loadWaiters = [];
    this._pending = new Map();
    this._streams = new Map();
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
    } else if (msg.type === "stream") {
      this._streams.get(msg.id)?.(msg.text);
    } else if (msg.type === "result") {
      const p = this._pending.get(msg.id);
      if (p) {
        this._pending.delete(msg.id);
        this._streams.delete(msg.id);
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

  /** Run one composed prompt string. opts: { maxNewTokens, numBeams, doSample }. onStream(partial). */
  run(input, opts = {}, onStream) {
    const id = ++this._id;
    if (onStream) this._streams.set(id, onStream);
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, input, opts });
    });
  }
}

/**
 * Compose the EXACT string the model was trained on. The schema textarea may hold several CREATE TABLE
 * statements (newlines or semicolons between them); we normalise them onto one line so the format matches
 * the training data, then append the natural-language question after "query for:".
 */
export function buildPrompt(schema, question) {
  const tables = String(schema || "")
    .split(/\n+/)
    .map((s) => s.trim())
    .filter(Boolean)
    .join(" ");
  return `tables:\n${tables}\nquery for:${String(question || "").trim()}`;
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Render the fed prompt with the schema block and the question block visually distinguished. */
export function renderFedPrompt(container, prompt) {
  const idx = prompt.indexOf("query for:");
  container.replaceChildren();
  if (idx === -1) {
    container.textContent = prompt;
    return;
  }
  const schemaPart = prompt.slice(0, idx);
  const queryPart = prompt.slice(idx);
  const s = document.createElement("span");
  s.className = "fed-schema";
  s.textContent = schemaPart;
  const q = document.createElement("span");
  q.className = "fed-query";
  q.textContent = queryPart;
  container.append(s, q);
}

/** Render input token strings as chips, exposing T5's SentencePiece ▁ word-boundary marks. */
export function renderTokens(container, tokenStrings) {
  container.replaceChildren(...(tokenStrings || []).map((t) => {
    const chip = document.createElement("span");
    chip.className = "tok";
    chip.textContent = t.replace(/▁/g, "·");
    if (/^(<|>)/.test(t) || /^<.*>$/.test(t)) chip.classList.add("tok-special");
    return chip;
  }));
}

/** Rough SQL keyword highlighter for read-only display of generated queries (no eval, pure formatting). */
const SQL_KEYWORDS =
  /\b(SELECT|FROM|WHERE|GROUP BY|ORDER BY|HAVING|LIMIT|JOIN|LEFT JOIN|RIGHT JOIN|INNER JOIN|OUTER JOIN|ON|AS|AND|OR|NOT|IN|IS|NULL|COUNT|SUM|AVG|MIN|MAX|DISTINCT|BETWEEN|LIKE|ASC|DESC|UNION|ALL|EXISTS|CASE|WHEN|THEN|ELSE|END)\b/gi;

export function highlightSQL(sql) {
  return escapeHTML(sql).replace(SQL_KEYWORDS, (m) => `<span class="sql-kw">${m}</span>`);
}

/** Draw the real per-token decode timeline as a bar chart of inter-token intervals (ms). */
export function drawTimeline(canvas, intervals) {
  const cs = getComputedStyle(document.body);
  const accent = cs.getPropertyValue("--accent").trim() || "#4b3aff";
  const muted = cs.getPropertyValue("--muted").trim() || "#888";
  const dpr = self.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600;
  const h = canvas.clientHeight || 90;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!intervals || intervals.length === 0) {
    ctx.fillStyle = muted;
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillText("No per-token timeline (beam search).", 8, h / 2);
    return;
  }
  const max = Math.max(...intervals, 1);
  const n = intervals.length;
  const bw = Math.max(1, w / n);
  for (let i = 0; i < n; i++) {
    const bh = Math.max(1, (intervals[i] / max) * (h - 6));
    ctx.fillStyle = accent;
    ctx.globalAlpha = 0.85;
    ctx.fillRect(i * bw, h - bh, Math.max(1, bw - 1), bh);
  }
  ctx.globalAlpha = 1;
  ctx.fillStyle = muted;
  ctx.font = "11px system-ui, sans-serif";
  ctx.fillText(`${n} tokens · peak ${max} ms/token`, 6, 12);
}

/** Copy text to the clipboard, returning true on success. Falls back to a hidden textarea + execCommand. */
export async function copyText(text) {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch { /* fall through to legacy path */ }
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.append(ta);
    ta.select();
    const ok = document.execCommand("copy");
    ta.remove();
    return ok;
  } catch {
    return false;
  }
}

export const SQL_CSS = `
.sql-io textarea, .sql-io input[type=text] { inline-size:100%; font-family:var(--font-mono); }
.schema-box { inline-size:100%; font-family:var(--font-mono); font-size:.82rem; min-inline-size:0; }
.sql-out-wrap { position:relative; margin-top:.3rem; }
.sql-out { border:1px solid var(--border); border-radius:var(--radius); background:var(--bg-raised);
  padding:.8rem 1rem; min-block-size:52px; overflow-x:auto; min-inline-size:0; }
.sql-out pre { margin:0; font-family:var(--font-mono); font-size:.88rem; white-space:pre-wrap;
  word-break:break-word; }
.sql-out:empty::before { content:"The generated SQL will stream in here."; color:var(--muted);
  font-family:var(--font-body); }
.sql-kw { color:var(--accent); font-weight:600; }
.copy-btn { font:inherit; font-size:.74rem; padding:.2rem .55rem; border-radius:6px;
  border:1px solid var(--border); background:var(--bg-secondary); color:var(--color); cursor:pointer; }
.copy-btn:hover { border-color:var(--accent); }
.copy-btn:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.fed-input { border:1px dashed var(--border-strong); border-radius:8px; background:var(--bg-secondary);
  padding:.5rem .7rem; font-family:var(--font-mono); font-size:.82rem; white-space:pre-wrap;
  word-break:break-word; overflow-x:auto; min-inline-size:0; }
.fed-schema { color:var(--muted); }
.fed-query { color:var(--color); background:color-mix(in srgb, var(--accent) 14%, transparent);
  border-radius:3px; }
.controls-grid { display:grid; gap:.9rem 1.2rem; grid-template-columns:repeat(auto-fit, minmax(160px, 1fr));
  align-items:end; margin:.6rem 0; }
.controls-grid label { display:flex; flex-direction:column; gap:.3rem; font-size:.82rem; }
.controls-grid input[type=range] { inline-size:100%; }
.controls-grid .val { font-family:var(--font-mono); color:var(--muted); font-size:.78rem; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem;
  color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.stat-row { display:grid; gap:.6rem; grid-template-columns:repeat(auto-fit, minmax(120px, 1fr));
  margin-top:.6rem; }
.stat { border:1px solid var(--border); border-radius:8px; background:var(--bg-secondary); padding:.5rem .7rem; }
.stat .k { font-family:var(--font-mono); font-size:.68rem; color:var(--muted); text-transform:uppercase;
  letter-spacing:.04em; }
.stat .v { font-family:var(--font-display); font-size:1.4rem; }
.tok-wrap { display:flex; flex-wrap:wrap; gap:3px; margin-top:.4rem; }
.tok { font-family:var(--font-mono); font-size:.75rem; padding:.1rem .35rem; border-radius:4px;
  border:1px solid var(--border); background:var(--bg-raised); word-break:break-all; }
.tok-special { color:var(--muted); border-style:dashed; }
.timeline { inline-size:100%; block-size:90px; display:block; }
.preset-grid { display:grid; gap:.6rem; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); margin:.6rem 0; }
.preset { text-align:start; border:1px solid var(--border); border-radius:8px; background:var(--bg-raised);
  padding:.55rem .7rem; cursor:pointer; font:inherit; }
.preset:hover { border-color:var(--accent); }
.preset:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.preset .t { font-weight:600; font-size:.9rem; } .preset .d { color:var(--muted); font-size:.78rem;
  font-family:var(--font-mono); }
.chip { font:inherit; font-size:.78rem; padding:.2rem .6rem; border-radius:999px; border:1px solid var(--border);
  background:var(--bg-raised); color:var(--color); cursor:pointer; }
.chip:hover { border-color:var(--accent); }
.chip:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.schema-tables { display:flex; flex-wrap:wrap; gap:.4rem; margin:.3rem 0; }
.tbl-pill { font-family:var(--font-mono); font-size:.74rem; padding:.15rem .5rem; border-radius:999px;
  border:1px solid var(--border); background:var(--bg-raised); }
.tbl-pill b { color:var(--accent); }
`;
