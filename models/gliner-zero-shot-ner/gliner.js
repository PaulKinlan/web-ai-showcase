// Front-end helpers for the GLiNER zero-shot NER page: the worker handshake, entity-type parsing, and
// highlighted-text rendering. All inference lives in worker.js (off the main thread).

export class GlinerEngine {
  constructor() {
    this.worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    this.ready = false;
    this._loadWaiters = [];
    this._pending = [];
    this.onProgress = null;
    this.device = "wasm";
    this.worker.addEventListener("message", (e) => this._onMessage(e.data));
    this.worker.addEventListener("error", (e) => {
      const err = new Error(e.message || "Worker failed to start");
      for (const w of this._loadWaiters) w.reject(err);
      this._loadWaiters = [];
      for (const p of this._pending) p.reject(err);
      this._pending = [];
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
    } else if (msg.type === "result") {
      this._pending.shift()?.resolve(msg);
    } else if (msg.type === "error") {
      const err = new Error(msg.message);
      const p = this._pending.shift();
      if (p) {
        p.reject(err); // an extract() failed
      } else {
        for (const w of this._loadWaiters) w.reject(err); // failed during load
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
  /** Extract entities of the given types from text → { spans:[{text,start,end,type,score}], ms }. */
  extract(text, entities, threshold = 0.5) {
    return new Promise((resolve, reject) => {
      this._pending.push({ resolve, reject });
      this.worker.postMessage({ type: "extract", text, entities, threshold });
    });
  }
}

/** Parse a comma-separated entity-type string into a clean, de-duplicated list. */
export function parseTypes(s) {
  return [...new Set(String(s).split(",").map((t) => t.trim().toLowerCase()).filter(Boolean))];
}

export function escapeHTML(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]),
  );
}

// Stable pastel colour per entity type (hashed hue) so the same type keeps its colour across runs.
export function typeColor(type) {
  let h = 0;
  for (let i = 0; i < type.length; i++) h = (h * 31 + type.charCodeAt(i)) % 360;
  return `hsl(${h} 70% 45%)`;
}

/** Render text with entities highlighted as coloured, labelled marks. */
export function renderHighlighted(container, text, spans) {
  const ordered = spans.slice().sort((a, b) => a.start - b.start);
  let html = "", pos = 0;
  for (const s of ordered) {
    if (s.start < pos) continue; // skip any residual overlap
    html += escapeHTML(text.slice(pos, s.start));
    const col = typeColor(s.type);
    html +=
      `<mark class="gl-ent" style="--c:${col}" title="${escapeHTML(s.type)} · ${
        (s.score * 100) | 0
      }%">` +
      `${escapeHTML(text.slice(s.start, s.end))}<sub>${escapeHTML(s.type)}</sub></mark>`;
    pos = s.end;
  }
  html += escapeHTML(text.slice(pos));
  container.innerHTML = html;
}

export const GLINER_CSS = `
  .gl-field { margin: 0.6rem 0; }
  .gl-field label { display: block; font-size: 0.85rem; margin-bottom: 0.2rem; color: var(--muted, #888); }
  .gl-field textarea, .gl-field input { width: 100%; box-sizing: border-box; padding: 0.55rem 0.65rem; border-radius: 8px; border: 1px solid #8886; font: inherit; font-size: 0.95rem; }
  .gl-field textarea { min-height: 5rem; resize: vertical; }
  .gl-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.5rem 0; }
  .gl-out { line-height: 2.1; font-size: 1.02rem; margin: 0.7rem 0; }
  .gl-ent { background: color-mix(in srgb, var(--c) 18%, transparent); border-bottom: 2px solid var(--c); border-radius: 3px; padding: 0.05em 0.15em; color: inherit; }
  .gl-ent sub { font-size: 0.6em; vertical-align: baseline; color: var(--c); font-weight: 700; margin-left: 0.25em; text-transform: uppercase; letter-spacing: 0.03em; }
  .gl-legend { display: flex; flex-wrap: wrap; gap: 0.4rem 0.9rem; margin: 0.4rem 0; font-size: 0.8rem; }
  .gl-legend span::before { content: ""; display: inline-block; width: 0.7rem; height: 0.7rem; border-radius: 2px; margin-right: 0.3rem; vertical-align: middle; background: var(--c); }
  .gl-status { font-family: var(--font-mono, monospace); font-size: 0.9rem; margin: 0.4rem 0; }
`;
