// Front-end helpers for the M2M100 translation pages. Owns the worker handshake, streaming, the
// language table, and render helpers. All inference lives in worker.js (off the main thread).

const WORKER_URL = "/web-ai-showcase/models/m2m100-translation/worker.js";

// A curated subset of M2M100's 100 languages (code → English name). M2M100 uses short language codes
// and translates directly between any pair — no English pivot.
export const LANGS = [
  ["en", "English"],
  ["fr", "French"],
  ["es", "Spanish"],
  ["de", "German"],
  ["it", "Italian"],
  ["pt", "Portuguese"],
  ["nl", "Dutch"],
  ["ru", "Russian"],
  ["zh", "Chinese"],
  ["ja", "Japanese"],
  ["ko", "Korean"],
  ["ar", "Arabic"],
  ["hi", "Hindi"],
  ["tr", "Turkish"],
  ["pl", "Polish"],
  ["uk", "Ukrainian"],
  ["vi", "Vietnamese"],
  ["id", "Indonesian"],
  ["sv", "Swedish"],
  ["fi", "Finnish"],
  ["el", "Greek"],
  ["he", "Hebrew"],
  ["th", "Thai"],
  ["cs", "Czech"],
  ["ro", "Romanian"],
  ["da", "Danish"],
  ["hu", "Hungarian"],
  ["fa", "Persian"],
  ["bn", "Bengali"],
  ["ca", "Catalan"],
];

export const LANG_NAME = Object.fromEntries(LANGS);

/** Fill a <select> with the language options; optionally select `code`. */
export function fillLangSelect(select, code) {
  select.replaceChildren(...LANGS.map(([c, name]) => {
    const o = document.createElement("option");
    o.value = c;
    o.textContent = `${name} (${c})`;
    if (c === code) o.selected = true;
    return o;
  }));
}

export class TranslateEngine {
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

  /** Translate text. opts: { numBeams }. onStream(partial). */
  translate(text, srcLang, tgtLang, opts = {}, onStream) {
    const id = ++this._id;
    if (onStream) this._streams.set(id, onStream);
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, text, srcLang, tgtLang, opts });
    });
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Render the language/token/timing readout for a completed translation. */
export function renderStats(els, r) {
  if (els.codes) els.codes.textContent = `${r.srcLang} → ${r.tgtLang}`;
  if (els.inTok) els.inTok.textContent = r.inTokens;
  if (els.outTok) els.outTok.textContent = r.outTokens;
  if (els.backend) els.backend.textContent = r.device.toUpperCase();
  if (els.ms) els.ms.textContent = (r.ms / 1000).toFixed(2) + " s";
  if (els.toksec) {
    const tps = r.ms > 0 ? (r.outTokens / (r.ms / 1000)) : 0;
    els.toksec.textContent = tps ? tps.toFixed(1) + " tok/s" : "–";
  }
}

export const TRANSLATE_CSS = `
.tr-io { display:grid; gap:1rem; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); }
.tr-io textarea { inline-size:100%; min-block-size:130px; resize:vertical; font-family:var(--font-body); }
.tr-out { border:1px solid var(--border); border-radius:var(--radius); background:var(--bg-raised);
  padding:.8rem 1rem; min-block-size:130px; white-space:pre-wrap; font-size:1.05rem; }
.tr-out:empty::before { content:"The translation will stream in here."; color:var(--muted); font-size:.95rem; }
.lang-row { display:grid; gap:.9rem 1.2rem; grid-template-columns:1fr auto 1fr; align-items:end; margin:.6rem 0; }
.lang-row label { display:flex; flex-direction:column; gap:.3rem; font-size:.82rem; }
.lang-row select { inline-size:100%; }
.swap-btn { align-self:end; }
.controls-grid { display:grid; gap:.9rem 1.2rem; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); align-items:end; margin:.6rem 0; }
.controls-grid label { display:flex; flex-direction:column; gap:.3rem; font-size:.82rem; }
.stat-row { display:grid; gap:.6rem; grid-template-columns:repeat(auto-fit, minmax(110px, 1fr)); margin-top:.6rem; }
.stat { border:1px solid var(--border); border-radius:8px; background:var(--bg-secondary); padding:.5rem .7rem; }
.stat .k { font-family:var(--font-mono); font-size:.66rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
.stat .v { font-family:var(--font-display); font-size:1.3rem; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem; color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.chip { font:inherit; font-size:.78rem; padding:.2rem .6rem; border-radius:999px; border:1px solid var(--border);
  background:var(--bg-raised); color:var(--color); cursor:pointer; }
.chip:hover { border-color:var(--accent); }
.telephone-step { border:1px solid var(--border); border-inline-start:4px solid var(--accent); border-radius:8px;
  background:var(--bg-raised); padding:.5rem .7rem; margin-top:.5rem; }
.telephone-step .meta { font-family:var(--font-mono); font-size:.72rem; color:var(--muted); }
.diff-same { color:var(--good); } .diff-drift { color:var(--warn); }
`;
