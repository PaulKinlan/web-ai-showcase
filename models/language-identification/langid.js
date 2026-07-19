// Front-end helpers for the XLM-RoBERTa language-identification pages. Owns the worker handshake and
// the renderers (top result, per-language probability bars). All inference lives in worker.js.

const WORKER_URL = "/web-ai-showcase/models/language-identification/worker.js";

// The model's 20 label codes → human names + a flag-ish emoji, purely for display.
export const LANGS = {
  ar: { name: "Arabic", flag: "🇸🇦" },
  bg: { name: "Bulgarian", flag: "🇧🇬" },
  de: { name: "German", flag: "🇩🇪" },
  el: { name: "Greek", flag: "🇬🇷" },
  en: { name: "English", flag: "🇬🇧" },
  es: { name: "Spanish", flag: "🇪🇸" },
  fr: { name: "French", flag: "🇫🇷" },
  hi: { name: "Hindi", flag: "🇮🇳" },
  it: { name: "Italian", flag: "🇮🇹" },
  ja: { name: "Japanese", flag: "🇯🇵" },
  nl: { name: "Dutch", flag: "🇳🇱" },
  pl: { name: "Polish", flag: "🇵🇱" },
  pt: { name: "Portuguese", flag: "🇵🇹" },
  ru: { name: "Russian", flag: "🇷🇺" },
  sw: { name: "Swahili", flag: "🇰🇪" },
  th: { name: "Thai", flag: "🇹🇭" },
  tr: { name: "Turkish", flag: "🇹🇷" },
  ur: { name: "Urdu", flag: "🇵🇰" },
  vi: { name: "Vietnamese", flag: "🇻🇳" },
  zh: { name: "Chinese", flag: "🇨🇳" },
};

export function langName(code) {
  return LANGS[code]?.name || code;
}
export function langFlag(code) {
  return LANGS[code]?.flag || "🏳️";
}

export class LangIdEngine {
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

  /** Detect → { text, scores:[{label,score} …sorted], ms, device }. */
  detect(text) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, text });
    });
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Big headline result: flag + language name + confidence, with an honest "uncertain" hint. */
export function renderTop(container, scores) {
  const top = scores[0];
  const conf = (top.score * 100).toFixed(1);
  const uncertain = top.score < 0.6;
  container.innerHTML = `<div class="li-top ${uncertain ? "unsure" : ""}">` +
    `<span class="li-flag">${langFlag(top.label)}</span>` +
    `<span class="li-name">${langName(top.label)}</span>` +
    `<span class="li-code">${top.label}</span>` +
    `<span class="li-conf">${conf}%</span></div>` +
    (uncertain
      ? `<p class="li-hint muted">Low confidence — the text is short, mixed, or outside the 20 supported languages.</p>`
      : "");
}

/** Per-language probability bars for the top-N languages (the softmax distribution). */
export function renderBars(container, scores, topN = 6) {
  const top = scores.slice(0, topN);
  container.replaceChildren(...top.map((s, i) => {
    const row = document.createElement("div");
    row.className = "li-bar-row" + (i === 0 ? " top" : "");
    const pct = (s.score * 100).toFixed(1);
    row.innerHTML = `<span class="li-bar-label">${langFlag(s.label)} ${langName(s.label)}</span>` +
      `<span class="li-bar-track"><span class="li-bar-fill" style="inline-size:${pct}%"></span></span>` +
      `<span class="li-bar-pct">${pct}%</span>`;
    return row;
  }));
}

export const LANGID_CSS = `
.li-top { display: flex; align-items: baseline; gap: .6rem; flex-wrap: wrap; margin-top: .5rem;
  padding: .7rem .9rem; border: 1px solid var(--accent); border-inline-start: 4px solid var(--accent);
  border-radius: 10px; background: var(--bg-raised); }
.li-top.unsure { border-color: var(--border-strong); border-inline-start-color: var(--warn); }
.li-flag { font-size: 1.8rem; line-height: 1; }
.li-name { font-family: var(--font-display); font-size: 1.4rem; }
.li-code { font-family: var(--font-mono); font-size: .8rem; color: var(--muted); }
.li-conf { margin-inline-start: auto; font-family: var(--font-mono); font-size: 1.1rem; font-weight: 600; }
.li-hint { font-size: .82rem; margin: .4rem 0 0; }
.li-bars { display: flex; flex-direction: column; gap: .4rem; margin-top: .5rem; }
.li-bar-row { display: grid; grid-template-columns: minmax(7rem, 9rem) 1fr auto; gap: .6rem; align-items: center; }
.li-bar-label { font-size: .82rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.li-bar-track { block-size: .7rem; background: var(--bg-secondary); border: 1px solid var(--border);
  border-radius: 999px; overflow: hidden; }
.li-bar-fill { display: block; block-size: 100%; background: var(--accent); }
.li-bar-row.top .li-bar-fill { background: var(--accent); }
.li-bar-pct { font-family: var(--font-mono); font-size: .76rem; color: var(--muted); min-inline-size: 3.2rem; text-align: end; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.chip:hover { border-color: var(--accent); }
`;
