// Front-end helpers shared by every StarCoder-FIM page: the worker handshake + streaming plumbing, a
// dependency-free streaming code highlighter, the FIM "wire" renderer, and the widget CSS. All real
// inference happens in worker.js (Transformers.js text-generation on the StarCoder-family
// tiny_starcoder_py). One engine drives all pages; the multi-model page adds a second embed worker.

const WORKER_URL = "/web-ai-showcase/models/starcoder-fim/worker.js";

export const FIM = { PREFIX: "<fim_prefix>", SUFFIX: "<fim_suffix>", MIDDLE: "<fim_middle>" };

export class StarCoderEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.onProgress = null;
    this._loadWaiters = [];
    this._active = null; // { id, onToken, onFirstToken, resolve, reject }
    this._id = 0;
    this.worker.addEventListener("message", (e) => this._onMessage(e.data));
    this.worker.addEventListener("error", (e) => {
      this._rejectAll(new Error(e.message || "Worker failed to start"));
    });
  }

  _rejectAll(err) {
    for (const w of this._loadWaiters) w.reject(err);
    this._loadWaiters = [];
    if (this._active) {
      this._active.reject(err);
      this._active = null;
    }
  }

  _onMessage(msg) {
    switch (msg.type) {
      case "progress":
        this.onProgress?.(msg.p);
        break;
      case "ready":
        this.ready = true;
        this.device = msg.device;
        for (const w of this._loadWaiters) w.resolve(msg.device);
        this._loadWaiters = [];
        break;
      case "first":
        if (this._active && this._active.id === msg.id) this._active.onFirstToken?.(msg.t);
        break;
      case "token":
        if (this._active && this._active.id === msg.id) this._active.onToken?.(msg.delta);
        break;
      case "done":
        if (this._active && this._active.id === msg.id) {
          this._active.resolve(msg);
          this._active = null;
        }
        break;
      case "error":
        if (this._active && this._active.id === msg.id) {
          this._active.reject(new Error(msg.message));
          this._active = null;
        } else {
          this._rejectAll(new Error(msg.message));
        }
        break;
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
   * Run a FIM or plain completion. `opts` = { mode:"fim"|"complete", prefix, suffix, code, maxTokens,
   * temperature }. Resolves { middle, wire, ms, ttft, tokens, device }.
   */
  run(opts, { onToken, onFirstToken } = {}) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._active = { id, onToken, onFirstToken, resolve, reject };
      this.worker.postMessage({ type: "run", id, opts });
    });
  }

  /** Best-effort cooperative interrupt (InterruptableStoppingCriteria in the worker). */
  stop() {
    this.worker.postMessage({ type: "stop" });
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// A small, language-agnostic keyword set (this model is Python-first but the highlighter stays generic
// so the multi-language page reads well too). Dependency-free.
const KEYWORDS = new Set(
  ("and as assert async await break class continue def del elif else except False finally for from " +
    "global if import in is lambda None nonlocal not or pass raise return True try while with yield " +
    "const let var function return class extends new this typeof void null undefined true false " +
    "fn let mut pub struct impl match trait enum use where fold map filter self int str bool float " +
    "func package interface range go defer chan select public static void print println println! " +
    "System out Vec Some None Ok Err")
    .split(/\s+/),
);

/** Dependency-free, streaming-safe highlighter. Tokenises RAW code and escapes as it goes. */
export function highlightCode(code) {
  const esc = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const patterns = [
    ["comment", /^(?:#[^\n]*|\/\/[^\n]*|--[^\n]*|\/\*[\s\S]*?\*\/|"""[\s\S]*?"""|'''[\s\S]*?''')/],
    ["string", /^(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)/],
    ["number", /^\b(?:0[xXbBoO][0-9a-fA-F_]+|\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?)\b/],
    ["ident", /^[A-Za-z_$][\w$]*/],
    ["other", /^[\s\S]/],
  ];
  let out = "";
  let s = code;
  let guard = 0;
  while (s.length && guard++ < 100000) {
    let matched = false;
    for (const [type, re] of patterns) {
      const m = re.exec(s);
      if (!m) continue;
      const tok = m[0];
      if (type === "comment") out += `<span class="tok-c">${esc(tok)}</span>`;
      else if (type === "string") out += `<span class="tok-s">${esc(tok)}</span>`;
      else if (type === "number") out += `<span class="tok-n">${esc(tok)}</span>`;
      else if (type === "ident") {
        out += KEYWORDS.has(tok) ? `<span class="tok-k">${esc(tok)}</span>` : esc(tok);
      } else out += esc(tok);
      s = s.slice(tok.length);
      matched = true;
      break;
    }
    if (!matched) {
      out += esc(s[0]);
      s = s.slice(1);
    }
  }
  return out;
}

/**
 * Render a code block whose middle segment (the model's FIM completion) is visibly highlighted between
 * the fixed prefix/suffix. `middle` may be streaming, so a caret is appended while `streaming`.
 */
export function renderFilled(el, prefix, middle, suffix, streaming) {
  el.innerHTML = highlightCode(prefix) +
    `<span class="fim-mid">${highlightCode(middle)}${
      streaming ? '<span class="caret"></span>' : ""
    }</span>` +
    highlightCode(suffix);
}

/** Render plain streaming code (basics/complete): highlighted, with a streaming caret. */
export function renderCode(el, code, streaming) {
  el.innerHTML = highlightCode(code) + (streaming ? '<span class="caret"></span>' : "");
}

/** Render the exact FIM wire (the sentinel-interleaved string the model saw) with the tokens marked. */
export function renderWire(el, wire) {
  const marked = escapeHTML(wire).replace(
    /(&lt;fim_(?:prefix|suffix|middle|pad)&gt;|&lt;\|endoftext\|&gt;)/g,
    '<span class="sentinel">$1</span>',
  );
  el.innerHTML = marked || '<span class="muted">Run something to see the FIM wire.</span>';
}

export const STARCODER_CSS = `
.sc-wrap { display:flex; flex-direction:column; gap:.6rem; }
.controls { display:flex; flex-wrap:wrap; gap:1rem 1.4rem; margin:.4rem 0 .2rem; }
.control { display:flex; flex-direction:column; gap:.2rem; font-size:.78rem; color:var(--muted); min-inline-size:9rem; }
.control .row { display:flex; align-items:center; gap:.5rem; }
.control input[type=range] { flex:1 1 auto; accent-color:var(--accent); min-inline-size:0; }
.control b { color:var(--color); font-family:var(--font-mono); font-weight:600; min-inline-size:2.6rem; text-align:right; }
.control select { font:inherit; padding:.25rem .4rem; border:1px solid var(--border); border-radius:6px; background:var(--bg-raised); color:var(--color); }
.fim-fields { display:grid; gap:.5rem; }
.fim-fields textarea, .code-input { field-sizing:content; inline-size:100%; min-block-size:2.4lh; max-block-size:12lh; resize:vertical; font-family:var(--font-mono); font-size:.82rem; padding:.5rem .6rem; border:1px solid var(--border); border-radius:8px; background:var(--bg-raised); color:var(--color); }
.fieldlabel { font-size:.74rem; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin-block-end:.15rem; display:block; }
.actionbar { display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; }
.actionbar button[type=button].secondary, .actionbar .secondary { }
.chip { font:inherit; font-size:.78rem; padding:.25rem .6rem; border-radius:999px; border:1px solid var(--border); background:var(--bg-raised); color:var(--color); cursor:pointer; text-align:left; min-block-size:2.1rem; }
.chip:hover { border-color:var(--accent); }
.chip[aria-pressed=true] { border-color:var(--accent); background:var(--accent); color:var(--accent-ink); }
.chips { display:flex; flex-wrap:wrap; gap:.4rem; margin:.2rem 0; }
.codeout { position:relative; border:1px solid var(--border); border-radius:8px; background:var(--bg-raised); overflow:hidden; }
.codeout-bar { display:flex; align-items:center; justify-content:space-between; gap:.5rem; padding:.35rem .6rem; border-bottom:1px solid var(--border); background:var(--bg-secondary); font-size:.72rem; color:var(--muted); }
.codeout-bar .lang { font-family:var(--font-mono); text-transform:uppercase; letter-spacing:.05em; }
.codeout pre { margin:0; padding:.7rem .8rem; overflow-x:auto; max-block-size:30rem; }
.codeout code { font-family:var(--font-mono); font-size:.82rem; line-height:1.55; white-space:pre; display:block; }
.copybtn { font:inherit; font-size:.72rem; padding:.2rem .55rem; border:1px solid var(--border); border-radius:6px; background:var(--bg-raised); color:var(--color); cursor:pointer; min-block-size:2rem; }
.copybtn:hover { border-color:var(--accent); }
.fim-mid { background:color-mix(in srgb, var(--accent) 18%, transparent); border-radius:3px; box-shadow:0 0 0 1px color-mix(in srgb, var(--accent) 35%, transparent); }
.tok-k { color:var(--accent); font-weight:600; }
.tok-s { color:#0a7d3c; }
.tok-c { color:var(--muted); font-style:italic; }
.tok-n { color:#b5651d; }
@media (prefers-color-scheme: dark) { .tok-s { color:#7ee2a8; } .tok-n { color:#e0a86b; } }
:root[data-theme=dark] .tok-s { color:#7ee2a8; }
:root[data-theme=dark] .tok-n { color:#e0a86b; }
:root[data-theme=light] .tok-s { color:#0a7d3c; }
:root[data-theme=light] .tok-n { color:#b5651d; }
.wire { font-family:var(--font-mono); font-size:.78rem; white-space:pre-wrap; word-break:break-word; max-block-size:20rem; overflow:auto; padding:.6rem .7rem; border:1px solid var(--border); border-radius:8px; background:var(--bg-raised); }
.wire .sentinel { color:var(--accent); font-weight:700; background:color-mix(in srgb, var(--accent) 12%, transparent); border-radius:3px; padding:0 .1rem; }
.inside-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:.6rem; margin:.4rem 0 .8rem; }
.stat { border:1px solid var(--border); border-radius:8px; background:var(--bg-raised); padding:.5rem .6rem; }
.stat .k { font-size:.68rem; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
.stat .v { font-family:var(--font-mono); font-size:1.05rem; color:var(--color); }
.stat .v small { font-size:.7rem; color:var(--muted); }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.76rem; color:var(--muted); margin-top:.5rem; }
.readout b { color:var(--color); font-weight:600; }
.caret { display:inline-block; inline-size:.5rem; block-size:1em; vertical-align:text-bottom; background:currentColor; animation:blink 1s steps(2) infinite; }
@keyframes blink { 50% { opacity:0; } }
@media (prefers-reduced-motion: reduce) { .caret { animation:none; } }
.fallback { border:1px solid var(--warn); border-radius:var(--radius); background:var(--bg-raised); padding:1rem; }
.fallback code { background:var(--bg-secondary); padding:.05rem .3rem; border-radius:4px; }
`;
