// Front-end helpers shared by every Qwen2.5-Coder (WebLLM) page: the worker handshake + streaming
// plumbing, the honest WebGPU gate/fallback, a dependency-free streaming code highlighter, and the
// widget CSS. All real inference happens in worker.js, which drives lib/webllm.js (MLC's WebGPU
// engine). We re-export the canonical webGPUAdapterAvailable() so pages gate on the exact helper the
// task mandates. The engine is parameterised by MLC model id so a page can run a second model for
// the multi-model compositions.

export { webGPUAdapterAvailable } from "/web-ai-showcase/lib/webllm.js";

const WORKER_URL = "/web-ai-showcase/models/qwen25-coder-webllm/worker.js";

export class WebLLMChatEngine {
  /** @param {string} [modelId] MLC build id — defaults to the page's primary Coder model. */
  constructor(modelId) {
    this.modelId = modelId || "Qwen2.5-Coder-1.5B-Instruct-q4f16_1-MLC";
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
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
        for (const w of this._loadWaiters) w.resolve();
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
          this._active.resolve({
            text: msg.text,
            ms: msg.ms,
            ttft: msg.ttft,
            chunks: msg.chunks,
            stats: msg.stats,
          });
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
    if (this.ready) return Promise.resolve();
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load", modelId: this.modelId });
    });
  }

  /**
   * Stream a chat completion. `req` = { messages, temperature, top_p, max_tokens }.
   * onToken(delta) fires per streamed chunk; onFirstToken(ttftMs) once.
   */
  chat(req, { onToken, onFirstToken } = {}) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._active = { id, onToken, onFirstToken, resolve, reject };
      this.worker.postMessage({ type: "run", id, req, modelId: this.modelId });
    });
  }

  /** Cooperative interrupt — WebLLM stops the decode loop and the current chat() resolves. */
  stop() {
    this.worker.postMessage({ type: "stop" });
  }
}

/** Detailed WebGPU probe used only to label the honest fallback (the gate itself uses
 *  webGPUAdapterAvailable()). Returns { ok, reason?, detail?, shaderF16? }. */
export async function probeWebGPU() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) {
    return { ok: false, reason: "no-gpu" };
  }
  let adapter = null;
  try {
    adapter = await navigator.gpu.requestAdapter();
  } catch (e) {
    return { ok: false, reason: "adapter-error", detail: String(e?.message ?? e) };
  }
  if (!adapter) return { ok: false, reason: "no-adapter" };
  return { ok: true, shaderF16: adapter.features?.has?.("shader-f16") ?? false };
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** ~4 chars per token is the usual rule of thumb — labelled as an estimate everywhere. */
export function estimateTokens(text) {
  return Math.max(1, Math.round((text || "").length / 4));
}

// A broad keyword set across the languages this page offers — the highlighter is intentionally
// language-agnostic (one tokenizer, shared keyword list) so it stays tiny and dependency-free.
const KEYWORDS = new Set([
  "abstract",
  "and",
  "as",
  "async",
  "await",
  "bool",
  "boolean",
  "break",
  "case",
  "catch",
  "char",
  "class",
  "const",
  "continue",
  "def",
  "default",
  "del",
  "do",
  "double",
  "elif",
  "else",
  "end",
  "enum",
  "export",
  "extends",
  "false",
  "final",
  "finally",
  "float",
  "fn",
  "for",
  "from",
  "func",
  "function",
  "global",
  "go",
  "if",
  "impl",
  "import",
  "in",
  "int",
  "interface",
  "is",
  "lambda",
  "let",
  "long",
  "match",
  "mut",
  "namespace",
  "new",
  "nil",
  "none",
  "not",
  "null",
  "or",
  "package",
  "pass",
  "print",
  "private",
  "protected",
  "public",
  "pub",
  "raise",
  "return",
  "select",
  "self",
  "static",
  "str",
  "string",
  "struct",
  "super",
  "switch",
  "this",
  "throw",
  "throws",
  "trait",
  "true",
  "try",
  "type",
  "typeof",
  "undefined",
  "union",
  "unsigned",
  "use",
  "using",
  "var",
  "void",
  "where",
  "while",
  "with",
  "yield",
  "SELECT",
  "FROM",
  "WHERE",
  "JOIN",
  "GROUP",
  "ORDER",
  "BY",
  "INSERT",
  "UPDATE",
  "DELETE",
  "CREATE",
  "TABLE",
  "INTO",
  "VALUES",
  "AND",
  "OR",
  "NOT",
  "NULL",
  "LEFT",
  "INNER",
  "ON",
  "AS",
  "LIMIT",
  "HAVING",
  "COUNT",
  "SUM",
  "AVG",
]);

/**
 * Dependency-free, streaming-safe syntax highlighter. Tokenises RAW code and escapes every piece as
 * it goes, so it never mangles HTML entities and never needs a library. Unclosed strings/comments
 * mid-stream simply render unhighlighted until they complete. Returns HTML for a <code> element.
 */
export function highlightCode(code) {
  const esc = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const patterns = [
    ["comment", /^(?:\/\/[^\n]*|#[^\n]*|--[^\n]*|\/\*[\s\S]*?\*\/|"""[\s\S]*?"""|'''[\s\S]*?''')/],
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

/** Pull fenced ```lang\n...``` code blocks out of an LLM reply; returns [{lang, code}] (or the whole
 *  reply as one plaintext block when the model answered with bare code / prose). */
export function extractCodeBlocks(text) {
  const s = String(text ?? "");
  const blocks = [];
  const re = /```([\w+-]*)\n?([\s\S]*?)```/g;
  let m;
  let last = 0;
  const prose = [];
  while ((m = re.exec(s))) {
    prose.push(s.slice(last, m.index).trim());
    blocks.push({ lang: (m[1] || "").toLowerCase(), code: m[2].replace(/\n$/, "") });
    last = re.lastIndex;
  }
  prose.push(s.slice(last).trim());
  return { blocks, prose: prose.filter(Boolean).join("\n\n") };
}

/** The labelled needs-WebGPU state. Never a faked reply — the real reason + how to enable, plus the
 *  ~880 MB download so a visitor knows what running it for real would cost. */
export function webgpuFallbackHTML(probe, sizeLabel = "~880 MB", vram = "~1.6 GB") {
  const reasons = {
    "no-gpu": "This browser doesn't expose the WebGPU API at all.",
    "no-adapter":
      "WebGPU is present but no GPU adapter is available here (normal in headless Chrome, many VMs, or when the GPU is blocklisted).",
    "adapter-error": "Requesting a WebGPU adapter threw an error.",
  };
  const why = reasons[probe.reason] ?? "WebGPU isn't usable here.";
  return `
    <strong>Qwen2.5-Coder runs on WebLLM, which needs WebGPU — and it isn't available in this browser.</strong>
    <p class="muted" style="margin:.4rem 0">${why}${
    probe.detail ? " (" + escapeHTML(probe.detail) + ")" : ""
  }
    WebLLM has <em>no WASM fallback</em>: a 1.5B coding model decoding token-by-token needs a real GPU,
    and the weights are a <strong>${sizeLabel}</strong> download (cached after the first load). So this
    page won't fake a completion. To run it for real:</p>
    <ul class="muted" style="margin:.2rem 0">
      <li>Open in <strong>Chrome or Edge 113+</strong> on a machine with a capable GPU (${vram} VRAM).</li>
      <li>Check <code>chrome://gpu</code> — "WebGPU" should read <em>Hardware accelerated</em>.</li>
      <li>If it's blocklisted, enable <code>chrome://flags/#enable-unsafe-webgpu</code> and relaunch.</li>
      <li>The q4f16_1 build is fastest on GPUs that report the <code>shader-f16</code> feature (most modern ones).</li>
    </ul>`;
}

export const CHAT_CSS = `
.chat-wrap { display:flex; flex-direction:column; gap:.6rem; }
.composer { display:flex; gap:.5rem; align-items:flex-end; }
.composer textarea { flex:1 1 auto; field-sizing:content; width:auto; min-block-size:2.6lh; max-block-size:10lh; resize:vertical; font-family:var(--font-mono); font-size:.82rem; }
.controls { display:flex; flex-wrap:wrap; gap:1rem 1.4rem; margin:.4rem 0 .2rem; }
.control { display:flex; flex-direction:column; gap:.2rem; font-size:.78rem; color:var(--muted); min-inline-size:9rem; }
.control .row { display:flex; align-items:center; gap:.5rem; }
.control input[type=range] { flex:1 1 auto; accent-color:var(--accent); }
.control b { color:var(--color); font-family:var(--font-mono); font-weight:600; min-inline-size:2.6rem; text-align:right; }
.control select { font:inherit; padding:.25rem .4rem; border:1px solid var(--border); border-radius:6px; background:var(--bg-raised); color:var(--color); }
.sysbox { display:flex; flex-direction:column; gap:.25rem; font-size:.82rem; }
.sysbox textarea { field-sizing:content; width:100%; min-block-size:2.2lh; max-block-size:7lh; }
.fallback { border:1px solid var(--warn); border-radius:var(--radius); background:var(--bg-raised); padding:1rem; }
.fallback code { background:var(--bg-secondary); padding:.05rem .3rem; border-radius:4px; }
.inside-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.6rem; margin:.4rem 0 .8rem; }
.stat { border:1px solid var(--border); border-radius:8px; background:var(--bg-raised); padding:.5rem .6rem; }
.stat .k { font-size:.68rem; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
.stat .v { font-family:var(--font-mono); font-size:1.05rem; color:var(--color); }
.stat .v small { font-size:.7rem; color:var(--muted); }
.wire { font-family:var(--font-mono); font-size:.76rem; white-space:pre-wrap; word-break:break-word; max-block-size:20rem; overflow:auto; }
.chip { font:inherit; font-size:.78rem; padding:.2rem .6rem; border-radius:999px; border:1px solid var(--border); background:var(--bg-raised); color:var(--color); cursor:pointer; text-align:left; }
.chip:hover { border-color:var(--accent); }
.chip[aria-pressed=true] { border-color:var(--accent); background:var(--accent); color:var(--accent-ink); }
.chips { display:flex; flex-wrap:wrap; gap:.4rem; margin:.2rem 0; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.76rem; color:var(--muted); margin-top:.5rem; }
.readout b { color:var(--color); font-weight:600; }
.caret { display:inline-block; inline-size:.5rem; block-size:1em; vertical-align:text-bottom; background:currentColor; animation:blink 1s steps(2) infinite; }
@keyframes blink { 50% { opacity:0; } }
@media (prefers-reduced-motion: reduce) { .caret { animation:none; } }
/* Code output — monospace, scrollable, syntax-highlighted. */
.codeout { position:relative; border:1px solid var(--border); border-radius:8px; background:var(--bg-raised); overflow:hidden; }
.codeout-bar { display:flex; align-items:center; justify-content:space-between; gap:.5rem; padding:.35rem .6rem; border-bottom:1px solid var(--border); background:var(--bg-secondary); font-size:.72rem; color:var(--muted); }
.codeout-bar .lang { font-family:var(--font-mono); text-transform:uppercase; letter-spacing:.05em; }
.codeout pre { margin:0; padding:.7rem .8rem; overflow:auto; max-block-size:30rem; }
.codeout code { font-family:var(--font-mono); font-size:.82rem; line-height:1.55; white-space:pre; display:block; }
.copybtn { font:inherit; font-size:.72rem; padding:.15rem .5rem; border:1px solid var(--border); border-radius:6px; background:var(--bg-raised); color:var(--color); cursor:pointer; }
.copybtn:hover { border-color:var(--accent); }
.tok-k { color:var(--accent); font-weight:600; }
.tok-s { color:#0a7d3c; }
.tok-c { color:var(--muted); font-style:italic; }
.tok-n { color:#b5651d; }
@media (prefers-color-scheme: dark) {
  .tok-s { color:#7ee2a8; }
  .tok-n { color:#e0a86b; }
}
:root[data-theme=dark] .tok-s { color:#7ee2a8; }
:root[data-theme=dark] .tok-n { color:#e0a86b; }
:root[data-theme=light] .tok-s { color:#0a7d3c; }
:root[data-theme=light] .tok-n { color:#b5651d; }
.proseout { white-space:pre-wrap; line-height:1.6; margin:.4rem 0; }
`;
