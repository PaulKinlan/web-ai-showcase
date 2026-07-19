// Front-end helpers for the T5-base (task-prefix seq2seq) pages. Keeps each page thin: it owns the
// worker handshake, streaming, the TASK-PREFIX table (the star of the demo), and the render helpers.
// All inference lives in worker.js (off the main thread).
//
// The mechanism this demo teaches: the ORIGINAL T5 is one text-to-text model that does many jobs, and
// the ONLY thing that tells it which job is a fixed TASK PREFIX prepended to the input. The TASKS table
// below is the single source of truth for those prefixes — the same builder produces the exact string
// shown in "see inside" and the string fed to the encoder in worker.js, so what you read is what runs.

const WORKER_URL = "/web-ai-showcase/models/t5-base-text2text/worker.js";

/**
 * The task-prefix catalogue. Each task is a real prefix T5-base was pre-trained on. `build(v)` turns
 * the user's field values into the exact model input string; `outputHint` describes the expected shape
 * so the page can label the answer honestly (e.g. CoLA emits acceptable/unacceptable, STS-B a number).
 */
export const TASKS = [
  {
    id: "translate-de",
    label: "Translate → German",
    group: "translate",
    blurb: "English in, German out",
    fields: [{ key: "text", kind: "textarea", label: "English text", sample: "The house is wonderful and the garden is quiet." }],
    build: (v) => `translate English to German: ${v.text || ""}`,
    outputHint: "German translation",
  },
  {
    id: "translate-fr",
    label: "Translate → French",
    group: "translate",
    blurb: "English in, French out",
    fields: [{ key: "text", kind: "textarea", label: "English text", sample: "Machine learning now runs directly in the browser." }],
    build: (v) => `translate English to French: ${v.text || ""}`,
    outputHint: "French translation",
  },
  {
    id: "translate-ro",
    label: "Translate → Romanian",
    group: "translate",
    blurb: "English in, Romanian out",
    fields: [{ key: "text", kind: "textarea", label: "English text", sample: "Good morning, welcome to the conference." }],
    build: (v) => `translate English to Romanian: ${v.text || ""}`,
    outputHint: "Romanian translation",
  },
  {
    id: "summarize",
    label: "Summarize",
    group: "summarize",
    blurb: "long text → shorter text",
    fields: [{
      key: "text",
      kind: "textarea",
      label: "Text to summarize",
      sample:
        "The web platform keeps gaining new capabilities. Browsers can now run machine learning models locally, on the user's own device. That means data never leaves the tab, there is no per-request server cost, and the model keeps working offline once it is cached. Developers get privacy and lower latency without standing up any inference infrastructure.",
    }],
    build: (v) => `summarize: ${v.text || ""}`,
    outputHint: "one-or-two-sentence summary",
  },
  {
    id: "cola",
    label: "Grammar check (CoLA)",
    group: "classify",
    blurb: "is this sentence grammatical?",
    fields: [{ key: "text", kind: "input", label: "A sentence to judge", sample: "The books is on the table." }],
    build: (v) => `cola sentence: ${v.text || ""}`,
    outputHint: 'the single word "acceptable" or "unacceptable"',
  },
  {
    id: "stsb",
    label: "Similarity (STS-B)",
    group: "score",
    blurb: "how alike are two sentences? (0–5)",
    fields: [
      { key: "a", kind: "input", label: "Sentence 1", sample: "A man is playing a guitar." },
      { key: "b", kind: "input", label: "Sentence 2", sample: "A person plays the guitar." },
    ],
    build: (v) => `stsb sentence1: ${v.a || ""} sentence2: ${v.b || ""}`,
    outputHint: "a similarity score from 0.0 (unrelated) to 5.0 (identical meaning)",
  },
];

export function taskById(id) {
  return TASKS.find((t) => t.id === id) || TASKS[0];
}

/** Build the exact prefixed input string for a task from its field values. Shared by UI + worker. */
export function buildInput(taskId, values) {
  return taskById(taskId).build(values || {});
}

export class T5Engine {
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

  /** Run one prefixed input. opts: { maxNewTokens, numBeams, doSample, temperature, topK }. onStream(partial). */
  run(input, opts = {}, onStream) {
    const id = ++this._id;
    if (onStream) this._streams.set(id, onStream);
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, input, opts });
    });
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

export function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

/** Render the input token strings as chips, exposing T5's SentencePiece ▁ word-boundary marks. */
export function renderTokens(container, tokenStrings) {
  container.replaceChildren(...(tokenStrings || []).map((t) => {
    const chip = document.createElement("span");
    chip.className = "tok";
    // ▁ marks a leading space in SentencePiece; show it as a visible middot for clarity.
    chip.textContent = t.replace(/▁/g, "·");
    if (/^<.*>$/.test(t)) chip.classList.add("tok-special");
    return chip;
  }));
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

/** Fill a readout row of {backend, ms, toksec, inTok, outTok} elements from a result. */
export function renderStats(els, r) {
  if (els.backend) els.backend.textContent = r.device.toUpperCase();
  if (els.ms) els.ms.textContent = (r.ms / 1000).toFixed(2) + " s";
  if (els.toksec) {
    const tps = r.ms > 0 ? (r.outTokens / (r.ms / 1000)) : 0;
    els.toksec.textContent = tps ? tps.toFixed(1) + " tok/s" : "–";
  }
  if (els.inTok) els.inTok.textContent = r.inTokens;
  if (els.outTok) els.outTok.textContent = r.outTokens;
}

export const T5_CSS = `
.t5-io textarea, .t5-io input[type=text] { inline-size:100%; font-family:var(--font-body); box-sizing:border-box; }
.t5-fields { display:flex; flex-direction:column; gap:.6rem; margin:.6rem 0; }
.t5-fields label { display:flex; flex-direction:column; gap:.3rem; font-size:.82rem; }
.t5-out { border:1px solid var(--border); border-radius:var(--radius); background:var(--bg-raised);
  padding:.8rem 1rem; min-block-size:60px; white-space:pre-wrap; word-break:break-word; line-height:1.6; }
.t5-out:empty::before { content:"The answer streams in here."; color:var(--muted); }
.t5-verdict { font-family:var(--font-display); font-size:1.5rem; color:var(--accent); }
.fed-input { border:1px dashed var(--border-strong); border-radius:8px; background:var(--bg-secondary);
  padding:.5rem .7rem; font-family:var(--font-mono); font-size:.82rem; white-space:pre-wrap; word-break:break-word; }
.fed-input .prefix { color:var(--accent); font-weight:700; }
.task-picker { display:flex; flex-wrap:wrap; gap:.4rem; margin:.4rem 0 .2rem; }
.task-btn { font:inherit; font-size:.82rem; padding:.35rem .7rem; border-radius:999px; border:1px solid var(--border);
  background:var(--bg-raised); color:var(--color); cursor:pointer; min-block-size:2.2rem; }
.task-btn:hover { border-color:var(--accent); }
.task-btn:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.task-btn[aria-pressed=true] { background:var(--accent); color:var(--accent-ink); border-color:var(--accent); }
.controls-grid { display:grid; gap:.9rem 1.2rem; grid-template-columns:repeat(auto-fit, minmax(160px, 1fr));
  align-items:end; margin:.6rem 0; }
.controls-grid label { display:flex; flex-direction:column; gap:.3rem; font-size:.82rem; }
.controls-grid input[type=range] { inline-size:100%; }
.controls-grid .val { font-family:var(--font-mono); color:var(--muted); font-size:.78rem; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem; color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.stat-row { display:grid; gap:.6rem; grid-template-columns:repeat(auto-fit, minmax(120px, 1fr)); margin-top:.6rem; }
.stat { border:1px solid var(--border); border-radius:8px; background:var(--bg-secondary); padding:.5rem .7rem; }
.stat .k { font-family:var(--font-mono); font-size:.68rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
.stat .v { font-family:var(--font-display); font-size:1.4rem; }
.tok-wrap { display:flex; flex-wrap:wrap; gap:3px; margin-top:.4rem; min-inline-size:0; }
.tok { font-family:var(--font-mono); font-size:.75rem; padding:.1rem .35rem; border-radius:4px;
  border:1px solid var(--border); background:var(--bg-raised); word-break:break-all; }
.tok-special { color:var(--muted); border-style:dashed; }
.timeline { inline-size:100%; block-size:90px; display:block; }
.preset-grid { display:grid; gap:.6rem; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); margin:.6rem 0; }
.preset { text-align:start; border:1px solid var(--border); border-radius:8px; background:var(--bg-raised);
  padding:.55rem .7rem; cursor:pointer; font:inherit; min-inline-size:0; }
.preset:hover { border-color:var(--accent); }
.preset:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.preset .t { font-weight:600; font-size:.9rem; display:block; }
.preset .d { color:var(--muted); font-size:.78rem; font-family:var(--font-mono); }
.chip { font:inherit; font-size:.78rem; padding:.2rem .6rem; border-radius:999px; border:1px solid var(--border);
  background:var(--bg-raised); color:var(--color); cursor:pointer; }
.chip:hover { border-color:var(--accent); }
.t5-note { font-size:.82rem; color:var(--muted); }
`;
