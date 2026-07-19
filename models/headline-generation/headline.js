// Front-end helpers for the headline / title generation pages. Keeps each page thin: it owns the worker
// handshake, streaming, and the render helpers (best headline, candidate list, input token chips, decode
// timeline). All inference lives in worker.js (off the main thread).
//
// The mechanism this demo teaches: writing a title is GENERATION, not extraction — the same T5 fine-tune
// on the same article produces one "best" beam-search headline and, with sampling turned on, a spread of
// different candidate headlines. The front end never invents text; every string shown comes from a real
// worker run.

const WORKER_URL = "/web-ai-showcase/models/headline-generation/worker.js";

/** Bundled example articles (rights-safe: written for this demo, no external source). */
export const SAMPLES = [
  {
    label: "Browser AI",
    text:
      "The web platform keeps gaining new capabilities. Browsers can now run machine learning models locally, on the user's own device. That means data never leaves the tab, there is no per-request server cost, and the model keeps working offline once it is cached. Developers get privacy and lower latency without standing up any inference infrastructure.",
  },
  {
    label: "Community garden",
    text:
      "A patch of unused land behind the old railway station has been turned into a community garden by a group of neighbours. Over one summer they cleared the rubble, built raised beds from reclaimed timber, and planted vegetables that are now shared freely with anyone who helps tend the plots. The council, which had planned to sell the land, says it is reconsidering after seeing how busy the space has become.",
  },
  {
    label: "Sleep study",
    text:
      "Researchers who tracked the sleep of two thousand office workers for a year found that those who kept a consistent bedtime, even on weekends, reported sharper focus and better mood than colleagues whose schedules drifted. The effect held regardless of how many total hours people slept, suggesting that regularity itself, not just duration, matters for how rested we feel.",
  },
];

export class HeadlineEngine {
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

  /**
   * Generate headlines for an article.
   * opts: { maxNewTokens, numCandidates, temperature, topK, noRepeat }. onStream(partialBest).
   * Resolves → { input, prefix, best, bestTokens, intervals, candidates:[{text,tokens,ms}],
   *             inTokens, inTokenStrings, temperature, topK, bestMs, ms, device }.
   */
  run(article, opts = {}, onStream) {
    const id = ++this._id;
    if (onStream) this._streams.set(id, onStream);
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, article, opts });
    });
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Render the input token strings as chips, exposing T5's SentencePiece ▁ word-boundary marks. */
export function renderTokens(container, tokenStrings) {
  container.replaceChildren(...(tokenStrings || []).map((t) => {
    const chip = document.createElement("span");
    chip.className = "tok";
    chip.textContent = t.replace(/▁/g, "·");
    if (/^<.*>$/.test(t)) chip.classList.add("tok-special");
    return chip;
  }));
}

/** Render sampled candidate headlines as a ranked list with per-candidate token counts. */
export function renderCandidates(container, candidates) {
  if (!candidates || candidates.length === 0) {
    container.replaceChildren();
    return;
  }
  // Deduplicate identical draws but keep a count so repeats are visible, not hidden.
  const seen = new Map();
  for (const c of candidates) {
    const key = c.text.toLowerCase();
    if (seen.has(key)) seen.get(key).n++;
    else seen.set(key, { ...c, n: 1 });
  }
  container.replaceChildren(...[...seen.values()].map((c, i) => {
    const row = document.createElement("div");
    row.className = "cand-row";
    row.innerHTML = `<span class="cand-i">${i + 1}</span>` +
      `<span class="cand-text">${escapeHTML(c.text) || "<em>(empty draw)</em>"}</span>` +
      `<span class="cand-meta">${c.tokens} tok${c.n > 1 ? ` · ×${c.n}` : ""}</span>`;
    return row;
  }));
}

/** Draw the real per-token decode timeline of the streamed "best" headline (inter-token intervals, ms). */
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
    ctx.fillText("Generate a headline to see the decode timeline.", 8, h / 2);
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

export const HEADLINE_CSS = `
.hg-io textarea { inline-size:100%; font-family:var(--font-body); box-sizing:border-box; }
.hg-fields { display:flex; flex-direction:column; gap:.3rem; font-size:.82rem; margin-top:.6rem; }
.hg-best { border:1px solid var(--accent); border-inline-start:4px solid var(--accent); border-radius:var(--radius);
  background:var(--bg-raised); padding:.7rem 1rem; min-block-size:2.4rem; line-height:1.5;
  font-family:var(--font-display); font-size:1.3rem; word-break:break-word; }
.hg-best:empty::before { content:"The best headline streams in here."; color:var(--muted); font-family:var(--font-body); font-size:1rem; }
.cand-list { display:flex; flex-direction:column; gap:.4rem; margin-top:.5rem; }
.cand-row { display:grid; grid-template-columns:auto 1fr auto; gap:.6rem; align-items:baseline;
  border:1px solid var(--border); border-radius:8px; background:var(--bg-raised); padding:.45rem .7rem; }
.cand-i { font-family:var(--font-mono); color:var(--muted); font-size:.78rem; }
.cand-text { font-size:.95rem; min-inline-size:0; word-break:break-word; }
.cand-meta { font-family:var(--font-mono); font-size:.72rem; color:var(--muted); white-space:nowrap; }
.fed-input { border:1px dashed var(--border-strong); border-radius:8px; background:var(--bg-secondary);
  padding:.5rem .7rem; font-family:var(--font-mono); font-size:.82rem; white-space:pre-wrap; word-break:break-word; }
.fed-input .prefix { color:var(--accent); font-weight:700; }
.sample-grid { display:flex; flex-wrap:wrap; gap:.4rem; margin:.4rem 0; }
.sample-btn { font:inherit; font-size:.82rem; padding:.35rem .7rem; border-radius:999px; border:1px solid var(--border);
  background:var(--bg-raised); color:var(--color); cursor:pointer; min-block-size:2.2rem; }
.sample-btn:hover { border-color:var(--accent); }
.sample-btn:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.controls-grid { display:grid; gap:.9rem 1.2rem; grid-template-columns:repeat(auto-fit, minmax(160px, 1fr));
  align-items:end; margin:.6rem 0; }
.controls-grid label { display:flex; flex-direction:column; gap:.3rem; font-size:.82rem; }
.controls-grid input[type=range] { inline-size:100%; accent-color:var(--accent); }
.controls-grid .val { font-family:var(--font-mono); color:var(--muted); font-size:.78rem; }
.mode-toggle { display:flex; flex-wrap:wrap; gap:.4rem; margin:.2rem 0 .4rem; }
.mode-btn { font:inherit; font-size:.82rem; padding:.35rem .8rem; border-radius:999px; border:1px solid var(--border);
  background:var(--bg-raised); color:var(--color); cursor:pointer; min-block-size:2.2rem; }
.mode-btn[aria-pressed=true] { background:var(--accent); color:var(--accent-ink); border-color:var(--accent); }
.mode-btn:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
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
.hg-note { font-size:.82rem; color:var(--muted); }
`;
