// Front-end helpers for the text-simplification pages. Keeps each page thin: it owns the worker
// handshake, streaming, a word-level diff, client-side READABILITY metrics (the before/after contrast
// that makes simplification legible), and the token / timeline render helpers. All inference lives in
// worker.js (off the main thread).

const WORKER_URL = "/web-ai-showcase/models/text-simplification/worker.js";

export class SimplifyEngine {
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

  /** Simplify one input. opts: { maxNewTokens }. onStream(partial). */
  simplify(input, opts = {}, onStream) {
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

// ── Readability metrics (the before/after contrast) ──

/** Rough English syllable count for one word — vowel-group heuristic with a few corrections. */
function syllables(word) {
  const w = String(word).toLowerCase().replace(/[^a-z]/g, "");
  if (!w) return 0;
  if (w.length <= 3) return 1;
  let s =
    (w.replace(/(?:[^laeiouy]es|ed|[^laeiouy]e)$/, "").replace(/^y/, "").match(/[aeiouy]{1,2}/g) ||
      []).length;
  return Math.max(1, s);
}

function words(s) {
  return String(s).trim().split(/\s+/).filter(Boolean);
}

function sentences(s) {
  return String(s).split(/[.!?]+/).map((x) => x.trim()).filter(Boolean);
}

/**
 * Readability of a passage. Returns { words, sentences, syllables, avgWordLen, avgSentLen,
 * complexWords, fleschEase, fkGrade }. Flesch Reading Ease (higher = easier) and Flesch–Kincaid grade
 * level are the standard plain-language yardsticks.
 */
export function readability(text) {
  const ws = words(text);
  const nW = ws.length || 1;
  const nS = Math.max(1, sentences(text).length);
  let syl = 0, complex = 0, chars = 0;
  for (const w of ws) {
    const c = syllables(w);
    syl += c;
    if (c >= 3) complex++;
    chars += w.replace(/[^a-zA-Z]/g, "").length;
  }
  const wps = nW / nS; // words per sentence
  const spw = syl / nW; // syllables per word
  const fleschEase = 206.835 - 1.015 * wps - 84.6 * spw;
  const fkGrade = 0.39 * wps + 11.8 * spw - 15.59;
  return {
    words: ws.length,
    sentences: nS,
    syllables: syl,
    avgWordLen: chars / nW,
    avgSentLen: wps,
    complexWords: complex,
    fleschEase: Math.max(0, Math.min(120, fleschEase)),
    fkGrade: Math.max(0, fkGrade),
  };
}

/** Map a Flesch Reading Ease score to a short human band label. */
export function easeBand(ease) {
  if (ease >= 90) return "very easy";
  if (ease >= 70) return "easy";
  if (ease >= 60) return "plain";
  if (ease >= 50) return "fairly hard";
  if (ease >= 30) return "hard";
  return "very hard";
}

// ── Word-level diff (so the change is legible, not just "here's a new sentence") ──

/** Longest-common-subsequence diff over word arrays → ops [{type:'equal'|'del'|'ins', text}]. */
export function diffWords(aStr, bStr) {
  const a = words(aStr), b = words(bStr);
  const n = a.length, m = b.length;
  const dp = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const ops = [];
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      ops.push({ type: "equal", text: a[i] });
      i++;
      j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      ops.push({ type: "del", text: a[i++] });
    } else {
      ops.push({ type: "ins", text: b[j++] });
    }
  }
  while (i < n) ops.push({ type: "del", text: a[i++] });
  while (j < m) ops.push({ type: "ins", text: b[j++] });
  return ops;
}

export function diffStats(ops) {
  let del = 0, ins = 0, eq = 0;
  for (const o of ops) {
    if (o.type === "del") del++;
    else if (o.type === "ins") ins++;
    else eq++;
  }
  return { del, ins, eq, changes: del + ins };
}

/** Render a unified inline diff into `container`: deletions struck-through, insertions highlighted. */
export function renderDiff(container, ops) {
  container.replaceChildren(...ops.map((o) => {
    const span = document.createElement("span");
    span.className = "diff-" + o.type;
    span.textContent = o.text;
    if (o.type === "del") span.setAttribute("aria-label", "removed: " + o.text);
    if (o.type === "ins") span.setAttribute("aria-label", "added: " + o.text);
    const wrap = document.createElement("span");
    wrap.append(span, document.createTextNode(" "));
    return wrap;
  }));
}

/** Render a before→after readability comparison table. */
export function renderReadability(container, before, after) {
  const rows = [
    [
      "Flesch reading ease",
      before.fleschEase.toFixed(0),
      after.fleschEase.toFixed(0),
      "up",
      `${easeBand(before.fleschEase)} → ${easeBand(after.fleschEase)}`,
    ],
    [
      "Grade level (F–K)",
      before.fkGrade.toFixed(1),
      after.fkGrade.toFixed(1),
      "down",
      "lower is simpler",
    ],
    ["Words", String(before.words), String(after.words), "flat", ""],
    ["Avg words / sentence", before.avgSentLen.toFixed(1), after.avgSentLen.toFixed(1), "down", ""],
    [
      "Avg syllables / word",
      (before.syllables / (before.words || 1)).toFixed(2),
      (after.syllables / (after.words || 1)).toFixed(2),
      "down",
      "",
    ],
    [
      "Complex words (3+ syll.)",
      String(before.complexWords),
      String(after.complexWords),
      "down",
      "",
    ],
  ];
  const table = document.createElement("table");
  table.className = "read-table";
  table.innerHTML =
    `<thead><tr><th scope="col">Metric</th><th scope="col">Before</th><th scope="col">After</th><th scope="col">Δ</th></tr></thead>`;
  const tb = document.createElement("tbody");
  for (const [name, b, a, dir, note] of rows) {
    const bv = parseFloat(b), av = parseFloat(a);
    let good = null;
    if (dir === "up") good = av > bv;
    else if (dir === "down") good = av < bv;
    const tr = document.createElement("tr");
    const delta = av - bv;
    const arrow = delta === 0 ? "→" : delta > 0 ? "▲" : "▼";
    tr.innerHTML =
      `<th scope="row">${name}${note ? ` <span class="muted rt-note">${note}</span>` : ""}</th>` +
      `<td class="num">${b}</td><td class="num">${a}</td>` +
      `<td class="num ${good === true ? "good" : good === false ? "bad" : ""}">${arrow} ${
        Math.abs(delta).toFixed(
          dir === "up" || name.includes("Grade") || name.includes("Avg") ? 1 : 0,
        )
      }</td>`;
    tb.append(tr);
  }
  table.append(tb);
  container.replaceChildren(table);
}

/** Render the input token strings as chips (T5 SentencePiece ▁ word marks shown as ·). */
export function renderTokens(container, tokenStrings) {
  container.replaceChildren(...(tokenStrings || []).map((t) => {
    const chip = document.createElement("span");
    chip.className = "tok";
    chip.textContent = t.replace(/▁/g, "·");
    if (/^(<|>)/.test(t) || /^<.*>$/.test(t)) chip.classList.add("tok-special");
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
    ctx.fillText("No per-token timeline yet.", 8, h / 2);
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

export const SIMPLIFY_CSS = `
.simp-out { border:1px solid var(--border); border-radius:var(--radius); background:var(--bg-raised);
  padding:.8rem 1rem; min-block-size:56px; white-space:pre-wrap; }
.simp-out:empty::before { content:"The simplified sentence will stream in here."; color:var(--muted); }
.diff-box { border:1px solid var(--border); border-radius:var(--radius); background:var(--bg-raised);
  padding:.8rem 1rem; line-height:1.7; }
.diff-equal { color:var(--color); }
.diff-del { color:var(--bad); text-decoration:line-through; text-decoration-thickness:1px; }
.diff-ins { color:var(--good); background:color-mix(in srgb, var(--good) 16%, transparent);
  border-radius:4px; padding:.02rem .18rem; font-weight:600; }
.diff-legend { display:flex; flex-wrap:wrap; gap:1rem; font-size:.76rem; color:var(--muted);
  font-family:var(--font-mono); margin-top:.5rem; }
.diff-legend .k { display:inline-flex; align-items:center; gap:.35rem; }
.diff-legend .sw { inline-size:.9rem; block-size:.9rem; border-radius:3px; display:inline-block; }
.read-table { border-collapse:collapse; inline-size:100%; font-size:.84rem; margin-top:.5rem; }
.read-table th, .read-table td { border:1px solid var(--border); padding:.4rem .6rem; text-align:start; }
.read-table thead th { background:var(--bg-raised); color:var(--muted); font-family:var(--font-mono);
  font-size:.72rem; text-transform:uppercase; letter-spacing:.04em; }
.read-table td.num, .read-table th[scope=col]:not(:first-child) { text-align:end; font-family:var(--font-mono); }
.read-table td.good { color:var(--good); font-weight:600; }
.read-table td.bad { color:var(--bad); font-weight:600; }
.read-table .rt-note { font-size:.7rem; font-weight:400; }
.ease-meters { display:grid; gap:1rem; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); margin-top:.6rem; }
.ease-meter { border:1px solid var(--border); border-radius:8px; background:var(--bg-secondary); padding:.6rem .8rem; }
.ease-meter .lbl { font-family:var(--font-mono); font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
.ease-meter .big { font-family:var(--font-display); font-size:1.8rem; line-height:1.1; }
.ease-meter .band { font-size:.8rem; color:var(--muted); }
.ease-meter .track { block-size:.5rem; background:var(--bg-raised); border:1px solid var(--border);
  border-radius:999px; overflow:hidden; margin-top:.4rem; }
.ease-meter .fill { block-size:100%; background:var(--accent); }
.fed-input { border:1px dashed var(--border-strong); border-radius:8px; background:var(--bg-secondary);
  padding:.5rem .7rem; font-family:var(--font-mono); font-size:.82rem; white-space:pre-wrap; word-break:break-word; }
.controls-grid { display:grid; gap:.9rem 1.2rem; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr));
  align-items:end; margin:.6rem 0; }
.controls-grid label { display:flex; flex-direction:column; gap:.3rem; font-size:.82rem; }
.controls-grid input[type=range] { inline-size:100%; }
.readout { display:flex; flex-wrap:wrap; gap:1rem; font-family:var(--font-mono); font-size:.78rem; color:var(--muted); margin-top:.6rem; }
.readout b { color:var(--color); font-weight:600; }
.stat-row { display:grid; gap:.6rem; grid-template-columns:repeat(auto-fit, minmax(120px, 1fr)); margin-top:.6rem; }
.stat { border:1px solid var(--border); border-radius:8px; background:var(--bg-secondary); padding:.5rem .7rem; }
.stat .k { font-family:var(--font-mono); font-size:.68rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
.stat .v { font-family:var(--font-display); font-size:1.4rem; }
.tok-wrap { display:flex; flex-wrap:wrap; gap:3px; margin-top:.4rem; }
.tok { font-family:var(--font-mono); font-size:.75rem; padding:.1rem .35rem; border-radius:4px;
  border:1px solid var(--border); background:var(--bg-raised); }
.tok-special { color:var(--muted); border-style:dashed; }
.timeline { inline-size:100%; block-size:90px; display:block; }
.preset-grid { display:grid; gap:.6rem; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); margin:.6rem 0; }
.preset { text-align:start; border:1px solid var(--border); border-radius:8px; background:var(--bg-raised);
  padding:.55rem .7rem; cursor:pointer; font:inherit; }
.preset:hover { border-color:var(--accent); }
.preset .t { font-weight:600; font-size:.9rem; } .preset .d { color:var(--muted); font-size:.78rem; font-family:var(--font-mono); }
.grid-2 { display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:1rem; align-items:start; }
.chip { font:inherit; font-size:.78rem; padding:.2rem .6rem; border-radius:999px; border:1px solid var(--border);
  background:var(--bg-raised); color:var(--color); cursor:pointer; }
.chip:hover { border-color:var(--accent); }
`;
