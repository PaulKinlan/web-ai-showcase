// Front-end helpers for the code-embeddings pages. Keeps each page thin: the worker handshake, the
// light client-side maths (cosine, similarity matrix), a small shared corpus of code snippets across
// languages, and the renderers for ranked search + heat-matrix. All inference lives in the worker
// (off the main thread); the maths here runs on a handful of already-computed 768-d vectors so it is
// cheap enough for the main thread.

/** Client for the code-embeddings worker (same message protocol as the other embedding workers). */
export class CodeEmbedClient {
  constructor(workerUrl = "/web-ai-showcase/models/code-embeddings/worker.js") {
    this.worker = new Worker(workerUrl, { type: "module" });
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

  /** Embed texts → { texts, embeddings:number[][] (unit vectors), norms, dim, ms, device } */
  embed(texts) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, texts });
    });
  }
}

/** Cosine similarity. Worker vectors are unit-length, so this is a dot product. */
export function cosine(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

/** Full pairwise similarity matrix for a set of embeddings. */
export function simMatrix(embeddings) {
  const n = embeddings.length;
  const m = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const s = i === j ? 1 : cosine(embeddings[i], embeddings[j]);
      m[i][j] = s;
      m[j][i] = s;
    }
  }
  return m;
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** Trim a snippet's shared leading indentation and cap the visible line count for compact display. */
export function prettySnippet(code, maxLines = 8) {
  const lines = code.replace(/\t/g, "  ").split("\n");
  while (lines.length && !lines[0].trim()) lines.shift();
  while (lines.length && !lines.at(-1).trim()) lines.pop();
  const indent = Math.min(
    ...lines.filter((l) => l.trim()).map((l) => l.match(/^ */)[0].length),
  );
  let out = lines.map((l) => l.slice(indent));
  if (out.length > maxLines) out = [...out.slice(0, maxLines), "…"];
  return out.join("\n");
}

// A shared corpus of real snippets across languages. `lang` drives the language tag; `intent` groups
// snippets that DO the same thing in different languages (used by the cross-language clone demo).
export const CORPUS = [
  {
    lang: "Python",
    intent: "reverse-string",
    code: `def reverse_string(s):\n    return s[::-1]`,
  },
  {
    lang: "JavaScript",
    intent: "reverse-string",
    code: `function reverseString(s) {\n  return s.split("").reverse().join("");\n}`,
  },
  {
    lang: "Go",
    intent: "reverse-string",
    code:
      `func Reverse(s string) string {\n    r := []rune(s)\n    for i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 {\n        r[i], r[j] = r[j], r[i]\n    }\n    return string(r)\n}`,
  },
  {
    lang: "Python",
    intent: "fib",
    code:
      `def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a`,
  },
  {
    lang: "Rust",
    intent: "fib",
    code:
      `fn fib(n: u32) -> u64 {\n    let (mut a, mut b) = (0u64, 1u64);\n    for _ in 0..n {\n        let t = a + b; a = b; b = t;\n    }\n    a\n}`,
  },
  {
    lang: "SQL",
    intent: "top-customers",
    code:
      `SELECT customer_id, SUM(total) AS spend\nFROM orders\nGROUP BY customer_id\nORDER BY spend DESC\nLIMIT 10;`,
  },
  {
    lang: "Python",
    intent: "read-json",
    code: `import json\nwith open("data.json") as f:\n    data = json.load(f)`,
  },
  {
    lang: "JavaScript",
    intent: "fetch-json",
    code: `async function getJSON(url) {\n  const res = await fetch(url);\n  return res.json();\n}`,
  },
  {
    lang: "Python",
    intent: "http-get",
    code: `import requests\nresp = requests.get(url)\ndata = resp.json()`,
  },
  {
    lang: "Bash",
    intent: "find-large",
    code: `find . -type f -size +100M -exec ls -lh {} \\;`,
  },
];

/** Render an n×n similarity heatmap into `container`. Labels are short S1…Sn; title carries detail. */
export function renderMatrix(container, matrix, labels) {
  const n = matrix.length;
  const short = labels.map((_, i) => `S${i + 1}`);
  const table = document.createElement("table");
  table.className = "sim-matrix";
  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  hr.appendChild(document.createElement("th"));
  for (let j = 0; j < n; j++) {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = short[j];
    th.title = labels[j];
    hr.appendChild(th);
  }
  thead.appendChild(hr);
  table.appendChild(thead);
  const tb = document.createElement("tbody");
  for (let i = 0; i < n; i++) {
    const tr = document.createElement("tr");
    const rh = document.createElement("th");
    rh.scope = "row";
    rh.textContent = short[i];
    rh.title = labels[i];
    tr.appendChild(rh);
    for (let j = 0; j < n; j++) {
      const td = document.createElement("td");
      const s = matrix[i][j];
      td.textContent = s.toFixed(2);
      td.style.background = simColor(s);
      td.style.color = s > 0.6 ? "var(--accent-ink)" : "var(--color)";
      td.title = `${labels[i]} ↔ ${labels[j]}: cosine ${s.toFixed(4)}`;
      if (i === j) td.classList.add("diag");
      tr.appendChild(td);
    }
    tb.appendChild(tr);
  }
  table.appendChild(tb);
  container.replaceChildren(table);
}

/** Map a cosine value in [-1,1] to accent opacity over a neutral track for a heatmap cell. */
export function simColor(sim) {
  const t = Math.max(0, Math.min(1, sim));
  return `color-mix(in srgb, var(--accent) ${(t * 100).toFixed(0)}%, transparent)`;
}

/**
 * Render a ranked list of code hits into `container`. Each hit = {code, lang, score, intent?}.
 * Shows a similarity bar, a language tag, and the (pretty-printed) snippet.
 */
export function renderCodeHits(container, hits, { showBar = true } = {}) {
  container.replaceChildren(
    ...hits.map((h, rank) => {
      const row = document.createElement("div");
      row.className = "code-hit";
      const head = document.createElement("div");
      head.className = "code-hit-head";
      const left = document.createElement("span");
      left.className = "code-hit-title";
      const langTag = document.createElement("span");
      langTag.className = "lang-tag";
      langTag.textContent = h.lang || "code";
      left.append(document.createTextNode(`${rank + 1}. `), langTag);
      const sc = document.createElement("span");
      sc.className = "code-hit-score";
      sc.textContent = h.score.toFixed(3);
      head.append(left, sc);
      row.append(head);
      if (showBar) {
        const bar = document.createElement("div");
        bar.className = "code-bar";
        const fill = document.createElement("div");
        fill.className = "code-fill";
        fill.style.inlineSize = `${Math.max(0, Math.min(1, h.score)) * 100}%`;
        bar.append(fill);
        row.append(bar);
      }
      const pre = document.createElement("pre");
      pre.className = "code-snip";
      pre.textContent = prettySnippet(h.code);
      row.append(pre);
      return row;
    }),
  );
}

/** Shared inline styles for the code-embedding widgets. Injected once per page. */
export const CODE_CSS = `
.sim-matrix { border-collapse: collapse; font-family: var(--font-mono); font-size: .8rem; margin-top: .5rem; }
.sim-matrix th, .sim-matrix td {
  border: 1px solid var(--border); padding: .3rem .45rem; text-align: center; min-inline-size: 3rem;
}
.sim-matrix thead th, .sim-matrix tbody th { color: var(--muted); background: var(--bg-raised); font-weight: 600; }
.sim-matrix td.diag { outline: 2px solid var(--border-strong); outline-offset: -2px; }
.matrix-scroll { overflow-x: auto; }
.code-list { display: flex; flex-direction: column; gap: .6rem; margin-top: .6rem; }
.code-hit { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised); padding: .55rem .7rem; }
.code-hit-head { display: flex; justify-content: space-between; gap: .5rem; align-items: baseline; }
.code-hit-title { display: flex; align-items: center; gap: .4rem; min-inline-size: 0; }
.code-hit-score { font-family: var(--font-mono); color: var(--muted); font-size: .8rem; white-space: nowrap; }
.lang-tag { display: inline-block; font-family: var(--font-mono); font-size: .68rem; padding: .05rem .4rem;
  border-radius: 999px; border: 1px solid var(--border); color: var(--muted); }
.code-bar { block-size: .4rem; background: var(--bg-secondary); border: 1px solid var(--border);
  border-radius: 999px; overflow: hidden; margin: .35rem 0; }
.code-fill { block-size: 100%; background: var(--accent); }
.code-snip { font-family: var(--font-mono); font-size: .76rem; margin: .3rem 0 0; padding: .5rem .6rem;
  background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 6px; overflow-x: auto;
  white-space: pre; }
.vec-strip { display: flex; flex-wrap: wrap; gap: 2px; margin-top: .4rem; }
.vec-cell { inline-size: 12px; block-size: 20px; border-radius: 2px; border: 1px solid var(--border); }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; align-items: start; }
.field { display: flex; flex-direction: column; gap: .3rem; font-size: .82rem; min-inline-size: 0; }
.field textarea, .field input { font-family: var(--font-mono); min-inline-size: 0; }
.chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px; border: 1px solid var(--border);
  background: var(--bg-raised); color: var(--color); cursor: pointer; }
.chip:hover { border-color: var(--accent); }
.snippet-toggle { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0; }
`;
