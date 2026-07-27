// Shared engine + rendering helpers for the multilingual-e5-small family pages.

export const E5_CSS = `
.e5-input { width: 100%; box-sizing: border-box; }
.e5-chips { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0; }
.e5-chip { min-height: 32px; }
.e5-sims { display: flex; flex-direction: column; gap: .45rem; margin-top: .6rem; }
.e5-sim-row { display: grid; grid-template-columns: minmax(0,1fr) auto; gap: .35rem .6rem; align-items: center; }
.e5-sim-row .txt { overflow-wrap: anywhere; }
.e5-bar { grid-column: 1 / -1; height: 10px; border-radius: 5px; background: color-mix(in srgb, currentColor 14%, transparent); position: relative; overflow: hidden; }
.e5-bar > span { position: absolute; inset: 0 auto 0 0; border-radius: 5px; background: var(--accent, #4f46e5); min-width: 2px; }
.e5-score { font-variant-numeric: tabular-nums; font-weight: 600; }
.e5-best { outline: 2px solid var(--accent, #4f46e5); outline-offset: 2px; border-radius: 4px; }
.e5-matrix-wrap { overflow-x: auto; max-width: 100%; }
.e5-matrix { border-collapse: collapse; font-size: .78rem; }
.e5-matrix th, .e5-matrix td { padding: .3rem .45rem; border: 1px solid color-mix(in srgb, currentColor 16%, transparent); text-align: center; font-variant-numeric: tabular-nums; }
.e5-matrix th { max-width: 12rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 600; }
.e5-dims { display: grid; grid-template-columns: repeat(24, 1fr); gap: 2px; margin-top: .4rem; }
.e5-dims > span { height: 14px; border-radius: 2px; }
.readout { display: flex; flex-wrap: wrap; gap: .8rem; font-size: .82rem; margin-top: .6rem; }
`;

export function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

export class E5Engine {
  constructor(workerUrl = new URL("./worker.js", import.meta.url)) {
    this.worker = new Worker(workerUrl, { type: "module" });
    this.pending = new Map();
    this.seq = 0;
    this.onProgress = null;
    this.onReady = null;
    this.device = null;
    this.worker.addEventListener("message", (e) => {
      const m = e.data;
      if (m.type === "progress" && this.onProgress) this.onProgress(m.p);
      else if (m.type === "ready") {
        this.device = m.device;
        if (this.onReady) this.onReady(m.device);
      } else if (m.type === "embed" || m.type === "error") {
        const p = this.pending.get(m.id);
        if (!p) return;
        this.pending.delete(m.id);
        if (m.type === "error") p.reject(new Error(m.message));
        else p.resolve(m);
      }
    });
  }

  load(onProgress) {
    this.onProgress = onProgress;
    this.worker.postMessage({ type: "load" });
    return new Promise((resolve) => {
      this.onReady = resolve;
    });
  }

  embed(items) {
    const id = ++this.seq;
    this.worker.postMessage({ type: "embed", id, items });
    return new Promise((resolve, reject) => this.pending.set(id, { resolve, reject }));
  }
}

// Colour a 0..1 cosine similarity as a cell background (white → indigo).
export function simColor(s) {
  const t = Math.max(0, Math.min(1, (s - 0.5) / 0.5)); // 0.5..1.0 is the meaningful band
  return `background: color-mix(in srgb, var(--accent, #4f46e5) ${
    Math.round(t * 70)
  }%, transparent)`;
}

// Render ranked similarity bars: items = [{ text, lang?, sim }] sorted desc; bestIdx highlighted.
export function renderSimList(el, rows, { showKind = true } = {}) {
  el.replaceChildren();
  for (const r of rows) {
    const row = document.createElement("div");
    row.className = "e5-sim-row" + (r.best ? " e5-best" : "");
    const txt = document.createElement("span");
    txt.className = "txt";
    txt.textContent = (showKind && r.kind ? `[${r.kind}] ` : "") + (r.lang ? `${r.lang} — ` : "") +
      r.text;
    const score = document.createElement("span");
    score.className = "e5-score";
    score.textContent = r.sim.toFixed(3);
    const bar = document.createElement("div");
    bar.className = "e5-bar";
    bar.setAttribute("role", "img");
    bar.setAttribute("aria-label", `similarity ${r.sim.toFixed(3)}`);
    const fill = document.createElement("span");
    fill.style.width = `${Math.max(2, Math.round(r.sim * 100))}%`;
    bar.append(fill);
    row.append(txt, score, bar);
    el.append(row);
  }
}

// Render the full cosine matrix as an accessible table.
export function renderMatrix(el, labels, sims) {
  el.replaceChildren();
  const table = document.createElement("table");
  table.className = "e5-matrix";
  const caption = document.createElement("caption");
  caption.textContent =
    "Cosine similarity between every pair of embeddings (1.0 = identical meaning)";
  table.append(caption);
  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  hr.append(document.createElement("th"));
  labels.forEach((l, i) => {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = `${i + 1}`;
    th.title = l;
    hr.append(th);
  });
  thead.append(hr);
  table.append(thead);
  const tbody = document.createElement("tbody");
  labels.forEach((l, i) => {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.scope = "row";
    th.textContent = `${i + 1}`;
    th.title = l;
    tr.append(th);
    sims[i].forEach((s, j) => {
      const td = document.createElement("td");
      td.textContent = s.toFixed(2);
      td.setAttribute("style", simColor(s));
      tr.append(td);
    });
    tbody.append(tr);
  });
  table.append(tbody);
  el.append(table);
}

// Render a compact bar-strip of one embedding vector's dimensions (real values).
export function renderDims(el, vector) {
  el.replaceChildren();
  const max = Math.max(...vector.map((v) => Math.abs(v)), 1e-9);
  const step = Math.max(1, Math.floor(vector.length / 192));
  for (let i = 0; i < vector.length; i += step) {
    const v = vector[i];
    const s = document.createElement("span");
    const mag = Math.abs(v) / max;
    s.setAttribute(
      "style",
      `background: ${v >= 0 ? "var(--accent, #4f46e5)" : "#c2410c"}; opacity: ${
        (0.15 + 0.85 * mag).toFixed(2)
      }`,
    );
    s.title = `dim ${i}: ${v.toFixed(4)}`;
    el.append(s);
  }
}
