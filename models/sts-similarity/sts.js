// Shared front-end helpers for the STS-B cross-encoder similarity pages. Keeps each page thin: it owns
// the worker handshake and the renderers (the 0–5 similarity gauge, the raw-output "see inside" panel,
// and the batched pair table). All inference lives in worker.js (off the main thread).

const WORKER_URL = "/web-ai-showcase/models/sts-similarity/worker.js";

export class STSEngine {
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

  /** Score one pair → { results:[{a,b,logit,sim,score5}], ms, device }. */
  score(a, b) {
    return this.scoreBatch([[a, b]]);
  }
  /** Score many pairs at once → { results:[{a,b,logit,sim,score5}], ms, device }. */
  scoreBatch(pairs) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, pairs });
    });
  }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

export function parseLines(text) {
  return text.split(/\n/).map((s) => s.trim()).filter(Boolean);
}

// Human-readable band for a 0–5 STS score (the STS-B rubric, paraphrased).
export function band(score5) {
  if (score5 >= 4.5) return { label: "Essentially identical", cls: "b5" };
  if (score5 >= 3.5) return { label: "Roughly equivalent (paraphrase)", cls: "b4" };
  if (score5 >= 2.5) return { label: "Related, some shared details", cls: "b3" };
  if (score5 >= 1.5) return { label: "Same topic, different point", cls: "b2" };
  if (score5 >= 0.5) return { label: "Barely related", cls: "b1" };
  return { label: "Unrelated", cls: "b0" };
}

/** Render the 0–5 similarity gauge. Accessible: role="meter" with aria-valuemin/max/now/text. */
export function renderGauge(container, score5) {
  const pct = Math.max(0, Math.min(100, (score5 / 5) * 100));
  const b = band(score5);
  container.innerHTML = `
    <div class="gauge" role="meter" aria-valuemin="0" aria-valuemax="5" aria-valuenow="${
    score5.toFixed(2)
  }" aria-label="Semantic similarity, 0 to 5" aria-valuetext="${score5.toFixed(2)} of 5 — ${
    escapeHTML(b.label)
  }">
      <div class="gauge-track">
        <div class="gauge-fill ${b.cls}" style="inline-size:${pct.toFixed(1)}%"></div>
        <div class="gauge-ticks"><span>0</span><span>1</span><span>2</span><span>3</span><span>4</span><span>5</span></div>
      </div>
      <div class="gauge-readout"><span class="gauge-num">${
    score5.toFixed(2)
  }</span><span class="gauge-max">/ 5</span><span class="gauge-band ${b.cls}">${
    escapeHTML(b.label)
  }</span></div>
    </div>`;
}

/** Render the "see inside" raw-output chain: logit → sigmoid → ×5. */
export function renderInside(container, r) {
  container.innerHTML = `
    <div class="inside-chain">
      <div class="chain-step"><span class="chain-label">raw regression logit</span><span class="chain-val">${
    r.logit.toFixed(3)
  }</span><span class="chain-note">the single number the cross-encoder emits</span></div>
      <div class="chain-arrow" aria-hidden="true">→ sigmoid →</div>
      <div class="chain-step"><span class="chain-label">calibrated similarity</span><span class="chain-val">${
    r.sim.toFixed(3)
  }</span><span class="chain-note">squashed to 0–1</span></div>
      <div class="chain-arrow" aria-hidden="true">→ × 5 →</div>
      <div class="chain-step"><span class="chain-label">STS-B human scale</span><span class="chain-val">${
    r.score5.toFixed(2)
  }</span><span class="chain-note">0 = unrelated · 5 = identical</span></div>
    </div>`;
}

/** Render a table of scored pairs, sorted by similarity (for dedup / batch views). */
export function renderPairTable(container, results, { threshold = null } = {}) {
  const sorted = [...results].sort((x, y) => y.score5 - x.score5);
  container.replaceChildren(...sorted.map((r) => {
    const b = band(r.score5);
    const row = document.createElement("div");
    const dup = threshold != null && r.score5 >= threshold;
    row.className = "pair-row" + (dup ? " dup" : "");
    const pct = (r.score5 / 5) * 100;
    row.innerHTML =
      `<div class="pair-texts"><span dir="auto">${
        escapeHTML(r.a)
      }</span><span class="pair-vs">↔</span><span dir="auto">${escapeHTML(r.b)}</span></div>` +
      `<div class="pair-bar"><span class="pair-fill ${b.cls}" style="inline-size:${
        pct.toFixed(1)
      }%"></span></div>` +
      `<div class="pair-meta"><b>${r.score5.toFixed(2)}</b> / 5 · ${escapeHTML(b.label)}${
        dup ? ' · <span class="dup-tag">≥ threshold → duplicate</span>' : ""
      }</div>`;
    return row;
  }));
}

export const STS_CSS = `
.gauge { margin-top: .4rem; }
.gauge-track { position: relative; block-size: 1.5rem; background: var(--bg-secondary);
  border: 1px solid var(--border); border-radius: 999px; overflow: hidden; }
.gauge-fill { block-size: 100%; border-radius: 999px 0 0 999px; transition: inline-size .35s ease; }
.gauge-fill.b5, .gauge-fill.b4 { background: var(--good); }
.gauge-fill.b3, .gauge-fill.b2 { background: var(--accent); }
.gauge-fill.b1, .gauge-fill.b0 { background: var(--warn); }
.gauge-ticks { position: absolute; inset: 0; display: flex; justify-content: space-between;
  pointer-events: none; padding: 0 .35rem; font-family: var(--font-mono); font-size: .6rem;
  color: var(--muted); align-items: center; }
.gauge-readout { display: flex; align-items: baseline; gap: .5rem; margin-top: .5rem; flex-wrap: wrap; }
.gauge-num { font-family: var(--font-display); font-size: 2rem; line-height: 1; }
.gauge-max { color: var(--muted); font-family: var(--font-mono); font-size: .9rem; }
.gauge-band { font-size: .85rem; padding: .15rem .55rem; border-radius: 999px; border: 1px solid var(--border);
  background: var(--bg-raised); }
.gauge-band.b5, .gauge-band.b4 { color: var(--good); border-color: color-mix(in srgb, var(--good) 45%, var(--border)); }
.gauge-band.b1, .gauge-band.b0 { color: var(--warn); border-color: color-mix(in srgb, var(--warn) 45%, var(--border)); }
.inside-chain { display: flex; flex-wrap: wrap; align-items: stretch; gap: .5rem; margin-top: .5rem; }
.chain-step { flex: 1 1 150px; min-inline-size: 0; border: 1px solid var(--border); border-radius: 8px;
  background: var(--bg-raised); padding: .5rem .6rem; display: flex; flex-direction: column; gap: .15rem; }
.chain-label { font-family: var(--font-mono); font-size: .68rem; color: var(--muted); }
.chain-val { font-family: var(--font-mono); font-size: 1.3rem; color: var(--color); }
.chain-note { font-size: .72rem; color: var(--muted); }
.chain-arrow { display: flex; align-items: center; font-family: var(--font-mono); font-size: .72rem;
  color: var(--muted); }
.pair-list { display: flex; flex-direction: column; gap: .5rem; margin-top: .5rem; }
.pair-row { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised); padding: .5rem .7rem; }
.pair-row.dup { border-color: var(--accent); border-inline-start: 4px solid var(--accent); }
.pair-texts { display: flex; flex-wrap: wrap; gap: .3rem .5rem; font-size: .9rem; align-items: baseline; }
.pair-vs { color: var(--muted); }
.pair-bar { block-size: .45rem; background: var(--bg-secondary); border: 1px solid var(--border);
  border-radius: 999px; overflow: hidden; margin: .35rem 0 .25rem; }
.pair-fill { display: block; block-size: 100%; }
.pair-fill.b5, .pair-fill.b4 { background: var(--good); }
.pair-fill.b3, .pair-fill.b2 { background: var(--accent); }
.pair-fill.b1, .pair-fill.b0 { background: var(--warn); }
.pair-meta { font-family: var(--font-mono); font-size: .74rem; color: var(--muted); }
.pair-meta b { color: var(--color); }
.dup-tag { color: var(--accent); }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.chip { font: inherit; font-size: .78rem; padding: .3rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer;
  min-block-size: 2.2rem; }
.chip:hover { border-color: var(--accent); }
.grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; }
.sts-inputs { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: .75rem; }
.sts-inputs textarea { inline-size: 100%; box-sizing: border-box; }
`;
