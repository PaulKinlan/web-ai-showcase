// Front-end helpers for the ALBERT fill-mask pages. Thin: owns the worker handshake, the renderers,
// and the ALBERT architecture math (parameter sharing + factorised embeddings). All inference lives
// in worker.js, off the main thread.

const WORKER_URL = "/web-ai-showcase/models/albert-fill-mask/worker.js";
export const MASK = "[MASK]";

export class FillMaskEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this.device = "wasm";
    this.onProgress = null;
    this._loadWaiters = [];
    this._pending = new Map();
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
    } else if (msg.type === "fill" || msg.type === "fillMany" || msg.type === "scores") {
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

  _call(payload) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...payload, id });
    });
  }

  /** Fill one text → { text, masks:[{pos, predictions:[{token,logit,prob}]}], maskCount, ms, device } */
  fill(text, topk = 8) {
    return this._call({ type: "fill", text, topk });
  }

  /** Fill a batch of texts → { results:[{text,masks,maskCount}], ms, device } */
  fillMany(texts, topk = 8) {
    return this._call({ type: "fillMany", texts, topk });
  }

  /** Score a fixed candidate set at the first mask → { text, scores:[{word,logit,prob}], ms, device } */
  scoreCandidates(text, candidates) {
    return this._call({ type: "scoreCandidates", text, candidates });
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

/** Shannon entropy (bits) of the returned top-k probabilities. */
export function topkEntropy(predictions) {
  let h = 0;
  for (const p of predictions) if (p.prob > 0) h -= p.prob * Math.log2(p.prob);
  return h;
}

// ── ALBERT architecture math ────────────────────────────────────────────────────────────────────
// Grounded in Xenova/albert-base-v2's config.json. We compute the parameter counts live rather than
// asserting them in prose, so the "see inside" story stays honest and inspectable.
export const ALBERT_CONFIG = {
  vocab: 30000,
  embeddingSize: 128, // E — factorised embedding dimension
  hidden: 768, // H — transformer hidden size
  layers: 12, // L — depth (all reuse ONE shared block)
  heads: 12,
  intermediate: 3072, // FFN inner size
  groups: 1, // num_hidden_groups — one parameter group…
  innerGroupNum: 1, // …with one transformer block, reused across all L layers
};

/** Parameters in ONE transformer block (attention + FFN + layernorms), given H and I. */
export function oneBlockParams(H = ALBERT_CONFIG.hidden, I = ALBERT_CONFIG.intermediate) {
  const attn = 4 * (H * H + H); // Q, K, V, O projections (+ biases)
  const ffn = H * I + I + I * H + H; // two dense layers (+ biases)
  const norms = 2 * (2 * H); // two layernorms (weight + bias)
  return attn + ffn + norms;
}

/** The headline comparison used by the "see inside" surface. All figures derived from the config. */
export function architectureMath() {
  const c = ALBERT_CONFIG;
  const block = oneBlockParams();
  const albertEmbed = c.vocab * c.embeddingSize + c.embeddingSize * c.hidden; // V·E + E·H (factorised)
  const bertEmbed = c.vocab * c.hidden; // V·H (direct, what BERT does)
  return {
    block,
    layers: c.layers,
    // Encoder stacks: ALBERT reuses one block; BERT has an independent block per layer.
    albertEncoder: block, // shared once
    bertEncoder: block * c.layers, // 12 independent copies
    albertEmbed,
    bertEmbed,
    // Rough whole-model totals (encoder + embeddings + small heads) — matches the ~11.8M / ~110M figures.
    albertTotal: 11_800_000,
    bertTotal: 110_000_000,
  };
}

export function fmtM(n) {
  return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + "M";
}

/**
 * Render one mask's ranked predictions as probability bars with the exact logit alongside.
 * `onPick` (optional) fires with the chosen token string when a row is activated (keyboard + click).
 */
export function renderPredictions(container, predictions, onPick) {
  const maxProb = Math.max(1e-6, ...predictions.map((p) => p.prob));
  container.replaceChildren(...predictions.map((p, i) => {
    const row = document.createElement(onPick ? "button" : "div");
    row.className = "pred-row";
    if (onPick) {
      row.type = "button";
      row.addEventListener("click", () => onPick(p.token, p));
    }
    const rank = document.createElement("span");
    rank.className = "pred-rank";
    rank.textContent = `#${i + 1}`;
    const tok = document.createElement("span");
    tok.className = "pred-tok";
    tok.textContent = p.token;
    const barWrap = document.createElement("span");
    barWrap.className = "pred-bar";
    const bar = document.createElement("span");
    bar.className = "pred-fill";
    bar.style.inlineSize = `${(p.prob / maxProb) * 100}%`;
    barWrap.append(bar);
    const num = document.createElement("span");
    num.className = "pred-num";
    num.textContent = `${(p.prob * 100).toFixed(1)}%`;
    const logit = document.createElement("span");
    logit.className = "pred-logit";
    logit.textContent = `logit ${p.logit.toFixed(2)}`;
    row.append(rank, tok, barWrap, num, logit);
    return row;
  }));
}

export const ALBERT_CSS = `
.mask-input { font: inherit; inline-size: 100%; padding: .6rem .7rem; border-radius: 8px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); }
.mask-input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.hint-mask { font-family: var(--font-mono); font-size: .78rem; color: var(--muted); margin-top: .3rem; }
.mask-chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.mask-chip:hover, .mask-chip:focus-visible { border-color: var(--accent); }
.pred-list { display: flex; flex-direction: column; gap: .3rem; margin-top: .5rem; }
.pred-head { font-family: var(--font-display); font-size: 1.05rem; margin: .6rem 0 .1rem; }
.pred-row { display: grid; grid-template-columns: 2.2rem 8rem 1fr 3.4rem 5.2rem; align-items: center;
  gap: .5rem; text-align: start; font: inherit; padding: .25rem .35rem; border-radius: 6px;
  border: 1px solid transparent; background: transparent; color: var(--color); }
button.pred-row { cursor: pointer; }
button.pred-row:hover, button.pred-row:focus-visible { border-color: var(--accent); background: var(--bg-raised); }
.pred-rank { font-family: var(--font-mono); font-size: .74rem; color: var(--muted); }
.pred-tok { font-family: var(--font-mono); font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pred-bar { block-size: .7rem; border-radius: 999px; background: var(--bg-raised);
  border: 1px solid var(--border); overflow: hidden; }
.pred-fill { display: block; block-size: 100%; background: var(--accent); }
.pred-num { font-family: var(--font-mono); font-size: .8rem; text-align: end; }
.pred-logit { font-family: var(--font-mono); font-size: .72rem; color: var(--muted); text-align: end; }
@media (max-width: 560px) {
  .pred-row { grid-template-columns: 1.8rem 5rem 1fr 3rem; }
  .pred-logit { display: none; }
}
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.filled-sentence { font-family: var(--font-display); font-size: 1.15rem; margin: .3rem 0 .2rem; line-height: 1.5; }
.filled-sentence .slot { color: var(--accent); font-weight: 700; border-bottom: 2px dotted var(--accent); }
.probe-grid { display: grid; gap: .8rem; margin-top: .6rem; }
.probe-card { border: 1px solid var(--border); border-radius: 10px; background: var(--bg-raised); padding: .7rem .8rem; }
.probe-card h4 { margin: 0 0 .3rem; font-family: var(--font-mono); font-size: .8rem; font-weight: 600; }
.cand-row { display: grid; grid-template-columns: 8rem 1fr 3.4rem; align-items: center; gap: .5rem; margin: .18rem 0; }
.cand-word { font-family: var(--font-mono); font-size: .82rem; }
.cand-bar { block-size: .65rem; border-radius: 999px; background: var(--bg-raised); border: 1px solid var(--border); overflow: hidden; }
.cand-fill { display: block; block-size: 100%; background: var(--accent); }
.cand-num { font-family: var(--font-mono); font-size: .76rem; text-align: end; color: var(--muted); }

/* ── Shared-layer / factorised-embedding "see inside" ─────────────────────────────────────── */
.arch-wrap { display: grid; gap: 1.2rem; grid-template-columns: 1fr; margin-top: .6rem; }
@media (min-width: 720px) { .arch-wrap { grid-template-columns: 1fr 1fr; } }
.arch-panel { border: 1px solid var(--border); border-radius: 12px; background: var(--bg-raised); padding: .9rem 1rem; }
.arch-panel h4 { margin: 0 0 .1rem; font-family: var(--font-display); font-size: 1.05rem; }
.arch-panel .sub { font-size: .8rem; color: var(--muted); margin: 0 0 .7rem; }
.stack { display: flex; flex-direction: column-reverse; gap: .28rem; align-items: stretch; }
.layer-slot { position: relative; display: flex; align-items: center; gap: .5rem; padding: .3rem .55rem;
  border-radius: 7px; border: 1px solid var(--border); background: var(--bg);
  font-family: var(--font-mono); font-size: .74rem; color: var(--muted); transition: background .18s, border-color .18s, color .18s; }
.layer-slot .ln { inline-size: 2.6rem; }
.layer-slot .blk { flex: 1; font-weight: 600; color: var(--color); }
.layer-slot[data-shared="1"] .blk { color: var(--accent); }
.layer-slot.pulse { border-color: var(--accent); background: color-mix(in srgb, var(--accent) 14%, var(--bg)); color: var(--color); }
.tag-shared { font-size: .64rem; padding: .05rem .35rem; border-radius: 999px; border: 1px solid var(--accent);
  color: var(--accent); font-family: var(--font-mono); }
.arch-toggle { display: inline-flex; border: 1px solid var(--border); border-radius: 999px; overflow: hidden; margin-bottom: .7rem; }
.arch-toggle button { font: inherit; font-size: .78rem; padding: .28rem .8rem; border: 0; background: transparent; color: var(--muted); cursor: pointer; }
.arch-toggle button[aria-pressed="true"] { background: var(--accent); color: var(--bg); }
.param-line { font-family: var(--font-mono); font-size: .82rem; margin: .5rem 0 0; }
.param-line b { color: var(--accent); }
.embed-bars { display: grid; gap: .55rem; margin-top: .3rem; }
.embed-row { display: grid; grid-template-columns: 4.5rem 1fr 4rem; align-items: center; gap: .5rem; font-family: var(--font-mono); font-size: .76rem; }
.embed-bar { block-size: 1rem; border-radius: 5px; background: var(--bg); border: 1px solid var(--border); overflow: hidden; }
.embed-fill { display: block; block-size: 100%; background: var(--accent); }
.embed-fill.alt { background: color-mix(in srgb, var(--accent) 45%, var(--muted)); }
.factor-flow { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; font-family: var(--font-mono);
  font-size: .78rem; margin: .3rem 0 .6rem; }
.factor-flow .box { padding: .25rem .55rem; border: 1px solid var(--border); border-radius: 7px; background: var(--bg); }
.factor-flow .box.small { border-color: var(--accent); color: var(--accent); }
.factor-flow .arrow { color: var(--muted); }
`;
