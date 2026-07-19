// Shared front-end helpers for the offensive-language moderation pages. Keeps each page thin: it owns the
// worker handshake and the label renderers. All inference (softmax severity scoring, batch triage, gating)
// lives in worker.js and runs off the main thread.

const WORKER_URL = "/web-ai-showcase/models/offensive-language-detection/worker.js";

// The ordered severity ladder. Index === the model's class index.
export const SEVERITY = ["acceptable", "inappropriate", "offensive", "violent"];

export class OffensiveEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.ready = false;
    this._loadWaiters = [];
    this._pending = new Map();
    this._id = 0;
    this.onProgress = null;
    this.device = "wasm";
    this.labels = SEVERITY;
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
      this.labels = msg.labels || SEVERITY;
      for (const w of this._loadWaiters) w.resolve(msg.device);
      this._loadWaiters = [];
    } else if (msg.type === "result" || msg.type === "triage" || msg.type === "gate") {
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

  /** Score one text → { text, labels:[{label,score,logit}], offensiveScore, argmax, ms, device }. */
  classify(text) {
    return this._call({ type: "run", text });
  }

  /** Batch triage many texts → { items:[{text,argmax,offensiveScore,labels,flag}], ms, device }. */
  triage(texts, threshold) {
    return this._call({ type: "triage", texts, threshold });
  }

  /** Moderation gate → { text, clean, offensiveScore, argmax, labels, ms, device }. */
  gate(text, threshold) {
    return this._call({ type: "gate", text, threshold });
  }

  _call(payload) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...payload, id });
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

/** Is this severity class one a moderation queue flags (offensive or violent)? */
export function isFlagClass(label) {
  return label === "offensive" || label === "violent";
}

/**
 * Render the four severity classes as probability bars into `container`. `labels` = [{label,score}]
 * sorted high→low. This is the SOFTMAX view: the four scores sum to 1 (one distribution over the
 * severity ladder), so the bars compete — unlike toxic-bert's independent sigmoids.
 */
export function renderSeverityBars(container, labels) {
  container.replaceChildren(
    ...labels.map((l) => {
      const pct = (l.score * 100).toFixed(1);
      const flagged = isFlagClass(l.label);
      const row = document.createElement("div");
      row.className = "bar-row" + (flagged ? " bar-flag" : " bar-ok");
      row.innerHTML = `
        <div class="bar-head">
          <span class="bar-label">${escapeHTML(l.label)}</span>
          <span class="bar-val">${pct}%</span>
        </div>
        <div class="bar-track" role="meter" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}"
             aria-label="${escapeHTML(l.label)}: ${pct} percent">
          <div class="bar-fill" style="inline-size:${pct}%"></div>
        </div>`;
      return row;
    }),
  );
}

/** Overall verdict from the aggregate offensive-or-worse probability vs a threshold. */
export function verdict(offensiveScore, argmax, threshold) {
  return { flag: offensiveScore >= threshold, offensiveScore, argmax };
}

export const OFFENSIVE_CSS = `
.verdict-row { display: flex; align-items: baseline; gap: .8rem; flex-wrap: wrap; margin-top: .6rem; }
.verdict-label { font-family: var(--font-display); font-size: 1.7rem; }
.verdict-label.flag { color: var(--bad); }
.verdict-label.allow { color: var(--good); }
.verdict-sub { font-family: var(--font-mono); color: var(--muted); font-size: .85rem; }
.bars { display: flex; flex-direction: column; gap: .5rem; margin-top: .6rem; max-inline-size: 560px; }
.bar-head { display: flex; justify-content: space-between; gap: .5rem; font-size: .85rem; }
.bar-label { font-family: var(--font-body); text-transform: capitalize; }
.bar-val { font-family: var(--font-mono); color: var(--muted); white-space: nowrap; }
.bar-track { block-size: .7rem; background: var(--bg-raised); border: 1px solid var(--border);
  border-radius: 999px; overflow: hidden; margin-top: .15rem; }
.bar-fill { block-size: 100%; background: var(--muted); border-radius: 999px; transition: inline-size .3s ease; }
.bar-flag .bar-fill { background: var(--bad); }
.bar-flag .bar-label { font-weight: 600; color: var(--bad); }
.bar-ok .bar-fill { background: var(--good); }
.thresh-wrap { display: flex; flex-direction: column; gap: .25rem; font-size: .82rem; max-inline-size: 560px; margin-top: .4rem; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono);
  font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.chip { font: inherit; font-size: .78rem; padding: .3rem .7rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer;
  min-block-size: 34px; }
.chip:hover { border-color: var(--accent); }
.inside-table { inline-size: 100%; border-collapse: collapse; font-size: .82rem; margin-top: .5rem; }
.inside-table th, .inside-table td { text-align: left; padding: .3rem .5rem;
  border-bottom: 1px solid var(--border); font-family: var(--font-mono); }
.inside-table th { color: var(--muted); font-weight: 600; }
.queue-row { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-raised);
  padding: .5rem .7rem; display: flex; justify-content: space-between; gap: .6rem; align-items: center; }
.queue-row.flag { border-inline-start: 4px solid var(--bad); }
.queue-row.allow { border-inline-start: 4px solid var(--good); }
.queue-text { flex: 1 1 auto; overflow-wrap: anywhere; }
.queue-meta { font-family: var(--font-mono); font-size: .74rem; color: var(--muted); white-space: nowrap; text-align: right; }
.badge { font-family: var(--font-mono); font-size: .68rem; padding: .1rem .45rem; border-radius: 999px;
  border: 1px solid var(--border); }
.badge.flag { color: var(--bad); border-color: var(--bad); }
.badge.allow { color: var(--good); border-color: var(--good); }
.queue-list { display: flex; flex-direction: column; gap: .5rem; margin-top: .6rem; }
`;
