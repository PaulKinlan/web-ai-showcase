// Front-end helpers for the NLI (textual entailment) page: the worker handshake + render/CSS helpers.
// All inference lives in worker.js (off the main thread).

export class NliEngine {
  constructor() {
    this.worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
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
        for (const w of this._loadWaiters) w.reject(new Error(msg.message));
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
  /** Classify a premise/hypothesis pair → { scores:[{label,prob}], top, ms }. */
  infer(premise, hypothesis) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, premise, hypothesis });
    });
  }
}

export function escapeHTML(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]),
  );
}

const LABEL_META = {
  entailment: { color: "#2bb59a", verdict: "The hypothesis follows from the premise" },
  contradiction: { color: "#e05b5b", verdict: "The hypothesis contradicts the premise" },
  neutral: { color: "#c79a3a", verdict: "Neither — the premise doesn't settle it" },
};
export function labelColor(label) {
  return LABEL_META[label]?.color ?? "#888";
}
export function labelVerdict(label) {
  return LABEL_META[label]?.verdict ?? "";
}

/** Render the three-way relationship as labelled probability bars. */
export function renderScores(container, scores) {
  container.innerHTML = scores.map((s) => {
    const pct = Math.round(s.prob * 100);
    return `<div class="nli-bar"><span class="nli-name" style="color:${labelColor(s.label)}">${
      escapeHTML(s.label)
    }</span>` +
      `<span class="nli-track"><i style="width:${Math.max(2, pct)}%;background:${
        labelColor(s.label)
      }"></i></span>` +
      `<span class="nli-pct">${pct}%</span></div>`;
  }).join("");
}

export const NLI_CSS = `
  .nli-field { margin: 0.5rem 0; }
  .nli-field label { display: block; font-size: 0.85rem; margin-bottom: 0.2rem; color: var(--muted, #888); }
  .nli-field input { width: 100%; box-sizing: border-box; padding: 0.55rem 0.65rem; border-radius: 8px; border: 1px solid #8886; font: inherit; font-size: 0.98rem; }
  .nli-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.5rem 0; }
  .nli-verdict { font-size: 1.1rem; font-weight: 600; margin: 0.7rem 0 0.2rem; }
  .nli-bars { display: flex; flex-direction: column; gap: 0.3rem; margin: 0.4rem 0; max-width: 34rem; }
  .nli-bar { display: grid; grid-template-columns: 7rem 1fr 2.6rem; align-items: center; gap: 0.5rem; font-size: 0.85rem; }
  .nli-name { text-transform: capitalize; font-weight: 600; }
  .nli-track { height: 9px; border-radius: 5px; background: #7772; overflow: hidden; }
  .nli-track > i { display: block; height: 100%; }
  .nli-pct { font-family: var(--font-mono, monospace); text-align: right; }
  .nli-status { font-family: var(--font-mono, monospace); font-size: 0.9rem; margin: 0.4rem 0; }
`;
