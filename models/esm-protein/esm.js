// Front-end helpers for the ESM-2 protein page: the worker handshake and residue-track rendering.
// All inference lives in worker.js (off the main thread).

export class EsmEngine {
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
    } else if (msg.type === "predict" || msg.type === "scan") {
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
  /** Mask residue `pos` → { pos, truth, top:[{aa,prob}], ms }. */
  predict(seq, pos, topK = 6) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "predict", id, seq, pos, topK });
    });
  }
  /** Per-residue conservation (masked-marginal prob of the true residue) → { conf:number[], ms }. */
  scan(seq) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "scan", id, seq });
    });
  }
}

export function escapeHTML(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]),
  );
}

/** Keep only valid amino-acid letters, uppercase. */
export function cleanSeq(s) {
  return String(s).toUpperCase().replace(/[^ACDEFGHIKLMNPQRSTVWY]/g, "");
}

// Low confidence (variable) = amber, high (conserved) = teal.
export function confColor(c) {
  const h = 40 + c * 130; // 40 (amber) → 170 (teal)
  return `hsl(${h} 65% ${72 - c * 22}%)`;
}

/**
 * Render a protein sequence as a grid of clickable residue cells. If `conf` is given, each cell is tinted
 * by its conservation score. Calls onClick(index) when a residue is clicked.
 */
export function renderSequence(container, seq, conf, selected) {
  container.innerHTML = seq.split("").map((aa, i) => {
    const style = conf ? `background:${confColor(conf[i])}` : "";
    const title = conf
      ? `pos ${i + 1} · ${aa} · fit ${(conf[i] * 100) | 0}%`
      : `pos ${i + 1} · ${aa}`;
    return `<button class="esm-res${
      i === selected ? " sel" : ""
    }" data-i="${i}" style="${style}" title="${title}">${aa}</button>`;
  }).join("");
}

export const ESM_CSS = `
  .esm-field { margin: 0.5rem 0; }
  .esm-field label { display: block; font-size: 0.85rem; margin-bottom: 0.2rem; color: var(--muted, #888); }
  .esm-field textarea { width: 100%; box-sizing: border-box; padding: 0.55rem 0.65rem; border-radius: 8px; border: 1px solid #8886; font-family: var(--font-mono, monospace); font-size: 0.9rem; min-height: 3.5rem; resize: vertical; word-break: break-all; }
  .esm-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.5rem 0; }
  .esm-controls { display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: center; margin: 0.5rem 0; }
  .esm-seq { display: flex; flex-wrap: wrap; gap: 2px; margin: 0.7rem 0; font-family: var(--font-mono, monospace); }
  .esm-res { width: 1.4rem; height: 1.7rem; border: 1px solid #8883; border-radius: 4px; background: #7771; color: inherit; font: inherit; font-weight: 600; cursor: pointer; padding: 0; }
  .esm-res:hover { outline: 2px solid #2bb59a; }
  .esm-res.sel { outline: 2px solid #2bb59a; box-shadow: 0 0 0 2px #2bb59a55; }
  .esm-legend { display: flex; flex-wrap: wrap; gap: 0.4rem 1rem; font-size: 0.78rem; margin: 0.2rem 0; align-items: center; }
  .esm-legend i { display: inline-block; width: 5rem; height: 0.7rem; border-radius: 3px; background: linear-gradient(90deg, hsl(40 65% 72%), hsl(170 65% 50%)); vertical-align: middle; }
  .esm-pred { display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.5rem 0; }
  .esm-pbar { font-family: var(--font-mono, monospace); font-size: 0.85rem; padding: 0.15rem 0.5rem; border-radius: 6px; background: #2bb59a22; border: 1px solid #2bb59a55; }
  .esm-pbar.hit { background: #2bb59a44; border-color: #2bb59a; font-weight: 700; }
  .esm-status { font-family: var(--font-mono, monospace); font-size: 0.9rem; margin: 0.4rem 0; }
`;
