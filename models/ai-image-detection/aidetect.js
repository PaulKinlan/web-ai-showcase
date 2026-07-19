// Shared front-end helpers for the AI-generated-image detection pages. Keeps each page thin: it owns
// the worker handshake, turns files/samples into data URLs, and renders the AI-generated/real verdict
// plus the class-probability bars. All inference happens off the main thread in worker.js.
//
// HONESTY NOTE (baked into the copy everywhere): this is an IMPERFECT signal, not proof. The model
// outputs a probability, never a certificate of origin. False positives (real photos flagged) and
// false negatives (AI images passed) both happen, and adversarial edits fool it. The UI frames the
// score as triage evidence — a reason to look closer — never a verdict.

const WORKER_URL = "/web-ai-showcase/models/ai-image-detection/worker.js";

// The SMOGY detector labels two classes. We normalise them to a consistent internal vocabulary so the
// UI never depends on the exact HF label strings.
export const AI_LABEL = "artificial"; // model's label for AI-generated / synthetic
export const REAL_LABEL = "human"; // model's label for a real, human-made photograph

export class AIDetectEngine {
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
    } else if (msg.type === "result" || msg.type === "screen" || msg.type === "gate") {
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

  /** Classify one image → { classes, ai, real, ms, device }. */
  classify(imageURL) {
    return this._call({ type: "run", image: imageURL });
  }

  /** Screen many images ([{key,url}]) → { items:[{key,ai,real,flagged}], ms, device }. */
  screen(images, threshold) {
    return this._call({ type: "screen", images, threshold });
  }

  /** Provenance gate → { flagged, ai, real, ms, device }. */
  gate(imageURL, threshold) {
    return this._call({ type: "gate", image: imageURL, threshold });
  }

  _call(payload) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...payload, id });
    });
  }
}

/** Read a File (from upload or drop) into a data URL usable by the worker. */
export function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/**
 * Render the AI-generated/real verdict into `els`. `threshold` is the minimum "AI-generated"
 * probability at which we FLAG an image for review. Flagged when ai >= threshold. The wording is
 * deliberately hedged — "likely AI-generated" / "likely real" — because the score is a signal, not
 * proof of origin.
 */
export function renderVerdict(els, { ai, real }, threshold) {
  const flagged = ai >= threshold;
  els.label.textContent = flagged ? "LIKELY AI-GENERATED" : "LIKELY REAL";
  els.label.className = "verdict-label " + (flagged ? "ai" : "real");
  els.sub.textContent = `AI-generated ${(ai * 100).toFixed(1)}% ${
    flagged ? "≥" : "<"
  } flag threshold ${threshold.toFixed(2)} — a signal to review, not proof`;
  els.fillReal.style.inlineSize = (real * 100).toFixed(1) + "%";
  els.fillAi.style.inlineSize = (ai * 100).toFixed(1) + "%";
  return flagged;
}

/** Render the two class-probability bars ([{label,prob}]). */
export function renderBars(container, classes) {
  const sorted = [...classes].sort((a, b) => b.prob - a.prob);
  const top = sorted[0]?.label;
  container.replaceChildren(...sorted.map((c) => {
    const pct = (c.prob * 100).toFixed(1);
    const nice = c.label === AI_LABEL
      ? "AI-generated"
      : c.label === REAL_LABEL
      ? "real photo"
      : c.label;
    const row = document.createElement("div");
    row.className = "bar-row" + (c.label === top ? " bar-top" : "");
    row.innerHTML = `
      <div class="bar-head"><span class="bar-label">${
      escapeHTML(nice)
    }</span><span class="bar-val">${pct}%</span></div>
      <div class="bar-track" role="meter" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}"
           aria-label="${escapeHTML(nice)}: ${pct} percent">
        <div class="bar-fill ${
      c.label === AI_LABEL ? "fill-ai" : "fill-real"
    }" style="inline-size:${pct}%"></div>
      </div>`;
    return row;
  }));
}

export const AIDETECT_CSS = `
.dropzone { border: 2px dashed var(--border-strong); border-radius: var(--radius); background: var(--bg-raised);
  padding: 1rem; text-align: center; cursor: pointer; transition: border-color .15s, background .15s;
  min-block-size: 44px; }
.dropzone.drag { border-color: var(--accent); background: var(--bg-secondary); }
.dropzone:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.preview-wrap { position: relative; display: inline-block; max-inline-size: 100%; }
.preview-img { max-inline-size: 100%; max-block-size: 320px; border-radius: 8px; display: block; }
.sample-strip { display: flex; gap: .5rem; flex-wrap: wrap; margin: .5rem 0; }
.sample-thumb { inline-size: 76px; block-size: 56px; object-fit: cover; border-radius: 6px;
  border: 2px solid transparent; cursor: pointer; padding: 0; }
.sample-thumb.active { border-color: var(--accent); }
.sample-thumb:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.verdict-row { display: flex; align-items: baseline; gap: .8rem; flex-wrap: wrap; margin-top: .6rem; }
.verdict-label { font-family: var(--font-display); font-size: 1.5rem; }
.verdict-label.real { color: var(--good); }
.verdict-label.ai { color: var(--warn); }
.verdict-sub { font-family: var(--font-mono); color: var(--muted); font-size: .8rem; }
.meter-dual { display: flex; block-size: .8rem; border: 1px solid var(--border); border-radius: 999px;
  overflow: hidden; margin-top: .5rem; max-inline-size: 420px; background: var(--bg-raised); }
.meter-real { background: var(--good); block-size: 100%; }
.meter-ai { background: var(--warn); block-size: 100%; }
.meter-labels { display: flex; justify-content: space-between; max-inline-size: 420px;
  font-family: var(--font-mono); font-size: .72rem; color: var(--muted); margin-top: .2rem; }
.bars { display: flex; flex-direction: column; gap: .5rem; margin-top: .6rem; max-inline-size: 420px; }
.bar-head { display: flex; justify-content: space-between; gap: .5rem; font-size: .85rem; }
.bar-label { font-family: var(--font-body); text-transform: uppercase; letter-spacing: .03em; font-size: .8rem; }
.bar-val { font-family: var(--font-mono); color: var(--muted); white-space: nowrap; }
.bar-track { block-size: .7rem; background: var(--bg-raised); border: 1px solid var(--border);
  border-radius: 999px; overflow: hidden; margin-top: .15rem; }
.bar-fill { block-size: 100%; border-radius: 999px; transition: inline-size .3s ease; background: var(--muted); }
.bar-fill.fill-real { background: var(--good); }
.bar-fill.fill-ai { background: var(--warn); }
.thresh-wrap { display: flex; flex-direction: column; gap: .25rem; font-size: .82rem; max-inline-size: 420px; margin-top: .4rem; }
.field-row { display: flex; flex-wrap: wrap; gap: .6rem; align-items: stretch; margin: .6rem 0; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .78rem;
  color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.caveat { border-inline-start: 3px solid var(--warn); background: var(--bg-raised); padding: .5rem .7rem;
  border-radius: 6px; font-size: .82rem; margin-top: .6rem; }
.inside-table { inline-size: 100%; border-collapse: collapse; font-size: .82rem; margin-top: .5rem; }
.inside-table th, .inside-table td { text-align: left; padding: .3rem .5rem;
  border-bottom: 1px solid var(--border); font-family: var(--font-mono); }
.inside-table th { color: var(--muted); font-weight: 600; }
.conf-meter { block-size: .9rem; border-radius: 999px; overflow: hidden; border: 1px solid var(--border);
  background: linear-gradient(to right, var(--good), var(--warn)); position: relative; margin: .3rem 0; max-inline-size: 420px; }
.conf-needle { position: absolute; top: -3px; bottom: -3px; inline-size: 3px; background: var(--color); border-radius: 2px; }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: .6rem; margin-top: .6rem; }
.gallery figure { margin: 0; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--bg-raised); }
.gallery img { inline-size: 100%; block-size: 90px; object-fit: cover; display: block; }
.gallery figcaption { padding: .3rem .4rem; font-family: var(--font-mono); font-size: .72rem; }
.gallery figure.flagged { border-color: var(--warn); }
.gallery figure.clear { border-color: var(--good); }
.gallery .cap-verdict.flagged { color: var(--warn); }
.gallery .cap-verdict.clear { color: var(--good); }
`;
