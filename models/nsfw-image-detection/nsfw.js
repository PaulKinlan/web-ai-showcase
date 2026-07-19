// Shared front-end helpers for the NSFW image-detection pages. Keeps each page thin: it owns the worker
// handshake, turns files/samples into data URLs, and renders the safe/NSFW verdict + probability bars.
// All inference happens off the main thread in worker.js.

const WORKER_URL = "/web-ai-showcase/models/nsfw-image-detection/worker.js";

export class NSFWEngine {
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

  /** Classify one image → { classes, sfw, nsfw, ms, device }. */
  classify(imageURL) {
    return this._call({ type: "run", image: imageURL });
  }

  /** Screen many images ([{key,url}]) → { items:[{key,sfw,nsfw,safe}], ms, device }. */
  screen(images, threshold) {
    return this._call({ type: "screen", images, threshold });
  }

  /** Safety gate → { safe, sfw, nsfw, ms, device }. */
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
 * Render the safe/NSFW verdict into `els`. `threshold` is the max NSFW probability tolerated before an
 * image is withheld. Verdict is SAFE when nsfw < threshold.
 */
export function renderVerdict(els, { sfw, nsfw }, threshold) {
  const safe = nsfw < threshold;
  els.label.textContent = safe ? "SAFE TO DISPLAY" : "WITHHOLD";
  els.label.className = "verdict-label " + (safe ? "safe" : "nsfw");
  els.sub.textContent = `nsfw ${(nsfw * 100).toFixed(1)}% ${safe ? "<" : "≥"} threshold ${
    threshold.toFixed(2)
  }`;
  els.fillSafe.style.inlineSize = (sfw * 100).toFixed(1) + "%";
  els.fillNsfw.style.inlineSize = (nsfw * 100).toFixed(1) + "%";
  return safe;
}

/** Render the two class probability bars ([{label,prob}]). */
export function renderBars(container, classes) {
  const sorted = [...classes].sort((a, b) => b.prob - a.prob);
  const top = sorted[0]?.label;
  container.replaceChildren(...sorted.map((c) => {
    const pct = (c.prob * 100).toFixed(1);
    const row = document.createElement("div");
    row.className = "bar-row" + (c.label === top ? " bar-top" : "");
    row.innerHTML = `
      <div class="bar-head"><span class="bar-label">${
      escapeHTML(c.label)
    }</span><span class="bar-val">${pct}%</span></div>
      <div class="bar-track" role="meter" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}"
           aria-label="${escapeHTML(c.label)}: ${pct} percent">
        <div class="bar-fill ${
      c.label === "nsfw" ? "fill-nsfw" : "fill-safe"
    }" style="inline-size:${pct}%"></div>
      </div>`;
    return row;
  }));
}

export const NSFW_CSS = `
.dropzone { border: 2px dashed var(--border-strong); border-radius: var(--radius); background: var(--bg-raised);
  padding: 1rem; text-align: center; cursor: pointer; transition: border-color .15s, background .15s; }
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
.verdict-label { font-family: var(--font-display); font-size: 1.6rem; }
.verdict-label.safe { color: var(--good); }
.verdict-label.nsfw { color: var(--bad); }
.verdict-sub { font-family: var(--font-mono); color: var(--muted); font-size: .82rem; }
.meter-dual { display: flex; block-size: .8rem; border: 1px solid var(--border); border-radius: 999px;
  overflow: hidden; margin-top: .5rem; max-inline-size: 420px; background: var(--bg-raised); }
.meter-safe { background: var(--good); block-size: 100%; }
.meter-nsfw { background: var(--bad); block-size: 100%; }
.meter-labels { display: flex; justify-content: space-between; max-inline-size: 420px;
  font-family: var(--font-mono); font-size: .72rem; color: var(--muted); margin-top: .2rem; }
.bars { display: flex; flex-direction: column; gap: .5rem; margin-top: .6rem; max-inline-size: 420px; }
.bar-head { display: flex; justify-content: space-between; gap: .5rem; font-size: .85rem; }
.bar-label { font-family: var(--font-body); text-transform: uppercase; letter-spacing: .03em; font-size: .8rem; }
.bar-val { font-family: var(--font-mono); color: var(--muted); white-space: nowrap; }
.bar-track { block-size: .7rem; background: var(--bg-raised); border: 1px solid var(--border);
  border-radius: 999px; overflow: hidden; margin-top: .15rem; }
.bar-fill { block-size: 100%; border-radius: 999px; transition: inline-size .3s ease; background: var(--muted); }
.bar-fill.fill-safe { background: var(--good); }
.bar-fill.fill-nsfw { background: var(--bad); }
.thresh-wrap { display: flex; flex-direction: column; gap: .25rem; font-size: .82rem; max-inline-size: 420px; margin-top: .4rem; }
.field-row { display: flex; flex-wrap: wrap; gap: .6rem; align-items: end; margin: .6rem 0; }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .78rem;
  color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
.chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.chip:hover { border-color: var(--accent); }
.inside-table { inline-size: 100%; border-collapse: collapse; font-size: .82rem; margin-top: .5rem; }
.inside-table th, .inside-table td { text-align: left; padding: .3rem .5rem;
  border-bottom: 1px solid var(--border); font-family: var(--font-mono); }
.inside-table th { color: var(--muted); font-weight: 600; }
.conf-meter { block-size: .9rem; border-radius: 999px; overflow: hidden; border: 1px solid var(--border);
  background: linear-gradient(to right, var(--good), var(--warn), var(--bad)); position: relative; margin: .3rem 0; max-inline-size: 420px; }
.conf-needle { position: absolute; top: -3px; bottom: -3px; inline-size: 3px; background: var(--color); border-radius: 2px; }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: .6rem; margin-top: .6rem; }
.gallery figure { margin: 0; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--bg-raised); }
.gallery img { inline-size: 100%; block-size: 90px; object-fit: cover; display: block; }
.gallery figcaption { padding: .3rem .4rem; font-family: var(--font-mono); font-size: .72rem; }
.gallery figure.safe { border-color: var(--good); }
.gallery figure.withheld { border-color: var(--bad); }
.gallery figure.withheld img { filter: blur(10px); }
.gallery .cap-verdict.safe { color: var(--good); }
.gallery .cap-verdict.withheld { color: var(--bad); }
`;
