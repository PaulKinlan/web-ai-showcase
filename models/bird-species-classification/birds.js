// Front-end helpers for the Bird species classification page. Owns the worker handshake, turns an
// uploaded/dropped/sample image into a data/URL, and renders the top-k species as confidence bars. All
// inference lives in worker.js, off the main thread.

const WORKER_URL = "/web-ai-showcase/models/bird-species-classification/worker.js";

export class BirdEngine {
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
    if (msg.type === "progress") this.onProgress?.(msg.p);
    else if (msg.type === "ready") {
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
  classify(imageURL, topK = 5) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "classify", id, imageURL, topK });
    });
  }
}

export function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result);
    fr.onerror = reject;
    fr.readAsDataURL(file);
  });
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/** The labels are ALL-CAPS species names ("AMERICAN FLAMINGO"); title-case them for display. */
export function prettyLabel(label) {
  return String(label).toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Render top-k species as labelled probability bars, top first. */
export function renderBars(container, labels) {
  container.replaceChildren(...labels.map(({ label, score }, i) => {
    const pct = (score * 100).toFixed(1);
    const row = document.createElement("div");
    row.className = "bird-bar" + (i === 0 ? " top" : "");
    const name = document.createElement("span");
    name.className = "bird-name";
    name.textContent = prettyLabel(label);
    const track = document.createElement("span");
    track.className = "bird-track";
    const fill = document.createElement("i");
    fill.style.width = Math.max(1.5, score * 100) + "%";
    track.append(fill);
    const val = document.createElement("span");
    val.className = "bird-pct";
    val.textContent = pct + "%";
    row.append(name, track, val);
    return row;
  }));
}

export const BIRD_CSS = `
.bird-drop { border: 2px dashed var(--border); border-radius: 12px; padding: 1.1rem; text-align: center;
  background: var(--bg-raised); transition: border-color .15s, background .15s; }
.bird-drop.drag { border-color: var(--accent); background: color-mix(in srgb, var(--accent) 10%, transparent); }
.bird-tools { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; justify-content: center; margin: .3rem 0; }
.bird-btn { font: inherit; font-size: .85rem; padding: .35rem .8rem; border-radius: 8px; border: 1px solid var(--border);
  background: var(--bg-raised); color: var(--color); cursor: pointer; }
.bird-btn:hover:not([disabled]), .bird-btn:focus-visible { border-color: var(--accent); }
.bird-btn[disabled] { opacity: .5; cursor: default; }
.bird-hint { font-size: .82rem; color: var(--muted); margin: .3rem 0; }
.bird-samples { display: flex; flex-wrap: wrap; gap: .5rem; justify-content: center; margin-top: .5rem; }
.bird-sample { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; padding: 0; cursor: pointer; background: none; line-height: 0; }
.bird-sample img { display: block; height: 64px; width: 64px; object-fit: cover; }
.bird-sample:hover, .bird-sample:focus-visible { border-color: var(--accent); }
.bird-wrap { display: flex; flex-wrap: wrap; gap: 1.2rem; align-items: flex-start; margin-top: .8rem; }
.bird-preview { flex: none; }
.bird-preview img { max-width: 260px; max-height: 260px; border-radius: 10px; border: 1px solid var(--border); display: block; }
.bird-results { flex: 1; min-width: 15rem; }
.bird-guess { font-size: 1.3rem; font-weight: 700; margin: 0 0 .5rem; min-height: 1.6rem; }
.bird-bars { display: flex; flex-direction: column; gap: .3rem; max-width: 26rem; }
.bird-bar { display: grid; grid-template-columns: 1fr 7rem 3rem; align-items: center; gap: .5rem; font-size: .86rem; }
.bird-bar.top .bird-name { font-weight: 700; }
.bird-name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bird-track { height: 9px; border-radius: 5px; background: color-mix(in srgb, var(--color) 12%, transparent); overflow: hidden; }
.bird-track > i { display: block; height: 100%; border-radius: 5px; background: linear-gradient(90deg, #2bb59a, #4ac6e0); }
.bird-bar.top .bird-track > i { background: linear-gradient(90deg, #2bb59a, #7fe8d2); }
.bird-pct { font-family: var(--font-mono, monospace); text-align: right; font-size: .8rem; color: var(--muted); }
.bird-bar.top .bird-pct { color: var(--color); }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono, monospace);
  font-size: .78rem; color: var(--muted); margin-top: .7rem; }
.readout b { color: var(--color); font-weight: 600; }
`;
