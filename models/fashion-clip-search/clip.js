// Front-end helpers for the fashion-clip pages. Thin: owns the worker handshake + renderers.
// All inference (embeddings + cosine) runs in worker.js, off the main thread.

const WORKER_URL = "/web-ai-showcase/models/fashion-clip-search/worker.js";
export const CATALOG = [
  { url: "catalog/dog-outdoor.jpg", label: "a dog outdoors" },
  { url: "catalog/bicycle-object.jpg", label: "a bicycle" },
  { url: "catalog/city-street.jpg", label: "a city street" },
  { url: "catalog/food-pizza.jpg", label: "pizza" },
  { url: "catalog/food-sushi.jpg", label: "sushi" },
  { url: "catalog/obj-room.jpg", label: "a room" },
  { url: "catalog/mountain-landscape.jpg", label: "mountains" },
  { url: "catalog/night-city.jpg", label: "a city at night" },
];

export class ClipEngine {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
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
      this.device = msg.device;
      for (const w of this._loadWaiters) w.resolve(msg.device);
      this._loadWaiters = [];
    } else if (msg.type === "catalog" || msg.type === "result") {
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
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
      this.worker.postMessage({ type: "load" });
    });
  }

  embedCatalog(urls) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "embedCatalog", id, imageURLs: urls });
    });
  }

  search(query) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "search", id, query });
    });
  }
}

export const CLIP_CSS = `
.query-row { display: flex; gap: .5rem; flex-wrap: wrap; align-items: center; margin: .5rem 0; }
input.query { flex: 1 1 16rem; font: inherit; padding: .5rem .6rem; border-radius: 8px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); }
.chips { display: flex; flex-wrap: wrap; gap: .4rem; margin: .4rem 0; }
.chip { font: inherit; font-size: .78rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--color); cursor: pointer; }
.chip:hover, .chip:focus-visible { border-color: var(--accent); }
.results { display: grid; grid-template-columns: repeat(auto-fill, minmax(11rem, 1fr)); gap: .7rem; margin-top: .6rem; }
.result-card { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--bg-raised); }
.result-card img { inline-size: 100%; block-size: 8.5rem; object-fit: cover; display: block; }
.result-card .meta { padding: .4rem .5rem; font-size: .78rem; }
.result-card .score { font-family: var(--font-mono); }
.scorebar { block-size: .45rem; border-radius: 999px; background: var(--bg-secondary); margin-top: .25rem; overflow: hidden; }
.scorebar .fill { display: block; block-size: 100%; background: var(--accent); }
.resource-warning { border: 1px solid var(--warn, var(--border)); border-inline-start: 4px solid var(--warn, var(--accent));
  border-radius: var(--radius, 8px); background: var(--bg-raised); color: var(--color); padding: .7rem .8rem;
  margin: .9rem 0; font-size: .86rem; line-height: 1.55; }
.resource-warning strong { color: var(--warn, var(--color)); }
.result-card.rank1 { border-color: var(--good); }
.result-card .rank { font-family: var(--font-mono); font-size: .7rem; color: var(--muted); }
.readout { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--font-mono); font-size: .78rem; color: var(--muted); margin-top: .6rem; }
.readout b { color: var(--color); font-weight: 600; }
`;
