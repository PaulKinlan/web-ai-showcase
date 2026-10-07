// Client for the EmbeddingGemma 2 pages. It owns the worker handshake for the THREE message kinds this
// model needs (text, media, re-load-with-more-encoders), the encoder-budget metadata that makes
// "selective encoder loading" honest, and a gallery renderer for image/video results.
//
// The pure maths (cosine, Matryoshka truncation, similarity matrix, PCA-2D, spherical k-means) and the
// matrix / projection / ranked-list renderers are NOT re-implemented here: they are the same algorithms
// the EmbeddingGemma-300m pages already use, and two copies of PCA would only be two places to drift.
// This module imports them from the sibling family's module. It never modifies that module.
import {
  cosine,
  EGEMMA_CSS,
  escapeHTML,
  kmeans,
  matryoshka,
  parseLines,
  pca2d,
  renderMatrix,
  renderProjection,
  renderRanked,
  simColor,
  simMatrix,
} from "../embeddinggemma/egemma.js";

export { cosine, escapeHTML, kmeans, matryoshka, parseLines, pca2d, renderMatrix, renderProjection, renderRanked, simColor, simMatrix };

export const MODELS_BASE = "/web-ai-showcase/models";
export const WORKERS = { egemma2: `${MODELS_BASE}/embeddinggemma-2/worker.js` };

export const MODEL_ID = "onnx-community/embeddinggemma-2-ONNX";

/**
 * The three selectively-loadable graphs, with the q4 download each one adds (decimal MB, measured from
 * the Hub file sizes) and the capability it buys. `text` is the always-on backbone.
 */
export const ENCODERS = {
  text: {
    label: "Text + code",
    params: "270M (130M transformer + 140M embedder)",
    q4MB: 175,
    adds: "Passages, queries, code, 100+ languages — 768-d vectors, 8192-token context.",
  },
  vision: {
    label: "Images + video frames",
    params: "170M",
    q4MB: 109,
    adds: "An image — or sampled video frames — lands in the same space as text, so text can rank images.",
  },
  audio: {
    label: "Audio",
    params: "300M",
    q4MB: 189,
    adds: "Mono 16 kHz speech and environmental sound, ~25 tokens per second of audio.",
  },
};

/** Total q4 download for a set of enabled encoders, in decimal MB. */
export function budgetMB(enabled) {
  return ["text", "vision", "audio"].reduce((sum, k) => sum + (enabled[k] ? ENCODERS[k].q4MB : 0), 0);
}

/** The one real trap on this family, kept in one place so every page can show it verbatim. */
export const WASM_BLOCKED_NOTE =
  "The q4/q8 exports will not run on the WASM execution provider: the graph's block-quantized " +
  "embedding lookup (GatherBlockQuantized) has no WASM kernel, so session creation fails. Measured in " +
  "headless Chrome. This demo therefore runs on WebGPU; the fp32 export does load on WASM but is 1085 MB " +
  "for the text encoder alone, which is not a phone-sized download.";

/**
 * Worker client. Protocol: `load` / `run` / `runMedia` / `unload` in; `progress` / `ready` / `result` /
 * `unloaded` / `error` out. Unlike the sibling embedders this client can RE-load with more encoders,
 * so `ready` is not a one-shot latch.
 */
export class EmbedClient2 {
  constructor(workerUrl = WORKERS.egemma2) {
    this.worker = new Worker(workerUrl, { type: "module" });
    this.device = "webgpu";
    this.dtype = "q4";
    this.modalities = { vision: false, audio: false };
    this.sessions = [];
    this._ready = false;
    this._loadWaiters = [];
    this._pending = new Map();
    this._id = 0;
    this.onProgress = null;
    this.worker.addEventListener("message", (e) => this._onMessage(e.data));
    this.worker.addEventListener("error", (e) => {
      const err = new Error(e.message || "Worker failed to start");
      this._fail(err);
    });
  }

  _fail(err) {
    for (const w of this._loadWaiters) w.reject(err);
    this._loadWaiters = [];
    for (const [, p] of this._pending) p.reject(err);
    this._pending.clear();
  }

  _onMessage(msg) {
    if (msg.type === "progress") {
      this.onProgress?.(msg.p);
    } else if (msg.type === "ready") {
      this._ready = true;
      this.device = msg.device;
      this.dtype = msg.dtype;
      this.modalities = msg.modalities;
      this.sessions = msg.sessions ?? [];
      for (const w of this._loadWaiters) w.resolve(msg);
      this._loadWaiters = [];
    } else if (msg.type === "unloaded") {
      this._ready = false;
      this.modalities = { vision: false, audio: false };
      this.sessions = [];
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
        this._fail(new Error(msg.message));
      }
    }
  }

  /** Load (or reload) with the requested encoder set. Resolves with { device, dtype, modalities, sessions }. */
  load(onProgress, { vision = false, audio = false, dtype = "q4", device = "webgpu" } = {}) {
    if (onProgress) this.onProgress = onProgress;
    this.worker.postMessage({ type: "load", vision, audio, dtype, device });
    return new Promise((resolve, reject) => {
      this._loadWaiters.push({ resolve, reject });
    });
  }

  /** Text → { embeddings: number[][] (768-d unit vectors), norms, dim, tokenCounts, mode, ms, device }. */
  embed(texts, mode = "document") {
    const id = ++this._id;
    return this._request(id, { type: "run", id, texts, mode });
  }

  /**
   * One inference for a whole experiment. Each item carries its own `{ text, mode }`, so a page can compare
   * six different task prefixes on the same pair in a single round-trip and get embeddings in item order.
   */
  embedBatch(items, label = "") {
    const id = ++this._id;
    return this._request(id, { type: "runBatch", id, items, label });
  }

  /**
   * Media → { embeddings, norms, dim, softTokens, ms, device }.
   * `images`/`videos` are samples (nested lists). `maxSoftTokens` moves the documented vision token budget
   * (70 / 140 / 280 / 560 / 1120 per image) — fewer soft tokens means less image detail but proportionally
   * less compute, which is visible as real latency on a software adapter.
   */
  embedMedia({ images = null, videos = null, text = null, maxSoftTokens = null } = {}) {
    const id = ++this._id;
    return this._request(id, { type: "runMedia", id, images, videos, text, maxSoftTokens });
  }

  _request(id, payload) {
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage(payload);
    });
  }
}

/** The engine every page on this family uses. */
export class EmbeddingGemma2Engine extends EmbedClient2 {}

/**
 * Render image (or video-frame) tiles with a cosine score against one query vector. Used by the
 * cross-modal rung, where an IMAGE is the retrieved item and the query is TEXT (or vice versa).
 */
export function renderGallery(container, items, { scoreLabel = "cosine" } = {}) {
  container.replaceChildren(...items.map((it, i) => {
    const card = document.createElement("figure");
    card.className = "gallery-card";
    const media = it.src
      ? Object.assign(document.createElement("img"), {
        src: it.src,
        alt: it.alt ?? it.label ?? `Result ${i + 1}`,
        loading: "lazy",
        decoding: "async",
        fetchpriority: i < 2 ? "high" : "auto",
      })
      : Object.assign(document.createElement("div"), { className: "gallery-missing", textContent: "no preview" });
    if (!it.src && it.preview) media.style.background = it.preview;
    const cap = document.createElement("figcaption");
    const label = document.createElement("span");
    label.className = "gallery-label";
    label.textContent = it.label ?? `#${i + 1}`;
    const score = document.createElement("span");
    score.className = "gallery-score";
    score.textContent = typeof it.score === "number" ? `${scoreLabel} ${it.score.toFixed(3)}` : "";
    cap.append(label, score);
    if (it.sub) {
      const sub = document.createElement("span");
      sub.className = "gallery-sub";
      sub.textContent = it.sub;
      cap.append(sub);
    }
    card.append(media, cap);
    return card;
  }));
}

/** A compact "which encoders are loaded" table — the honest budget readout for selective loading. */
export function renderEncoderBudget(container, enabled, totalMB) {
  const table = document.createElement("table");
  table.className = "inside-table encoder-table";
  const caption = document.createElement("caption");
  caption.textContent =
    "EmbeddingGemma 2 is a 270M text backbone plus two independently loadable encoders. Only the graphs " +
    "you enable are downloaded.";
  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  for (const label of ["Encoder", "Parameters", "q4 download", "State", "What it adds"]) {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = label;
    hr.append(th);
  }
  thead.append(hr);
  const body = document.createElement("tbody");
  for (const key of ["text", "vision", "audio"]) {
    const e = ENCODERS[key];
    const on = !!enabled[key];
    const tr = document.createElement("tr");
    tr.className = on ? "on" : "off";
    const th = document.createElement("th");
    th.scope = "row";
    th.textContent = e.label;
    const cells = [e.params, `${e.q4MB} MB`, on ? "loaded" : "not loaded", e.adds];
    tr.append(th, ...cells.map((text) => {
      const td = document.createElement("td");
      td.textContent = text;
      return td;
    }));
    body.append(tr);
  }
  table.append(caption, thead, body);
  const note = document.createElement("p");
  note.className = "ctx-note";
  note.textContent =
    `Loaded budget: ${totalMB} MB (q4). Enabling an encoder re-creates the session — an explicit, priced ` +
    `choice, not a free toggle.`;
  container.replaceChildren(table, note);
}

/** CSS the second family adds on top of the shared embedder styles. */
export const EGEMMA2_CSS = `${EGEMMA_CSS}
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: .7rem; margin-top: .6rem; }
.gallery-card { margin: 0; border: 1px solid var(--border); border-radius: 10px; overflow: hidden; background: var(--bg-raised); display: flex; flex-direction: column; }
.gallery-card img { display: block; inline-size: 100%; block-size: 120px; object-fit: cover; background: var(--bg-secondary); }
.gallery-missing { block-size: 120px; background: var(--bg-secondary); display: grid; place-items: center; font-size: .75rem; color: var(--muted); }
.gallery-card figcaption { padding: .4rem .5rem; display: flex; flex-direction: column; gap: .1rem; font-size: .78rem; }
.gallery-label { color: var(--color); }
.gallery-score { font-family: var(--font-mono); color: var(--accent); }
.gallery-sub { font-family: var(--font-mono); font-size: .72rem; color: var(--muted); }
.encoder-table caption { text-align: start; color: var(--muted); font-size: .8rem; padding-bottom: .4rem; }
.encoder-table tr.off td, .encoder-table tr.off th { opacity: .5; }
.encoder-table tr.on th[scope="row"]::after { content: " ●"; color: var(--good); }
.chip-row { display: flex; flex-wrap: wrap; gap: .4rem; }
.drop-guide { border: 2px dashed var(--border-strong); border-radius: 10px; padding: .8rem; text-align: center; font-size: .82rem; color: var(--muted); }
.drop-guide.hot { border-color: var(--accent); color: var(--color); }
`;
