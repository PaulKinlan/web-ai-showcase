// Client for the EmbeddingGemma 2 pages. It owns the worker handshake for text embedding and selective
// configuration. Upstream describes multimodal capabilities (images, video, audio) in the same vector
// space, but this family is text-only by decision: media inference is unproven on this VM and no page
// exercises cross-modal retrieval.
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
 * The three upstream graphs, with the q4 download each one adds (decimal MB, measured from the Hub
 * file sizes) and card-documented capabilities. This showcase runs the text backbone only; vision and
 * audio are not loaded or verified here.
 */
export const ENCODERS = {
  text: {
    label: "Text + code",
    params: "270M (130M transformer + 140M embedder)",
    q4MB: 175,
    adds: "Passages, queries, code, 100+ languages — 768-d vectors (model card documents 8192-token context; exported config differs, unverified here).",
  },
  vision: {
    label: "Images + video frames",
    params: "170M",
    q4MB: 109,
    adds: "Documented by model card to map images/video frames into the same space as text; not loaded or verified in this showcase.",
  },
  audio: {
    label: "Audio",
    params: "300M",
    q4MB: 189,
    adds: "Documented by model card for mono 16 kHz speech and environmental sound (~25 tokens/s); not loaded or verified in this showcase.",
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
 * Worker client. Protocol: `load` / `run` / `runBatch` / `unload` in; `progress` / `ready` / `result` /
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
      if (!this._ready || this._loadWaiters.length > 0) {
        this.dispose(err);
      } else {
        this._fail(err);
      }
    });
  }

  /**
   * Shared teardown/disposal helper. Terminates the underlying worker, nulls the reference,
   * resets readiness/modalities, and rejects any pending waiters. Safe to call multiple times.
   */
  dispose(err) {
    if (this.worker) {
      try {
        this.worker.terminate();
      } catch {
        // ignore
      }
      this.worker = null;
    }
    this._ready = false;
    this.modalities = { vision: false, audio: false };
    this.sessions = [];
    this._fail(err || new Error("Worker disposed"));
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
        const err = new Error(msg.message);
        if (!this._ready || this._loadWaiters.length > 0) {
          this.dispose(err);
        } else {
          this._fail(err);
        }
      }
    }
  }

  /** Load (or reload) with the requested encoder set. Resolves with { device, dtype, modalities, sessions }. */
  async load(onProgress, { vision = false, audio = false, dtype = "q4", device = "webgpu" } = {}) {
    if (onProgress) this.onProgress = onProgress;
    if (!this.worker) {
      throw new Error("Worker is not available or has been disposed");
    }
    try {
      this.worker.postMessage({ type: "load", vision, audio, dtype, device });
      return await new Promise((resolve, reject) => {
        this._loadWaiters.push({ resolve, reject });
      });
    } catch (err) {
      this.dispose(err);
      throw err;
    }
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

  _request(id, payload) {
    return new Promise((resolve, reject) => {
      if (!this.worker) {
        reject(new Error("Worker is not available or has been disposed"));
        return;
      }
      this._pending.set(id, { resolve, reject });
      try {
        this.worker.postMessage(payload);
      } catch (err) {
        this._pending.delete(id);
        reject(err);
      }
    });
  }
}

/** The engine every page on this family uses. */
export class EmbeddingGemma2Engine extends EmbedClient2 {}

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
