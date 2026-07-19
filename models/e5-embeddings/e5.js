// Front-end helpers for the E5-small-v2 embedding pages. Keeps each page thin: it owns the E5 worker
// handshake (which carries the query:/passage: PREFIX KIND), and re-uses the GTE page's pure maths and
// renderers (cosine, similarity matrix, PCA-2D, matrix/projection/ranked renderers) so every embedding
// page in the showcase looks and computes the same way — one source of truth. All inference lives in the
// workers (off the main thread); the light client maths here runs on a handful of already-computed
// 384-d vectors.
//
// Why E5 is its OWN family, not a GTE/MiniLM skin: E5 REQUIRES asymmetric instruction prefixes —
// "query: " on searches, "passage: " on documents — because it was contrastively trained on
// (query, passage) pairs. GTE uses no prefix; BGE uses one query-only instruction; MiniLM uses none.
// The prefix is not decoration: drop it and retrieval degrades. This module makes the prefix a
// first-class, visible parameter.

import {
  cosine,
  EmbedClient,
  escapeHTML,
  GTE_CSS,
  MODEL_LABELS,
  parseLines,
  pca2d,
  renderMatrix,
  renderProjection,
  renderRanked,
  simMatrix,
  WORKERS,
} from "/web-ai-showcase/models/gte-embeddings/gte.js";

const WORKER_URL = "/web-ai-showcase/models/e5-embeddings/worker.js";

/** E5 client. Unlike the generic EmbedClient, embed() takes a prefix KIND ("query"|"passage"|"raw"). */
export class E5Engine {
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

  /**
   * Embed texts with the given prefix kind → { texts, kind, embeddings:number[][] (unit), norms, dim,
   * ms, device }. kind defaults to "passage" (documents); pass "query" for searches, "raw" to skip the
   * prefix entirely (for the See-inside demonstration only).
   */
  embed(texts, kind = "passage") {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run", id, texts, kind });
    });
  }
}

// Re-export the shared embedding helpers so E5 pages import everything from one module.
export {
  cosine,
  EmbedClient,
  escapeHTML,
  GTE_CSS,
  MODEL_LABELS,
  parseLines,
  pca2d,
  renderMatrix,
  renderProjection,
  renderRanked,
  simMatrix,
  WORKERS,
};
