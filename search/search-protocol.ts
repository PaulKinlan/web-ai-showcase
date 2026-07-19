// Typed worker message contract for the client-side model explorer (route: /explore/, not yet built).
//
// This is the CONTRACT, not the implementation. It layers concrete method payloads/results on top of
// the repo's transport envelope in `lib/worker-protocol.js` (WorkerClient / serveWorker), which already
// provides per-request ids, transfer, streamed progress, AbortSignal cancellation, latest-wins channels,
// bounded backpressure, and deterministic teardown. Nothing here re-invents that; it only names the
// methods the explorer worker serves and the shapes that flow through them.
//
// Copyable: import the runtime constants from `search-protocol.ts`, or hand the interfaces to any
// TS-aware editor. The worker validates `payload` shapes at the boundary; the client is typed here.

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// Channels (latest-wins keys passed as RequestOptions.channel). A newer request on a channel
// supersedes older in-flight/queued ones — their late responses are dropped by WorkerClient.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
export const CHANNEL = {
  /** Every keystroke/filter change routes here so only the latest query survives (stale-suppression). */
  SEARCH: "search",
  /** Index build / rebuild — a newer build supersedes an in-flight one. */
  BUILD: "build",
} as const;

export const METHOD = {
  BUILD: "build",
  SEARCH: "search",
  FACETS: "facets",
  EXPLAIN: "explain",
  EMBED_STATUS: "embedStatus",
  EMBED_DOWNLOAD: "embedDownload",
} as const;
export type Method = (typeof METHOD)[keyof typeof METHOD];

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// Filter model — mirrors models.json + inventory facets. All optional; omitted = no constraint.
// Serialisable to/from URL state (shareable deep links) 1:1.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
export interface Filters {
  task?: string[]; // e.g. ["automatic-speech-recognition", "text-generation"]
  modality?: ("text" | "vision" | "vision-language" | "audio" | "audio-language" | "code")[];
  license?: string[]; // SPDX-ish ids; "" bucket = unknown-license
  runtime?: ("transformers.js" | "webllm" | "onnxruntime-web" | "mediapipe")[];
  backend?: ("wasm" | "webgpu")[];
  status?: ("built" | "pending" | "blocked")[];
  /** Model asset size window in MB (inclusive). */
  sizeMinMB?: number;
  sizeMaxMB?: number;
  /** Device capability gate: only models this device can actually run (probed WebGPU + rough memory). */
  device?: "this-device" | "any";
  /** Quality floor from inventory signals (likes/downloads percentile bucket). */
  minQuality?: number; // 0..1
  /** Restrict to one canonical family (dedup key) — used by "more like this / same family". */
  canonicalFamily?: string;
}

export type SearchMode = "hybrid" | "lexical" | "semantic";

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// build — deterministic index generation from models.json (+ inventory metadata). Idempotent; a
// version/checksum lets the client skip a rebuild when the cached index still matches.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
export interface BuildPayload {
  /** Skip rebuild if a persisted index with this schema+checksum is already loaded. */
  expectChecksum?: string;
  /** Force a full rebuild even if a persisted index exists (e.g. after a data refresh). */
  force?: boolean;
}
export interface BuildResult {
  schemaVersion: number;
  checksum: string; // sha-256 of the normalised corpus → index cache key + staleness check
  docCount: number;
  terms: number;
  indexBytes: number;
  vecBytes: number;
  source: "fresh-build" | "persisted-cache";
  buildMs: number;
}
export interface BuildProgress {
  status: "fetching" | "indexing" | "embedding-index" | "persisting";
  progress: number; // 0..100
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// search — the hybrid query. Semantic scoring is used only when the embedding model is ready
// (EmbedState "ready"); otherwise the worker degrades to lexical+filters and says so in the result.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
export interface SearchPayload {
  q: string;
  filters?: Filters;
  mode?: SearchMode; // default "hybrid"
  /** Hybrid blend weight: 0 = pure BM25, 1 = pure semantic. Default ~0.5, tunable in UI. */
  alpha?: number;
  k?: number; // page size, default 24
  cursor?: number; // offset for "load more"
  /** Expand query via alias/canonical-family map before scoring (e.g. "whisper" → asr family). */
  expandAliases?: boolean;
}
export interface MatchExplanation {
  bm25: number;
  semantic: number | null; // null when semantic disabled/unavailable
  filterHits: string[]; // which filters this doc satisfied
  matchedTerms: string[]; // lexical terms that hit (for highlight)
  aliasExpanded?: string[]; // alias/family terms folded into the query that matched
}
export interface SearchHit {
  slug: string;
  name: string;
  task: string;
  modality: string;
  sizeMB: number;
  license: string;
  canonicalFamily: string;
  score: number;
  explain: MatchExplanation;
}
export interface SearchResult {
  hits: SearchHit[];
  total: number; // matches after filters (for "N results")
  cursor: number | null; // next offset, or null at end
  mode: SearchMode;
  semanticApplied: boolean; // false ⇒ degraded to lexical (embedding model absent/loading)
  tookMs: number;
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// facets — counts per filter value for the CURRENT query+filters (drives the filter UI badges).
// ─────────────────────────────────────────────────────────────────────────────────────────────────
export interface FacetsPayload {
  q?: string;
  filters?: Filters;
}
export interface FacetsResult {
  task: Record<string, number>;
  modality: Record<string, number>;
  license: Record<string, number>;
  runtime: Record<string, number>;
  backend: Record<string, number>;
  sizeBuckets: { label: string; min: number; max: number; count: number }[];
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// explain — full match explanation for one hit (opened detail / "why this matched").
// ─────────────────────────────────────────────────────────────────────────────────────────────────
export interface ExplainPayload {
  slug: string;
  q: string;
  filters?: Filters;
}
export type ExplainResult = MatchExplanation & { slug: string };

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// Embedding model lifecycle (invariant 12: auto-init if cached-current, else EXPLICIT download).
// The embedder (gte-small / all-MiniLM-L6-v2 via transformers.js) runs in THIS worker, off the main
// thread. Lexical search is fully usable while it is absent/loading.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
export type EmbedState =
  | "absent" // not cached — needs an explicit user-initiated download
  | "downloading"
  | "initialising"
  | "ready"
  | "unsupported" // e.g. WebGPU-only path with no adapter — honest fallback to lexical
  | "error";
export interface EmbedStatusResult {
  state: EmbedState;
  modelId: string;
  sizeMB: number;
  cached: boolean; // already validated in cache ⇒ auto-inits without a download
}
export interface EmbedDownloadProgress {
  status: "downloading" | "initialising";
  progress: number; // 0..100
  receivedBytes?: number;
  totalBytes?: number;
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// Method table the worker serves (payload → result). Progress-emitting methods noted.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
export interface SearchWorkerMethods {
  build(p: BuildPayload): BuildResult; // emits BuildProgress
  search(p: SearchPayload): SearchResult; // channel: CHANNEL.SEARCH (latest-wins)
  facets(p: FacetsPayload): FacetsResult; // channel: CHANNEL.SEARCH
  explain(p: ExplainPayload): ExplainResult;
  embedStatus(): EmbedStatusResult;
  embedDownload(): EmbedStatusResult; // emits EmbedDownloadProgress; user-initiated only
}

// Usage (main thread), copyable:
//
//   import { WorkerClient } from "/lib/worker-protocol.js";
//   import { METHOD, CHANNEL } from "/search/search-protocol.ts";
//   const client = new WorkerClient({ url: new URL("./explore-worker.js", import.meta.url), name: "explore" });
//   await client.ready;
//   await client.request(METHOD.BUILD, { }, { onProgress: renderBuildBar });
//   const ac = new AbortController();
//   const { result } = await client.request(
//     METHOD.SEARCH,
//     { q, filters, mode: "hybrid", alpha: 0.5, k: 24 },
//     { channel: CHANNEL.SEARCH, signal: ac.signal },   // latest keystroke wins; older auto-superseded
//   );
