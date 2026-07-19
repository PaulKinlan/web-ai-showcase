// Search worker for the client-side model explorer (route: /explore/).
//
// Implements the typed contract in search/search-protocol.ts on top of the repo's transport,
// lib/worker-protocol.js (per-request ids, transfer, streamed progress, AbortSignal cancellation,
// latest-wins channels, bounded backpressure, deterministic teardown). Everything here runs OFF the
// main thread (CLAUDE.md invariant 15): BM25 lexical scoring, brute-force int8 cosine, filtering,
// facets, explanation, AND the query embedding. The main thread only paints.
//
// Contract methods served: build · search · facets · explain · embedStatus · embedDownload.
//
// Index lifecycle (ADR §6):
//   • build() loads the prebuilt, deterministic artifact from search/index/ (index.json +
//     vectors.i8.bin) OR from IndexedDB when a matching schemaVersion+checksum is already cached
//     (source:"persisted-cache" ⇒ no re-fetch). A checksum mismatch or force rebuilds from the artifact.
//   • The embedder (Xenova/all-MiniLM-L6-v2, q8, off-main-thread) is auto-initialised WHEN ALREADY
//     CACHED (invariant 12); otherwise it stays "absent" and the user must explicitly embedDownload().
//     Lexical BM25 + every filter work fully WITHOUT it (semanticApplied:false surfaced honestly).
//
// modern-web-guidance retained + applied (ids): break-up-long-tasks + schedule-tasks-by-priority
// (build yields on a deadline; queries are latest-wins on channel "search"), identify-inp-causes +
// interactions-in-complex-layouts (all scoring here → the page's INP stays input-delay only),
// performance (offload > any main-thread frame slice), accessibility (worker feeds the a11y UI).

import { serveWorker, yieldToMain } from "../lib/worker-protocol.js";
import { TRANSFORMERS_URL } from "../lib/webai.js";
import { recordValidated, scanCachedFiles } from "../lib/model-cache.js";

const BASE = new URL("../", import.meta.url); // repo root (works under /web-ai-showcase/ base path)
const DIM = 384;
const K1 = 1.2;
const B = 0.75;
const EMBED_MODEL = "Xenova/all-MiniLM-L6-v2";
const EMBED_KEY = `transformers.js::${EMBED_MODEL}::q8`;

// ── IndexedDB persistence for the hydrated index (schema + checksum keyed) ─────────────────────────
const IDB_NAME = "web-ai-explore";
const IDB_STORE = "index";
const IDB_KEY = "search-index";
function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(IDB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(IDB_STORE, { keyPath: "key" });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
async function idbGet(key) {
  try {
    const db = await openDB();
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, "readonly");
      const r = tx.objectStore(IDB_STORE).get(key);
      r.onsuccess = () => resolve(r.result ?? null);
      r.onerror = () => reject(r.error);
    });
  } catch {
    return null;
  }
}
async function idbPut(rec) {
  try {
    const db = await openDB();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, "readwrite");
      tx.objectStore(IDB_STORE).put(rec);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch { /* private mode etc. — degrade to no-persistence */ }
}

// ── tokenizer — MUST match scripts/build-search-index.mjs exactly ──────────────────────────────────
const STOP = new Set(
  "the a an and or of to for with in on is are be this that it as at by from into over your you can"
    .split(" "),
);
function tokenize(text) {
  const out = [];
  for (const raw of String(text).toLowerCase().split(/[^a-z0-9]+/)) {
    if (raw.length < 2 || STOP.has(raw)) continue;
    out.push(raw);
  }
  return out;
}

// ── module state (hydrated index) ──────────────────────────────────────────────────────────────────
let DOCS = [];
let POSTINGS = null; // Map<term, Int32Array [docId,tf,...]>
let DF = null; // Map<term, docFreq>
let DOCLEN = null; // Float64Array
let AVGDL = 0;
let N = 0;
let VEC_I8 = null; // Int8Array(N*DIM)
let ALIASES = {};
let FAMILIES = {};
let FACET_VALUES = {};
let SCHEMA = 0;
let CHECKSUM = "";
let META = null;

// ── embedder state (query embedding, off the main thread) ─────────────────────────────────────────
const EMBED = { state: "absent", pipe: null, sizeMB: 25, cached: false };

async function fetchJson(path) {
  const r = await fetch(new URL(path, BASE));
  if (!r.ok) throw new Error(`fetch ${path} → ${r.status}`);
  return r.json();
}
async function fetchBuf(path) {
  const r = await fetch(new URL(path, BASE));
  if (!r.ok) throw new Error(`fetch ${path} → ${r.status}`);
  return r.arrayBuffer();
}

function hydrate(index, vectorsBuf) {
  DOCS = index.docs;
  N = index.lexical.N;
  DOCLEN = Float64Array.from(index.lexical.docLen);
  AVGDL = index.lexical.avgdl;
  DF = new Map();
  POSTINGS = new Map();
  const { terms, df, postings } = index.lexical;
  for (let i = 0; i < terms.length; i++) {
    DF.set(terms[i], df[i]);
    POSTINGS.set(terms[i], Int32Array.from(postings[i]));
  }
  VEC_I8 = new Int8Array(vectorsBuf);
  ALIASES = index.aliases || {};
  FAMILIES = index.families || {};
  FACET_VALUES = index.facetValues || {};
  SCHEMA = index.schemaVersion;
  CHECKSUM = index.checksum;
}

// ── build: load persisted (cache) or fetch prebuilt artifact, then persist + auto-init embedder ────
async function build(payload = {}, { onProgress, signal }) {
  const t0 = performance.now();
  META = await fetchJson("search/index/meta.json").catch(() => null);
  const expect = payload.expectChecksum || META?.checksum || null;

  // 1. Try the persisted (IndexedDB) copy unless a rebuild is forced.
  if (!payload.force) {
    onProgress?.({ status: "fetching", progress: 5 });
    const cached = await idbGet(IDB_KEY);
    if (
      cached && cached.schemaVersion === (META?.schemaVersion ?? cached.schemaVersion) &&
      (!expect || cached.checksum === expect)
    ) {
      onProgress?.({ status: "indexing", progress: 60 });
      hydrate(cached.index, cached.vectors);
      await maybeAutoInitEmbedder();
      onProgress?.({ status: "persisting", progress: 100 });
      return {
        result: buildResult("persisted-cache", performance.now() - t0),
      };
    }
  }

  // 2. Fresh build from the deterministic prebuilt artifact.
  onProgress?.({ status: "fetching", progress: 15 });
  const [index, vectorsBuf] = await Promise.all([
    fetchJson("search/index/index.json"),
    fetchBuf("search/index/vectors.i8.bin"),
  ]);
  if (signal?.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
  delete index._semHash; // internal generator field — not needed at runtime
  onProgress?.({ status: "indexing", progress: 55 });
  hydrate(index, vectorsBuf);
  await yieldToMain();

  onProgress?.({ status: "persisting", progress: 85 });
  await idbPut({
    key: IDB_KEY,
    schemaVersion: SCHEMA,
    checksum: CHECKSUM,
    index,
    vectors: vectorsBuf,
  });
  await maybeAutoInitEmbedder();
  onProgress?.({ status: "persisting", progress: 100 });
  return { result: buildResult("fresh-build", performance.now() - t0) };
}

function buildResult(source, buildMs) {
  return {
    schemaVersion: SCHEMA,
    checksum: CHECKSUM,
    docCount: N,
    terms: POSTINGS.size,
    indexBytes: META?.indexBytes ?? 0,
    vecBytes: VEC_I8?.byteLength ?? 0,
    source,
    buildMs: +buildMs.toFixed(1),
  };
}

// ── embedder lifecycle (invariant 12) ──────────────────────────────────────────────────────────────
async function maybeAutoInitEmbedder() {
  // Auto-initialise ONLY when the embedder is already cached+current (returning users). Never triggers
  // a large download implicitly — an absent embedder stays "absent" until embedDownload().
  try {
    const files = await scanCachedFiles(EMBED_MODEL);
    EMBED.cached = files.length > 0;
  } catch {
    EMBED.cached = false;
  }
  if (EMBED.cached && EMBED.state === "absent") {
    await loadEmbedder(null); // auto-init from cache; no user gesture, no network transfer of weights
  }
}

async function loadEmbedder(onProgress) {
  if (EMBED.state === "ready") return;
  EMBED.state = onProgress ? "downloading" : "initialising";
  try {
    const { pipeline, env } = await import(/* @vite-ignore */ TRANSFORMERS_URL);
    env.allowLocalModels = false;
    // Force wasm + q8 so the query embedding matches the prebuilt int8 doc vectors' distribution
    // (batch/dtype parity — see search/ARCHITECTURE.md) and stays deterministic across devices.
    EMBED.pipe = await pipeline("feature-extraction", EMBED_MODEL, {
      device: "wasm",
      dtype: "q8",
      progress_callback: (p) => {
        if (p?.status === "progress" && typeof p.progress === "number") {
          onProgress?.({
            status: "downloading",
            progress: p.progress,
            receivedBytes: p.loaded,
            totalBytes: p.total,
          });
        } else if (p?.status === "ready" || p?.status === "done") {
          onProgress?.({ status: "initialising", progress: 100 });
        }
      },
    });
    // Warm + confirm the output dimension before declaring ready.
    const probe = await EMBED.pipe("warmup", { pooling: "mean", normalize: true });
    if (probe?.data?.length !== DIM) {
      throw new Error(`embedder dim ${probe?.data?.length} != ${DIM}`);
    }
    EMBED.state = "ready";
    EMBED.cached = true;
    recordValidated({
      key: EMBED_KEY,
      modelId: EMBED_MODEL,
      runtime: "transformers.js",
      dtype: "q8",
    })
      .catch(() => {});
  } catch (err) {
    EMBED.state = "error";
    EMBED.pipe = null;
    throw err;
  }
}

async function embedQuery(q) {
  const out = await EMBED.pipe(q, { pooling: "mean", normalize: true });
  const f = out.data;
  const qi = new Int8Array(DIM);
  for (let d = 0; d < DIM; d++) qi[d] = Math.max(-127, Math.min(127, Math.round(f[d] * 127)));
  return qi;
}

function embedStatus() {
  return {
    result: {
      state: EMBED.state,
      modelId: EMBED_MODEL,
      sizeMB: EMBED.sizeMB,
      cached: EMBED.cached,
    },
  };
}

async function embedDownload(_payload, { onProgress }) {
  if (EMBED.state !== "ready") await loadEmbedder(onProgress);
  return {
    result: {
      state: EMBED.state,
      modelId: EMBED_MODEL,
      sizeMB: EMBED.sizeMB,
      cached: EMBED.cached,
    },
  };
}

// ── query parse + alias expansion ──────────────────────────────────────────────────────────────────
function expand(qTokens, expandAliases) {
  const base = new Set(qTokens);
  const added = new Set();
  if (expandAliases !== false) {
    for (const tok of qTokens) {
      const exp = ALIASES[tok];
      if (exp) { for (const e of exp) for (const t of tokenize(e)) if (!base.has(t)) added.add(t); }
    }
  }
  return { terms: [...base, ...added], aliasExpanded: [...added] };
}

// ── filtering — returns a Set of candidate docIds (null = all) + which filters each doc satisfied ──
function passesFilter(d, filters) {
  const hits = [];
  if (!filters) return { pass: true, hits };
  const inList = (val, list, key) => {
    if (!list || !list.length) return true;
    const ok = list.includes(val);
    if (ok) hits.push(`${key}:${val}`);
    return ok;
  };
  if (!inList(d.task, filters.task, "task")) return { pass: false, hits };
  if (!inList(d.modality, filters.modality, "modality")) return { pass: false, hits };
  if (!inList(d.license, filters.license, "license")) return { pass: false, hits };
  if (!inList(d.runtime, filters.runtime, "runtime")) return { pass: false, hits };
  if (!inList(d.backend, filters.backend, "backend")) return { pass: false, hits };
  if (!inList(d.status, filters.status, "status")) return { pass: false, hits };
  if (!inList(d.relKind, filters.relKind, "relKind")) return { pass: false, hits };
  if (!inList(d.tier, filters.tier, "tier")) return { pass: false, hits };
  if (typeof filters.sizeMinMB === "number" && d.sizeMB < filters.sizeMinMB) {
    return { pass: false, hits };
  }
  if (typeof filters.sizeMaxMB === "number" && d.sizeMB > filters.sizeMaxMB) {
    return { pass: false, hits };
  }
  if (typeof filters.sizeMinMB === "number" || typeof filters.sizeMaxMB === "number") {
    hits.push("size");
  }
  if (typeof filters.minQuality === "number" && d.qualityPercentile < filters.minQuality) {
    return { pass: false, hits };
  }
  if (typeof filters.minQuality === "number") hits.push("quality");
  if (filters.canonicalFamily && d.canonicalFamily !== filters.canonicalFamily) {
    return { pass: false, hits };
  }
  if (filters.canonicalFamily) hits.push(`family:${d.canonicalFamily}`);
  // device gate ("this-device"): only models this device can actually run.
  if (filters.device === "this-device") {
    if (d.requiresWebGPU && !DEVICE.webgpu) return { pass: false, hits };
    if (d.sizeMB > DEVICE.maxSizeMB) return { pass: false, hits };
    hits.push("device");
  }
  return { pass: true, hits };
}

// device capability (probed once on init; conservative memory ceiling)
const DEVICE = { webgpu: false, maxSizeMB: 4096 };
async function probeDevice() {
  try {
    if ("gpu" in navigator) DEVICE.webgpu = (await navigator.gpu.requestAdapter()) != null;
  } catch { /* no adapter */ }
  const mem = navigator.deviceMemory; // GiB, coarse
  if (typeof mem === "number") DEVICE.maxSizeMB = Math.max(256, mem * 1024 * 0.6);
}

// ── BM25 over a candidate set ───────────────────────────────────────────────────────────────────────
function bm25(qTerms, candidateSet) {
  const scores = new Map();
  const matchedByDoc = new Map();
  for (const term of qTerms) {
    const arr = POSTINGS.get(term);
    if (!arr) continue;
    const df = DF.get(term) || arr.length / 2;
    const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
    for (let p = 0; p < arr.length; p += 2) {
      const docId = arr[p], tf = arr[p + 1];
      if (candidateSet && !candidateSet.has(docId)) continue;
      const dl = DOCLEN[docId];
      const denom = tf + K1 * (1 - B + (B * dl) / AVGDL);
      scores.set(docId, (scores.get(docId) || 0) + idf * ((tf * (K1 + 1)) / denom));
      let mt = matchedByDoc.get(docId);
      if (!mt) matchedByDoc.set(docId, mt = new Set());
      mt.add(term);
    }
  }
  return { scores, matchedByDoc };
}

function cosineI8(qi8, candidateIds) {
  const out = new Map();
  for (const i of candidateIds) {
    let s = 0;
    const off = i * DIM;
    for (let d = 0; d < DIM; d++) s += qi8[d] * VEC_I8[off + d];
    out.set(i, s / (127 * 127)); // rough [-1,1]
  }
  return out;
}

function minMaxNorm(map) {
  let lo = Infinity, hi = -Infinity;
  for (const v of map.values()) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const span = hi - lo;
  const norm = new Map();
  for (const [k, v] of map) norm.set(k, span > 1e-9 ? (v - lo) / span : (map.size ? 1 : 0));
  return norm;
}

// ── search — hybrid, AFTER filters (contract: SearchResult) ────────────────────────────────────────
async function search(payload, { signal }) {
  const t0 = performance.now();
  const {
    q = "",
    filters,
    mode = "hybrid",
    alpha = 0.5,
    k = 24,
    cursor = 0,
    expandAliases = true,
  } = payload || {};

  // 1. Filter candidate set.
  const candidateSet = new Set();
  const filterHitsById = new Map();
  for (let i = 0; i < N; i++) {
    const { pass, hits } = passesFilter(DOCS[i], filters);
    if (pass) {
      candidateSet.add(i);
      filterHitsById.set(i, hits);
    }
  }

  const { terms: qTerms, aliasExpanded } = expand(tokenize(q), expandAliases);
  const wantSemantic = mode !== "lexical" && EMBED.state === "ready" && q.trim().length > 0;

  // 2. Lexical scores.
  const hasQuery = q.trim().length > 0;
  const { scores: rawBm25, matchedByDoc } = hasQuery
    ? bm25(qTerms, candidateSet)
    : { scores: new Map(), matchedByDoc: new Map() };

  // 3. Semantic scores (query embedding runs HERE, off the main thread). Per ADR §5 step 3, semantic
  // scores EVERY surviving (filtered) doc — not just lexical hits — so the intent model can surface
  // relevant models the keywords missed (typos, paraphrases). Brute-force int8 cosine over the whole
  // candidate set is ~1 ms (measured), so this is cheap. A relevance FLOOR keeps totals meaningful:
  // a doc counts as a semantic match only above SEM_FLOOR, so gibberish still yields zero.
  const SEM_FLOOR = 0.3;
  let rawSem = null;
  if (wantSemantic) {
    if (signal?.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
    const qi8 = await embedQuery(q);
    if (signal?.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
    rawSem = cosineI8(qi8, [...candidateSet]);
  }

  // 4. Combine. The RESULT SET is the union of lexical hits and above-floor semantic hits (mode-gated);
  // ranking blends min-max-normalised BM25 + cosine over that set.
  let ranked;
  if (!hasQuery) {
    // No query: rank by priority tier then quality percentile (a browsable default).
    const TIER_RANK = { high: 5, built: 4, medium: 3, low: 2, superseded: 1, blocked: 0 };
    ranked = [...candidateSet].map((
      id,
    ) => [id, (TIER_RANK[DOCS[id].tier] ?? 2) + DOCS[id].qualityPercentile]);
  } else if (mode === "lexical" || !rawSem) {
    ranked = [...rawBm25.entries()];
  } else {
    const nb = minMaxNorm(rawBm25);
    const ns = minMaxNorm(rawSem);
    // Members: lexical hits (any bm25) ∪ semantic hits above the floor. In pure-semantic mode, lexical
    // hits below the floor are dropped so results stay on-topic.
    const members = new Set();
    if (mode !== "semantic") { for (const id of rawBm25.keys()) members.add(id); }
    for (const [id, v] of rawSem) if (v >= SEM_FLOOR) members.add(id);
    if (mode === "semantic") {
      ranked = [...members].map((id) => [id, ns.get(id) ?? 0]);
    } else {
      ranked = [...members].map((
        id,
      ) => [id, alpha * (ns.get(id) ?? 0) + (1 - alpha) * (nb.get(id) ?? 0)]);
    }
  }
  ranked.sort((a, b) => b[1] - a[1]);

  const total = ranked.length;
  const page = ranked.slice(cursor, cursor + k);
  const hits = page.map(([id, score]) => {
    const d = DOCS[id];
    return {
      slug: d.slug,
      name: d.name,
      task: d.task,
      modality: d.modality,
      sizeMB: d.sizeMB,
      license: d.license,
      canonicalFamily: d.canonicalFamily,
      score: +score.toFixed(4),
      // Extra display fields the UI needs (contract SearchHit + honest badges).
      meta: {
        status: d.status,
        runtime: d.runtime,
        backend: d.backend,
        relKind: d.relKind,
        relLabel: d.relLabel,
        confidence: d.confidence,
        evalPending: d.evalPending,
        tier: d.tier,
        tierRationale: d.tierRationale,
        canonicalAlternative: d.canonicalAlternative,
        downloads: d.downloads,
        likes: d.likes,
        qualityPercentile: d.qualityPercentile,
        blurb: d.blurb,
        unlocks: d.unlocks,
        blockedReason: d.blockedReason,
        demoRoute: d.demoRoute,
        hfUrl: d.hfUrl,
        familyCount: FAMILIES[d.canonicalFamily]?.length ?? 1,
      },
      explain: {
        bm25: +(rawBm25.get(id) ?? 0).toFixed(4),
        semantic: rawSem ? +(rawSem.get(id) ?? 0).toFixed(4) : null,
        filterHits: filterHitsById.get(id) || [],
        matchedTerms: [...(matchedByDoc.get(id) || [])],
        aliasExpanded,
      },
    };
  });

  return {
    result: {
      hits,
      total,
      cursor: cursor + k < total ? cursor + k : null,
      mode,
      semanticApplied: !!rawSem,
      tookMs: +(performance.now() - t0).toFixed(1),
    },
  };
}

// ── facets — counts per filter value for the current query, drill-down style ───────────────────────
function facets(payload) {
  const { q = "", filters } = payload || {};
  const qTerms = q.trim() ? expand(tokenize(q), filters?.expandAliases !== false).terms : null;
  // Base candidate set from the query (lexical membership only — no scoring needed).
  let queryMembers = null;
  if (qTerms) {
    queryMembers = new Set();
    for (const term of qTerms) {
      const arr = POSTINGS.get(term);
      if (!arr) continue;
      for (let p = 0; p < arr.length; p += 2) queryMembers.add(arr[p]);
    }
  }
  const dims = [
    "task",
    "modality",
    "license",
    "runtime",
    "backend",
    "status",
    "relKind",
    "tier",
    "confidence",
  ];
  const counts = {};
  for (const dim of dims) counts[dim] = {};
  const sizeBucketsDef = [
    { label: "< 50 MB", min: 0, max: 50 },
    { label: "50–150 MB", min: 50, max: 150 },
    { label: "150–500 MB", min: 150, max: 500 },
    { label: "500 MB–2 GB", min: 500, max: 2000 },
    { label: "> 2 GB", min: 2000, max: Infinity },
  ].map((b) => ({ ...b, count: 0 }));

  for (let i = 0; i < N; i++) {
    if (queryMembers && !queryMembers.has(i)) continue;
    const d = DOCS[i];
    // For each dimension, count applying ALL filters EXCEPT that dimension (drill-down semantics).
    for (const dim of dims) {
      const others = filters ? { ...filters, [dim]: undefined } : undefined;
      if (passesFilter(d, others).pass) {
        const v = d[dim];
        if (v != null && v !== "") counts[dim][v] = (counts[dim][v] || 0) + 1;
      }
    }
    if (
      passesFilter(
        d,
        filters ? { ...filters, sizeMinMB: undefined, sizeMaxMB: undefined } : undefined,
      ).pass
    ) {
      for (const b of sizeBucketsDef) if (d.sizeMB >= b.min && d.sizeMB < b.max) b.count++;
    }
  }
  return {
    result: {
      task: counts.task,
      modality: counts.modality,
      license: counts.license,
      runtime: counts.runtime,
      backend: counts.backend,
      status: counts.status,
      relKind: counts.relKind,
      tier: counts.tier,
      confidence: counts.confidence,
      sizeBuckets: sizeBucketsDef.map((b) => ({ ...b, max: b.max === Infinity ? null : b.max })),
    },
  };
}

// ── explain — full match rationale for one hit ─────────────────────────────────────────────────────
async function explain(payload, { signal }) {
  const { slug, q = "", filters } = payload || {};
  const id = DOCS.findIndex((d) => d.slug === slug);
  if (id < 0) throw new Error(`unknown slug: ${slug}`);
  const { terms: qTerms, aliasExpanded } = expand(tokenize(q), filters?.expandAliases !== false);
  const { scores, matchedByDoc } = bm25(qTerms, new Set([id]));
  let semantic = null;
  if (EMBED.state === "ready" && q.trim()) {
    if (signal?.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
    const qi8 = await embedQuery(q);
    semantic = cosineI8(qi8, [id]).get(id) ?? 0;
  }
  const { hits } = passesFilter(DOCS[id], filters);
  return {
    result: {
      slug,
      bm25: +(scores.get(id) ?? 0).toFixed(4),
      semantic: semantic == null ? null : +semantic.toFixed(4),
      filterHits: hits,
      matchedTerms: [...(matchedByDoc.get(id) || [])],
      aliasExpanded,
    },
  };
}

serveWorker({
  async init() {
    await probeDevice();
  },
  methods: { build, search, facets, explain, embedStatus, embedDownload },
  onDispose() {
    try {
      EMBED.pipe?.dispose?.();
    } catch { /* ignore */ }
    EMBED.pipe = null;
  },
});
